#!/usr/bin/env python3
"""Run prepared answer-blind MLLM requests through DashScope's compatible API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_payload(request: dict, package: Path, model: str) -> dict:
    content = []
    for index, image in enumerate(request["images"], 1):
        label = f"Image {index}: {image['role']}"
        if image.get("timestamp_seconds") is not None:
            label += f", timestamp {image['timestamp_seconds']} seconds"
        if image.get("scene_instance_id"):
            label += f", scene ID {image['scene_instance_id']}"
        content.extend([
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": data_url(package / image["path"])}},
        ])
    question = request["question"]
    if request.get("options"):
        question += "\nOptions:\n" + "\n".join(request["options"])
    content.append({"type": "text", "text": question})
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": content},
        ],
    }


def call_api(
    payload: dict, api_key: str, base_url: str, timeout: int, retries: int = 3
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            if response.ok:
                return response.json()
            if response.status_code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"DashScope HTTP {response.status_code}: {response.text}"
                )
            error = RuntimeError(
                f"DashScope HTTP {response.status_code}: {response.text}"
            )
        except requests.RequestException as exc:
            error = exc
        if attempt == retries:
            raise RuntimeError(
                f"DashScope request failed after {retries + 1} attempts: {error}"
            ) from error
        time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def atomic_write(path: Path, payload: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="qwen3-vl-32b-instruct")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--condition")
    parser.add_argument("--question-id", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not configured")
    prepared = json.loads(args.requests.read_text(encoding="utf-8"))
    selected = [
        item for item in prepared["requests"]
        if (args.condition is None or item["condition"] == args.condition)
        and (args.question_id is None or int(item["question_id"]) == args.question_id)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = json.loads(args.output.read_text(encoding="utf-8")) if args.output.is_file() else []
    completed = {item["request_id"] for item in predictions}
    for item in selected:
        if item["request_id"] in completed:
            continue
        payload = request_payload(item, args.package.resolve(), args.model)
        started = time.monotonic()
        response = call_api(
            payload, api_key, args.base_url, args.timeout, retries=args.retries
        )
        answer = response["choices"][0]["message"]["content"]
        predictions.append({
            "request_id": item["request_id"],
            "answer": answer,
            "model_requested": args.model,
            "model_returned": response.get("model"),
            "response_id": response.get("id"),
            "usage": response.get("usage"),
            "latency_seconds": round(time.monotonic() - started, 3),
            "raw_response": response,
        })
        atomic_write(args.output, predictions)
        print(f"Completed {item['request_id']}: {answer!r}")
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
