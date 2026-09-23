#!/usr/bin/env python3
"""Run prepared answer-blind MLLM requests through DashScope's compatible API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
DEFAULT_DECODING = {"temperature": 0, "max_tokens": 64}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_payload(
    request: dict,
    package: Path,
    model: str,
    decoding: dict | None = None,
) -> dict:
    decoding = dict(DEFAULT_DECODING if decoding is None else decoding)
    content = []
    if request.get("scene_context") is not None:
        content.append({
            "type": "text",
            "text": (
                "Structured 3D scene context (coordinates are Pi3 model units):\n"
                + json.dumps(
                    request["scene_context"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            ),
        })
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
        **decoding,
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


def atomic_write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_metadata(
    requests_path: Path,
    package: Path,
    prepared: dict,
    model: str,
    base_url: str,
    decoding: dict,
) -> dict:
    package_manifest = package / "package_manifest.json"
    package_hash = sha256(package_manifest)
    if prepared.get("package_manifest_sha256") != package_hash:
        raise ValueError("prepared requests belong to a different package manifest")
    return {
        "schema_version": 1,
        "request_manifest_sha256": sha256(requests_path),
        "package_manifest_sha256": package_hash,
        "model": model,
        "provider": "DashScope OpenAI-compatible API",
        "base_url": base_url.rstrip("/"),
        "decoding": decoding,
        "protocol_version": prepared.get("protocol_version"),
    }


def validate_unique_request_ids(rows: list[dict], label: str) -> set[str]:
    identifiers = [str(item["request_id"]) for item in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate request IDs in {label}")
    return set(identifiers)


def load_resume_state(
    output: Path, metadata_path: Path, expected_metadata: dict
) -> tuple[list[dict], set[str]]:
    if output.exists() != metadata_path.exists():
        raise ValueError("resume requires both predictions and matching run metadata")
    if not output.exists():
        return [], set()
    actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if actual_metadata != expected_metadata:
        raise ValueError("resume metadata does not match the requested run")
    predictions = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("predictions output must contain a JSON list")
    return predictions, validate_unique_request_ids(predictions, "existing predictions")


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
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not configured")
    requests_path = args.requests.resolve()
    package = args.package.resolve()
    output = args.output.resolve()
    prepared = json.loads(requests_path.read_text(encoding="utf-8"))
    all_request_ids = validate_unique_request_ids(prepared["requests"], "request manifest")
    decoding = {"temperature": 0, "max_tokens": args.max_tokens}
    metadata = run_metadata(
        requests_path, package, prepared, args.model, args.base_url, decoding
    )
    selected = [
        item for item in prepared["requests"]
        if (args.condition is None or item["condition"] == args.condition)
        and (args.question_id is None or int(item["question_id"]) == args.question_id)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(output.suffix + ".run.json")
    predictions, completed = load_resume_state(output, metadata_path, metadata)
    unknown = completed - all_request_ids
    if unknown:
        raise ValueError(f"existing predictions contain unknown request IDs: {sorted(unknown)}")
    if not metadata_path.exists():
        atomic_write(metadata_path, metadata)
    for item in selected:
        if item["request_id"] in completed:
            continue
        payload = request_payload(item, package, args.model, decoding)
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
        atomic_write(output, predictions)
        print(f"Completed {item['request_id']}: {answer!r}")
    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
