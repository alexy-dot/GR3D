#!/usr/bin/env python3
"""Shared identity, coordinate-system, and metric-scale contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


OSI_ID_NAMESPACE = "osi_bench_numeric_id"
ASSOCIATION_METHODS = {
    "manual_box",
    "detector_plus_manual_confirmation",
    "ocr_tag_association",
}
METRIC_SOURCE_TYPES = {
    "known_distance",
    "sensor_ground_truth",
    "stereo_calibration",
    "externally_validated_metric_model",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_external_identity(value: object, *, required: bool = False) -> dict | None:
    if value is None:
        if required:
            raise ValueError("external_identity is required")
        return None
    if not isinstance(value, dict):
        raise ValueError("external_identity must be an object")

    namespace = str(value.get("namespace", "")).strip()
    scene_id = str(value.get("scene_id", "")).strip()
    object_id = str(value.get("object_id", "")).strip()
    method = str(value.get("association_method", "")).strip()
    verified = value.get("verified")
    if not namespace:
        raise ValueError("external_identity.namespace is required")
    if not scene_id.isdigit():
        raise ValueError("external_identity.scene_id must be numeric")
    if not object_id.isdigit():
        raise ValueError("external_identity.object_id must be numeric")
    if method not in ASSOCIATION_METHODS:
        raise ValueError(
            "external_identity.association_method must be one of "
            + ", ".join(sorted(ASSOCIATION_METHODS))
        )
    if not isinstance(verified, bool):
        raise ValueError("external_identity.verified must be boolean")

    normalized = {
        "namespace": namespace,
        "scene_id": scene_id.zfill(4),
        "object_id": object_id,
        "object_id_numeric": int(object_id),
        "association_method": method,
        "verified": verified,
    }
    if value.get("notes") is not None:
        normalized["notes"] = str(value["notes"])
    return normalized


def external_identity_key(identity: dict) -> tuple[str, str, int]:
    normalized = normalize_external_identity(identity, required=True)
    assert normalized is not None
    return (
        normalized["namespace"],
        normalized["scene_id"],
        normalized["object_id_numeric"],
    )


def pi3_coordinate_system(model: str) -> dict:
    if model not in {"pi3", "pi3x"}:
        raise ValueError(f"unsupported reconstruction model: {model}")
    return {
        "name": f"{model}_world",
        "units": f"{model}_model_units",
        "axis_values_are_meters": False,
        "metric_scale_validated": False,
        "meters_per_unit": None,
        "scale_provenance": None,
    }


def validate_coordinate_system(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("coordinate_system must be an object")
    name = str(value.get("name", "")).strip()
    units = str(value.get("units", "")).strip()
    if not name or not units:
        raise ValueError("coordinate_system requires name and units")
    metric = value.get("metric_scale_validated")
    axes_are_meters = value.get("axis_values_are_meters")
    if not isinstance(metric, bool) or not isinstance(axes_are_meters, bool):
        raise ValueError("coordinate_system metric declarations must be boolean")
    meters_per_unit = value.get("meters_per_unit")
    if metric:
        try:
            meters_per_unit = float(meters_per_unit)
        except (TypeError, ValueError) as error:
            raise ValueError("validated metric coordinates require meters_per_unit") from error
        if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be positive and finite")
    elif meters_per_unit is not None:
        raise ValueError("non-metric coordinates must not declare meters_per_unit")
    return {
        **value,
        "name": name,
        "units": units,
        "meters_per_unit": meters_per_unit,
    }


def load_metric_scale_calibration(
    path: Path,
    *,
    coordinate_system: dict,
    expected_pi3_manifest_sha256: str,
) -> dict:
    calibration_path = path.expanduser().resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "validated":
        raise ValueError("metric scale calibration must be schema v1 and validated")
    if payload.get("source_type") not in METRIC_SOURCE_TYPES:
        raise ValueError("metric scale calibration has an unsupported source_type")
    if payload.get("independent_of_benchmark_answers") is not True:
        raise ValueError("metric scale must be independent of benchmark answers")
    if payload.get("pi3_manifest_sha256") != expected_pi3_manifest_sha256:
        raise ValueError("metric scale calibration belongs to another Pi3 run")
    coordinate = validate_coordinate_system(coordinate_system)
    compatible_names = {
        coordinate["name"],
        coordinate.get("source_coordinate_system_name"),
    }
    compatible_names.discard(None)
    if payload.get("coordinate_system_name") not in compatible_names:
        raise ValueError("metric scale calibration coordinate system mismatch")

    source_artifact = Path(str(payload.get("source_artifact", ""))).expanduser()
    if not source_artifact.is_absolute():
        source_artifact = calibration_path.parent / source_artifact
    source_artifact = source_artifact.resolve()
    if not source_artifact.is_file():
        raise FileNotFoundError(source_artifact)
    if file_sha256(source_artifact) != payload.get("source_artifact_sha256"):
        raise ValueError("metric scale source artifact hash mismatch")
    try:
        meters_per_unit = float(payload["meters_per_model_unit"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("metric scale requires meters_per_model_unit") from error
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("meters_per_model_unit must be positive and finite")
    return {
        **payload,
        "meters_per_model_unit": meters_per_unit,
        "calibration_path": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
    }
