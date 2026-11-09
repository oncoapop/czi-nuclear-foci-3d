"""File and metadata helpers for CZI foci analysis."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import czi_reader, ims_reader
from .config import AnalysisConfig


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    file_name: str
    source_path: str
    condition: str
    condition_code: str
    timepoint_hr: str
    replicate: str
    acquisition_date_folder: str


def pixel_size_um(
    metadata: dict,
    mode: str = "2d",
    resolution_level: int = 0,
) -> tuple[float, float, float]:
    """Voxel size in microns as (z, y, x).

    ``resolution_level`` applies to .ims files only and MUST match the level
    the pixel data was read at — the Imaris pyramid halves Y/X per level while
    ``DataSetInfo/Image`` always reports level-0 dimensions. CZI has no
    pyramid here, so the argument is ignored for that format.
    """
    if "has_dataset_info" in metadata:
        # .ims metadata shape (see ims_reader.inspect_ims).
        return ims_reader.pixel_size_um_ims(
            metadata, mode=mode, resolution_level=resolution_level
        )

    values: dict[str, float] = {}
    for item in metadata.get("scaling") or []:
        if item.get("Id") in {"X", "Y", "Z"} and item.get("Value"):
            values[item["Id"]] = float(item["Value"]) * 1_000_000.0

    z = values.get("Z", float("nan"))
    y = values.get("Y", float("nan"))
    x = values.get("X", float("nan"))

    if mode == "3d" and np.isnan(z):
        raise ValueError("Z voxel spacing is missing from CZI metadata but required for 3D mode. Please supply voxel sizes or use 2D mode.")

    return z, y, x


def channel_metadata(inspection: dict, index: int) -> str:
    metadata = inspection["metadata"]
    if "has_dataset_info" in metadata:
        return ims_reader.channel_metadata_ims(metadata, index)
    channels = metadata.get("channels") or []
    if index >= len(channels):
        return f"C{index}"
    channel = channels[index]
    parts = [f"C{index}", channel.get("Name", ""), channel.get("Fluor", "")]
    ex = channel.get("ExcitationWavelength")
    em = channel.get("EmissionWavelength")
    if ex and em:
        parts.append(f"Ex {ex} / Em {em} nm")
    return " | ".join(p for p in parts if p)


def inspect_czi(path: Path) -> dict:
    if path.suffix.lower() == ".ims":
        return ims_reader.inspect_ims(path)
    return czi_reader.inspect(path, include_stats=False, preview_dir=None, plane_dir=None)


def read_channel_arrays(
    path: Path,
    inspection: dict,
    mode: str = "2d",
    resolution_level: int = 0,
) -> dict[int, np.ndarray]:
    """Read per-channel pixel data.

    For .ims, pair every call with ``pixel_size_um(..., resolution_level=N)``
    using the same ``resolution_level``, or physical-unit measurements will be
    wrong by a factor of 2**N.
    """
    if path.suffix.lower() == ".ims":
        return ims_reader.read_channel_arrays_ims(
            path, inspection, mode=mode, resolution_level=resolution_level
        )

    arrays_by_channel: dict[int, dict[int, np.ndarray]] = {}
    with path.open("rb") as handle:
        for index, block in enumerate(inspection["subblocks"]):
            if block["compression_code"] != 0:
                raise ValueError(f"Compressed CZI subblock is not supported: {path}")
            if block["dtype"] not in {"uint8", "uint16"}:
                raise ValueError(f"Expected Gray8/Gray16 channel data, found {block['dtype']}: {path}")
            dims = block["dimensions"]
            channel = int(dims.get("C", {}).get("start", index))
            z = int(dims.get("Z", {}).get("start", 0))
            width = int(dims["X"]["stored_size"])
            height = int(dims["Y"]["stored_size"])
            dtype = np.dtype(block["dtype"]).newbyteorder("<")
            handle.seek(block["data_offset"])
            raw = czi_reader.read_exact(handle, block["data_size"])
            plane = np.frombuffer(raw, dtype=dtype).reshape((height, width))
            if channel not in arrays_by_channel:
                arrays_by_channel[channel] = {}
            arrays_by_channel[channel][z] = plane

    final_arrays: dict[int, np.ndarray] = {}
    for channel, z_planes in arrays_by_channel.items():
        if mode == "2d":
            # For 2D mode, if there are multiple Z planes, just take the first one or MIP?
            # Original code would just overwrite the array for the channel, keeping the last read block.
            # We will use the z=0 or the min z.
            # By default, to preserve 2D behaviour on single plane data, just take the first plane.
            # For true Z-stacks processed in 2D mode, this will silently drop planes, but preserves the original behaviour which was not stack-aware.
            min_z = min(z_planes.keys())
            final_arrays[channel] = z_planes[min_z]
        else:
            # 3D mode: assemble into a 3D array (Z, Y, X)
            sorted_z = sorted(z_planes.keys())
            stack = np.stack([z_planes[z] for z in sorted_z])
            final_arrays[channel] = stack

    return final_arrays


def load_condition_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    if "condition_code" not in rows[0] or "condition" not in rows[0]:
        raise ValueError("Condition map must include condition_code and condition columns")
    return {row["condition_code"]: row for row in rows}


def parse_sample_metadata(path: Path, config: AnalysisConfig, condition_map: dict[str, dict[str, str]]) -> SampleMetadata:
    match = re.match(config.filename.regex, path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename does not match configured regex: {path.name}")
    condition_code = match.group(config.filename.condition_group).strip()
    mapped = condition_map.get(condition_code, {})
    condition = mapped.get("condition", condition_code).strip()
    timepoint = match.group(config.filename.timepoint_group).strip()
    replicate = match.group(config.filename.replicate_group).strip().zfill(2)
    date_folder = path.parent.name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.parent.name) else ""
    safe_condition = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_") or "conditionUNK"
    safe_experiment = re.sub(r"[^A-Za-z0-9]+", "_", config.experiment_name).strip("_") or "experiment"
    sample_id = f"{safe_experiment}_{date_folder or 'dateUNK'}_{timepoint}hr_{safe_condition}_rep{replicate}"
    return SampleMetadata(
        sample_id=sample_id,
        file_name=path.name,
        source_path=str(path),
        condition=condition,
        condition_code=condition_code,
        timepoint_hr=timepoint,
        replicate=replicate,
        acquisition_date_folder=date_folder,
    )


def selected_files_from_root(root: Path, config: AnalysisConfig) -> list[Path]:
    pattern = re.compile(config.filename.regex, re.IGNORECASE)
    return sorted(path for path in root.rglob("*.czi") if pattern.match(path.name))


def selected_files_from_manifest(manifest: Path) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "source_path" not in rows[0]:
        raise ValueError("Manifest must contain a source_path column")
    return [Path(row["source_path"]) for row in rows]
