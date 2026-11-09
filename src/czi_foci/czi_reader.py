#!/usr/bin/env python3
"""Read CZI metadata and uncompressed image planes without changing the source."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


SEGMENT_HEADER = struct.Struct("<16sqq")
SUBBLOCK_FIXED_HEADER_SIZE = 16
DIRECTORY_ENTRY_SIZE = 256

PIXEL_TYPES = {
    0: ("Gray8", "uint8", 1),
    1: ("Gray16", "uint16", 2),
    2: ("Gray32Float", "float32", 4),
    3: ("Bgr24", "uint8", 3),
    4: ("Bgr48", "uint16", 6),
    8: ("Bgr96Float", "float32", 12),
    9: ("Bgra32", "uint8", 4),
    10: ("Gray64ComplexFloat", "complex64", 8),
    11: ("Bgr192ComplexFloat", "complex64", 24),
    12: ("Gray32", "uint32", 4),
    13: ("Gray64Float", "float64", 8),
}

COMPRESSION_TYPES = {
    0: "uncompressed",
    1: "JPEG",
    2: "LZW",
    4: "JPEG XR",
    5: "camera-specific",
    6: "system-specific",
}


@dataclass(frozen=True)
class Segment:
    offset: int
    identifier: str
    allocated_size: int
    used_size: int

    @property
    def payload_offset(self) -> int:
        return self.offset + SEGMENT_HEADER.size


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of file: requested {size}, received {len(data)}")
    return data


def iter_segments(handle: BinaryIO, file_size: int) -> Iterator[Segment]:
    offset = 0
    while offset + SEGMENT_HEADER.size <= file_size:
        handle.seek(offset)
        raw_identifier, allocated_size, used_size = SEGMENT_HEADER.unpack(
            read_exact(handle, SEGMENT_HEADER.size)
        )
        identifier = raw_identifier.rstrip(b"\0").decode("ascii", errors="replace")
        if not (identifier.startswith("ZISRAW") or identifier == "DELETED"):
            raise ValueError(f"Invalid CZI segment identifier at byte {offset}: {identifier!r}")
        if allocated_size < 0 or used_size < 0 or used_size > allocated_size:
            raise ValueError(f"Invalid segment sizes at byte {offset}")
        yield Segment(offset, identifier, allocated_size, used_size)
        offset += SEGMENT_HEADER.size + allocated_size

    if offset != file_size:
        raise ValueError(f"Trailing or truncated data: parser stopped at {offset} of {file_size} bytes")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_text(root: ET.Element, name: str) -> str | None:
    for element in root.iter():
        if local_name(element.tag) == name and element.text and element.text.strip():
            return element.text.strip()
    return None


def metadata_summary(root: ET.Element) -> dict[str, object]:
    image_fields = {}
    for name in (
        "SizeX",
        "SizeY",
        "SizeZ",
        "SizeC",
        "SizeT",
        "SizeS",
        "PixelType",
        "ComponentBitCount",
        "AcquisitionDateAndTime",
    ):
        value = first_text(root, name)
        if value is not None:
            image_fields[name] = value

    channels_by_id: dict[str, dict[str, str]] = {}
    for element in root.iter():
        if local_name(element.tag) != "Channel":
            continue
        channel = {key: value for key, value in element.attrib.items() if key in {"Id", "Name"}}
        for child in element.iter():
            name = local_name(child.tag)
            if name in {
                "Fluor",
                "ExcitationWavelength",
                "EmissionWavelength",
                "IlluminationType",
                "ContrastMethod",
            } and child.text and child.text.strip():
                channel[name] = child.text.strip()
        channel_id = channel.get("Id")
        if channel_id and channel_id.startswith("Channel:"):
            channels_by_id.setdefault(channel_id, {}).update(channel)

    def channel_sort_key(channel_id: str) -> tuple[int, str]:
        suffix = channel_id.rsplit(":", 1)[-1]
        return (int(suffix), channel_id) if suffix.isdigit() else (sys.maxsize, channel_id)

    channels = [channels_by_id[key] for key in sorted(channels_by_id, key=channel_sort_key)]

    scaling = []
    for element in root.iter():
        if local_name(element.tag) != "Distance":
            continue
        item = dict(element.attrib)
        for child in element:
            if child.text and child.text.strip():
                item[local_name(child.tag)] = child.text.strip()
        if item.get("Id") in {"X", "Y", "Z"}:
            scaling.append(item)

    objectives = []
    for element in root.iter():
        if local_name(element.tag) != "Objective":
            continue
        objective = dict(element.attrib)
        for child in element.iter():
            name = local_name(child.tag)
            if name in {"Name", "NominalMagnification", "LensNA", "Immersion"}:
                if child.text and child.text.strip():
                    objective[name] = child.text.strip()
        if objective:
            objectives.append(objective)

    return {
        "root": local_name(root.tag),
        "image": image_fields,
        "channels": channels,
        "scaling": scaling,
        "objectives": objectives,
    }


def parse_metadata(handle: BinaryIO, segment: Segment) -> tuple[ET.Element, int]:
    handle.seek(segment.payload_offset)
    xml_size, attachment_size = struct.unpack("<II", read_exact(handle, 8))
    handle.seek(segment.payload_offset + 256)
    xml_bytes = read_exact(handle, xml_size)
    xml_bytes = xml_bytes.rstrip(b"\0")
    root = ET.fromstring(xml_bytes)
    return root, attachment_size


def parse_directory_entry(raw: bytes) -> dict[str, object]:
    if raw[:2] != b"DV":
        raise ValueError(f"Unsupported CZI directory schema: {raw[:2]!r}")

    pixel_type = struct.unpack_from("<i", raw, 2)[0]
    file_position = struct.unpack_from("<q", raw, 6)[0]
    file_part = struct.unpack_from("<i", raw, 14)[0]
    compression = struct.unpack_from("<i", raw, 18)[0]
    pyramid_type = raw[22]
    dimension_count = struct.unpack_from("<i", raw, 28)[0]
    if not 0 <= dimension_count <= 10:
        raise ValueError(f"Implausible CZI dimension count: {dimension_count}")

    dimensions = {}
    offset = 32
    for _ in range(dimension_count):
        dimension = raw[offset : offset + 4].rstrip(b"\0").decode("ascii")
        start, size, start_coordinate, stored_size = struct.unpack_from("<iifi", raw, offset + 4)
        dimensions[dimension] = {
            "start": start,
            "size": size,
            "start_coordinate": start_coordinate,
            "stored_size": stored_size,
        }
        offset += 20

    pixel_name, dtype, bytes_per_pixel = PIXEL_TYPES.get(
        pixel_type, (f"unknown-{pixel_type}", None, None)
    )
    return {
        "pixel_type_code": pixel_type,
        "pixel_type": pixel_name,
        "dtype": dtype,
        "bytes_per_pixel": bytes_per_pixel,
        "file_position": file_position,
        "file_part": file_part,
        "compression_code": compression,
        "compression": COMPRESSION_TYPES.get(compression, f"unknown-{compression}"),
        "pyramid_type": pyramid_type,
        "dimensions": dimensions,
    }


def parse_subblock(handle: BinaryIO, segment: Segment, include_stats: bool) -> dict[str, object]:
    handle.seek(segment.payload_offset)
    metadata_size, attachment_size, data_size = struct.unpack(
        "<IIQ", read_exact(handle, SUBBLOCK_FIXED_HEADER_SIZE)
    )
    directory_raw = read_exact(handle, DIRECTORY_ENTRY_SIZE)
    entry = parse_directory_entry(directory_raw)
    # ZISRAW subblock layout: [sizes header | DirectoryEntryDV | fill] then
    # metadata, then pixel data, then attachment. Two errors were present here:
    # the leading block is padded to at least DIRECTORY_ENTRY_SIZE (256) bytes
    # INCLUDING the 16-byte sizes header, so adding SUBBLOCK_FIXED_HEADER_SIZE
    # on top double-counted it; and the attachment trails the pixel data, so it
    # must not shift the data offset at all. Together they pushed data_offset
    # 16 bytes past the first pixel byte, shifting every decoded plane by 8
    # uint16 pixels.
    #
    # The leading block is expressed from the directory entry's own size rather
    # than hard-coding 256, to match the spec. Note this does NOT currently
    # widen support: parse_directory_entry rejects dimension_count > 10, so the
    # computed size peaks at 16 + 32 + 20*10 = 248 and the 256-byte floor always
    # wins. The expression only becomes load-bearing if that cap is raised.
    directory_entry_size = 32 + 20 * len(entry["dimensions"])
    leading_block = max(DIRECTORY_ENTRY_SIZE, SUBBLOCK_FIXED_HEADER_SIZE + directory_entry_size)
    data_offset = segment.payload_offset + leading_block + metadata_size
    result = {
        "segment_offset": segment.offset,
        "metadata_size": metadata_size,
        "attachment_size": attachment_size,
        "data_offset": data_offset,
        "data_size": data_size,
        **entry,
    }

    if include_stats:
        if entry["compression_code"] != 0:
            result["statistics"] = {"status": "not computed for compressed data"}
        else:
            handle.seek(data_offset)
            pixel_bytes = read_exact(handle, data_size)
            expected = entry["bytes_per_pixel"]
            dimensions = entry["dimensions"]
            if expected is not None:
                for dimension in dimensions.values():
                    expected *= dimension["stored_size"]
            if expected is not None and expected != data_size:
                raise ValueError(
                    f"Subblock at {segment.offset} has {data_size} data bytes; expected {expected}"
                )
            result["sha256"] = hashlib.sha256(pixel_bytes).hexdigest()
            try:
                import numpy as np

                values = np.frombuffer(pixel_bytes, dtype=np.dtype(entry["dtype"]).newbyteorder("<"))
                result["statistics"] = {
                    "count": int(values.size),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "mean": float(values.mean()),
                }
            except (ImportError, TypeError):
                result["statistics"] = {"status": "NumPy unavailable for this pixel type"}

    return result


def export_preview(handle: BinaryIO, subblock: dict[str, object], output: Path) -> None:
    if subblock["compression_code"] != 0:
        raise ValueError("Preview export currently supports uncompressed subblocks only")
    if subblock["pixel_type"] not in {"Gray8", "Gray16"}:
        raise ValueError(f"Preview export does not support {subblock['pixel_type']}")

    dimensions = subblock["dimensions"]
    width = dimensions["X"]["stored_size"]
    height = dimensions["Y"]["stored_size"]
    handle.seek(subblock["data_offset"])
    pixel_bytes = read_exact(handle, subblock["data_size"])

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --preview-dir") from exc

    mode = "L" if subblock["pixel_type"] == "Gray8" else "I;16"
    image = Image.frombytes(mode, (width, height), pixel_bytes)
    image.thumbnail((512, 512))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing preview: {output}")
    image.save(output, format="PNG")


def export_plane(handle: BinaryIO, subblock: dict[str, object], output: Path) -> None:
    if subblock["compression_code"] != 0:
        raise ValueError("Plane export currently supports uncompressed subblocks only")
    if subblock["pixel_type"] not in {"Gray8", "Gray16"}:
        raise ValueError(f"Plane export does not support {subblock['pixel_type']}")

    dimensions = subblock["dimensions"]
    width = dimensions["X"]["stored_size"]
    height = dimensions["Y"]["stored_size"]
    handle.seek(subblock["data_offset"])
    pixel_bytes = read_exact(handle, subblock["data_size"])

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --plane-dir") from exc

    mode = "L" if subblock["pixel_type"] == "Gray8" else "I;16"
    image = Image.frombytes(mode, (width, height), pixel_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing plane: {output}")
    image.save(output, format="TIFF", compression="raw")


def inspect(
    path: Path,
    include_stats: bool,
    preview_dir: Path | None,
    plane_dir: Path | None = None,
) -> dict[str, object]:
    file_size = path.stat().st_size
    result: dict[str, object] = {
        "source": str(path),
        "source_size_bytes": file_size,
        "source_sha256": None,
        "metadata": None,
        "subblocks": [],
        "segment_counts": {},
    }

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        result["source_sha256"] = digest.hexdigest()

        segments = list(iter_segments(handle, file_size))
        counts: dict[str, int] = {}
        for segment in segments:
            counts[segment.identifier] = counts.get(segment.identifier, 0) + 1
            if segment.identifier == "ZISRAWMETADATA":
                root, attachment_size = parse_metadata(handle, segment)
                result["metadata"] = {
                    **metadata_summary(root),
                    "attachment_size": attachment_size,
                }
            elif segment.identifier == "ZISRAWSUBBLOCK":
                result["subblocks"].append(parse_subblock(handle, segment, include_stats))
        result["segment_counts"] = counts

        if preview_dir is not None:
            previews = []
            for index, subblock in enumerate(result["subblocks"]):
                channel = subblock["dimensions"].get("C", {}).get("start", index)
                output = preview_dir / f"channel_{channel}_subblock_{index}.png"
                export_preview(handle, subblock, output)
                previews.append(str(output))
            result["previews"] = previews

        if plane_dir is not None:
            planes = []
            for index, subblock in enumerate(result["subblocks"]):
                channel = subblock["dimensions"].get("C", {}).get("start", index)
                output = plane_dir / f"channel_{channel}_subblock_{index}.tif"
                export_plane(handle, subblock, output)
                planes.append(str(output))
            result["planes"] = planes

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("czi", type=Path, help="CZI file to inspect read-only")
    parser.add_argument("--stats", action="store_true", help="Compute per-subblock checksums and statistics")
    parser.add_argument("--preview-dir", type=Path, help="Export non-overwriting 512 px PNG previews")
    parser.add_argument("--plane-dir", type=Path, help="Export non-overwriting full-resolution TIFF planes")
    parser.add_argument("--json-output", type=Path, help="Write the summary JSON without overwriting")
    args = parser.parse_args()

    if not args.czi.is_file():
        parser.error(f"CZI file not found: {args.czi}")
    if args.json_output and args.json_output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.json_output}")

    try:
        result = inspect(args.czi, args.stats, args.preview_dir, args.plane_dir)
    except (OSError, ValueError, ET.ParseError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
