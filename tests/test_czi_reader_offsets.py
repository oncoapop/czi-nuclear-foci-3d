"""Pixel-correctness tests for the CZI subblock reader.

Added after `parse_subblock` was found to compute `data_offset` 16 bytes past
the first pixel byte, shifting every decoded plane by 8 uint16 pixels. Nothing
in the suite caught it, because the reader's offset arithmetic had no test at
all.

Two layers:

* `SubblockOffsetTests` builds a synthetic ZISRAW subblock payload and checks
  the offset arithmetic directly. Runs anywhere, needs no data.
* `ReaderMatchesReferenceTests` decodes real planes and compares them to
  aicspylibczi (an independent implementation wrapping Zeiss's libCZI).

Pointing the real-file comparison at data
-----------------------------------------
Set `CZI_TEST_MANIFEST` to either a directory holding .czi files, or a CSV
with a `source_path` column (the repo's sample_sheet.csv has one)::

    CZI_TEST_MANIFEST=/path/to/sample_sheet.csv \\
        python -m unittest discover -s tests -t tests

Without it the test falls back to sample_sheet.csv at the repo root, and
skips if neither is reachable. A skip here means the comparison did NOT run;
check the skip reason rather than reading a green suite as verification.
"""
from __future__ import annotations

import csv
import io
import os
import struct
import unittest
from pathlib import Path

from czi_foci import czi_reader

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ENV = "CZI_TEST_MANIFEST"


def build_directory_entry(dimensions: list[tuple[str, int, int, int]]) -> bytes:
    """Minimal valid DirectoryEntryDV: 32-byte head + 20 bytes per dimension."""
    head = bytearray(32)
    head[0:2] = b"DV"
    struct.pack_into("<i", head, 2, 1)              # pixel_type 1 -> Gray16/uint16
    struct.pack_into("<q", head, 6, 0)              # file_position
    struct.pack_into("<i", head, 14, 0)             # file_part
    struct.pack_into("<i", head, 18, 0)             # compression 0 -> uncompressed
    struct.pack_into("<i", head, 28, len(dimensions))
    body = b"".join(
        name.encode("ascii").ljust(4, b"\0") + struct.pack("<iifi", start, size, 0.0, stored)
        for name, start, size, stored in dimensions
    )
    return bytes(head) + body


class SubblockOffsetTests(unittest.TestCase):
    """data_offset must point at the first pixel byte, not past it."""

    DIMS = [("X", 0, 4, 4), ("Y", 0, 4, 4), ("C", 0, 1, 1)]

    def _parse(self, metadata_size: int, attachment_size: int, dimensions=None):
        dimensions = self.DIMS if dimensions is None else dimensions
        data_size = 32
        payload = struct.pack("<IIQ", metadata_size, attachment_size, data_size)
        payload += build_directory_entry(dimensions)
        payload = payload.ljust(256, b"\0")         # writer pads the leading block to 256
        payload += b"M" * metadata_size
        payload += b"P" * data_size
        payload += b"A" * attachment_size

        handle = io.BytesIO(b"\0" * 32 + payload)   # 32 bytes standing in for the segment header
        segment = czi_reader.Segment(
            offset=32 - czi_reader.SEGMENT_HEADER.size,
            identifier="ZISRAWSUBBLOCK",
            allocated_size=len(payload),
            used_size=len(payload),
        )
        return czi_reader.parse_subblock(handle, segment, include_stats=False), handle

    def test_data_offset_lands_on_the_first_pixel_byte(self) -> None:
        result, handle = self._parse(metadata_size=100, attachment_size=0)
        handle.seek(result["data_offset"])
        self.assertEqual(handle.read(result["data_size"]), b"P" * 32)

    def test_attachment_does_not_shift_the_data_offset(self) -> None:
        """The attachment trails the pixel data, so it must not move data_offset."""
        without, _ = self._parse(metadata_size=64, attachment_size=0)
        with_attachment, handle = self._parse(metadata_size=64, attachment_size=48)
        self.assertEqual(without["data_offset"], with_attachment["data_offset"])
        handle.seek(with_attachment["data_offset"])
        self.assertEqual(handle.read(with_attachment["data_size"]), b"P" * 32)

    def test_metadata_size_shifts_the_data_offset(self) -> None:
        """Metadata sits between the leading block and the pixel data."""
        small, _ = self._parse(metadata_size=0, attachment_size=0)
        large, handle = self._parse(metadata_size=128, attachment_size=0)
        self.assertEqual(large["data_offset"] - small["data_offset"], 128)
        handle.seek(large["data_offset"])
        self.assertEqual(handle.read(large["data_size"]), b"P" * 32)

    def test_leading_block_holds_at_the_256_byte_floor(self) -> None:
        """parse_directory_entry caps dimension_count at 10, so 16 + 32 + 20*10
        = 248 never exceeds the 256-byte floor. This pins that reasoning: if the
        cap is ever raised, this test documents what has to be rechecked."""
        dims = [(name, 0, 1, 1) for name in ("X", "Y", "C", "Z", "T", "S", "M", "H", "I", "V")]
        self.assertEqual(len(dims), 10)
        result, handle = self._parse(metadata_size=0, attachment_size=0, dimensions=dims)
        self.assertEqual(result["data_offset"], 32 + 256)
        handle.seek(result["data_offset"])
        self.assertEqual(handle.read(result["data_size"]), b"P" * 32)


def discover_czi(limit: int = 3) -> list[Path]:
    """Real CZI files from CZI_TEST_MANIFEST, else the repo's sample_sheet.csv."""
    sources: list[Path] = []
    configured = os.environ.get(MANIFEST_ENV)
    candidates = [Path(configured)] if configured else []
    candidates.append(REPO_ROOT / "sample_sheet.csv")

    for candidate in candidates:
        if candidate.is_dir():
            sources = sorted(candidate.glob("*.czi"))[:limit]
        elif candidate.is_file() and candidate.suffix.lower() == ".czi":
            sources = [candidate]
        elif candidate.is_file():
            with candidate.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    path = Path(row.get("source_path", ""))
                    if path.is_file():
                        sources.append(path)
                        if len(sources) >= limit:
                            break
        if sources:
            break
    return [p for p in sources if p.is_file()][:limit]


class ReaderMatchesReferenceTests(unittest.TestCase):
    """Decoded pixels must match an independent CZI implementation."""

    def setUp(self) -> None:
        # Setting CZI_TEST_MANIFEST is an explicit request for this comparison.
        # Skipping it then would report success for work that never happened,
        # which is how the data_offset bug survived review in the first place.
        # Unasked-for (no env var) it still skips, so CI without data stays green.
        requested = os.environ.get(MANIFEST_ENV)
        install_hint = "pip install -e '.[test]' -c constraints-analysis.txt"

        try:
            import aicspylibczi  # noqa: F401
        except ImportError:
            if requested:
                self.fail(
                    f"{MANIFEST_ENV} is set, so the real-file comparison was asked "
                    f"for, but aicspylibczi is missing and it cannot run. Install "
                    f"the pinned test extra: {install_hint}"
                )
            self.skipTest(f"aicspylibczi not installed; enable with: {install_hint}")

        self.samples = discover_czi()
        if not self.samples:
            if requested:
                self.fail(
                    f"{MANIFEST_ENV}={requested!r} was supplied but resolved to no "
                    "readable .czi file, so the comparison did not run"
                )
            self.skipTest(
                f"no CZI reachable; set {MANIFEST_ENV} to a .czi directory or a CSV "
                "with a source_path column"
            )

    def test_every_channel_matches_aicspylibczi(self) -> None:
        import numpy as np
        from aicspylibczi import CziFile

        for path in self.samples:
            inspection = czi_reader.inspect(
                path, include_stats=False, preview_dir=None, plane_dir=None
            )
            reference = CziFile(str(path))
            with self.subTest(czi=path.name):
                compared = 0
                for block in inspection["subblocks"]:
                    if block["compression_code"] != 0:
                        continue
                    dims = block["dimensions"]
                    channel = int(dims.get("C", {}).get("start", 0))
                    z = int(dims.get("Z", {}).get("start", 0))
                    height = int(dims["Y"]["stored_size"])
                    width = int(dims["X"]["stored_size"])

                    with path.open("rb") as handle:
                        handle.seek(block["data_offset"])
                        raw = czi_reader.read_exact(handle, block["data_size"])
                    dtype = np.dtype(block["dtype"]).newbyteorder("<")
                    mine = np.frombuffer(raw, dtype=dtype).reshape((height, width))

                    kwargs = {"C": channel}
                    if "Z" in dims:
                        kwargs["Z"] = z
                    theirs = np.squeeze(reference.read_image(**kwargs)[0])

                    self.assertTrue(
                        np.array_equal(mine, theirs),
                        f"{path.name} C={channel} Z={z} differs from aicspylibczi",
                    )
                    compared += 1
                self.assertGreater(compared, 0, f"{path.name}: no planes compared")


if __name__ == "__main__":
    unittest.main()
