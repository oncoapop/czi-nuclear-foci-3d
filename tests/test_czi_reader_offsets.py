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

Without it the test falls back to sample_sheet.csv at the repo root, and skips
if neither is reachable. **Set it and that fallback is off**: a manifest naming
files that cannot be resolved must fail, not quietly compare some other
dataset. Review found the earlier version doing exactly that -- a deliberately
dead manifest still compared three files from sample_sheet.csv, and the
verification that was supposed to prove fail-closed behaviour had passed only
because the worktree it ran in had no sample sheet.

A skip here means the comparison did NOT run; check the skip reason rather than
reading a green suite as verification.
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
    """Real CZI files from CZI_TEST_MANIFEST, else the repo's sample_sheet.csv.

    Setting the environment variable disables the sample_sheet.csv fallback.
    An explicitly supplied manifest that resolves to nothing has to come back
    empty so the caller can fail; silently substituting a different dataset
    would report a comparison against files the caller never asked for.
    """
    sources: list[Path] = []
    configured = os.environ.get(MANIFEST_ENV)
    candidates = (
        [Path(configured)] if configured else [REPO_ROOT / "sample_sheet.csv"]
    )

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
            # CziFile(str) opens a handle internally and exposes no close(), so
            # every comparison file leaked one and emitted a ResourceWarning.
            # Constructing it from a handle we own puts the lifetime back
            # under `with`.
            with path.open("rb") as stream, self.subTest(czi=path.name):
                reference = CziFile(stream)
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


class ManifestFallbackTests(unittest.TestCase):
    """An explicitly supplied manifest must never fall back to sample_sheet.csv.

    Regression test. `discover_czi` used to append the repo's sample_sheet.csv
    to the candidate list unconditionally, so `CZI_TEST_MANIFEST` pointing at a
    nonexistent path still resolved three files from the sample sheet and the
    comparison ran against a dataset nobody asked for. The fail-closed check
    added alongside it missed this because it ran in a worktree with no sample
    sheet, leaving the fallback branch unexercised.

    Builds its own fake repo root, so it needs no acquisition volume and is
    unaffected by whether this checkout has a sample sheet.
    """

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        decoy = root / "fallback.czi"
        decoy.write_bytes(b"")                      # only has to exist
        with (root / "sample_sheet.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_path"])
            writer.writeheader()
            writer.writerow({"source_path": str(decoy)})

        self.root = root
        self.decoy = decoy
        self._saved_root = globals()["REPO_ROOT"]
        globals()["REPO_ROOT"] = root
        self._saved_env = os.environ.get(MANIFEST_ENV)
        os.environ.pop(MANIFEST_ENV, None)

    def tearDown(self) -> None:
        globals()["REPO_ROOT"] = self._saved_root
        if self._saved_env is None:
            os.environ.pop(MANIFEST_ENV, None)
        else:
            os.environ[MANIFEST_ENV] = self._saved_env
        self._tmp.cleanup()

    def test_fallback_is_used_when_the_variable_is_unset(self) -> None:
        """Guards the test below: the fallback has to be live for it to mean
        anything. Without this, a broken sample sheet would make the
        suppression test pass vacuously -- the exact failure being fixed."""
        self.assertEqual(discover_czi(), [self.decoy])

    def test_dead_manifest_does_not_fall_back(self) -> None:
        os.environ[MANIFEST_ENV] = str(self.root / "no-such-manifest.csv")
        self.assertEqual(
            discover_czi(), [],
            "a dead CZI_TEST_MANIFEST fell back to sample_sheet.csv; the "
            "comparison would have run against files the caller never named",
        )

    def test_manifest_naming_only_unreadable_rows_does_not_fall_back(self) -> None:
        """A manifest that exists but whose rows point nowhere is the same
        failure wearing a different hat."""
        dangling = self.root / "dangling.csv"
        with dangling.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_path"])
            writer.writeheader()
            writer.writerow({"source_path": str(self.root / "missing.czi")})
        os.environ[MANIFEST_ENV] = str(dangling)
        self.assertEqual(discover_czi(), [])

    def test_reader_comparison_fails_rather_than_skips_on_a_dead_manifest(self) -> None:
        """End to end: the suite must go red, not green-with-a-skip."""
        os.environ[MANIFEST_ENV] = str(self.root / "no-such-manifest.csv")
        case = ReaderMatchesReferenceTests("test_every_channel_matches_aicspylibczi")
        with self.assertRaises(case.failureException):
            case.setUp()


if __name__ == "__main__":
    unittest.main()
