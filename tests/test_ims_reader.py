from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from czi_foci.io import inspect_czi, pixel_size_um, read_channel_arrays
from czi_foci.ims_reader import (
    _decode_attr,
    inspect_ims,
    pixel_size_um_ims,
    read_channel_arrays_ims,
    read_channel_planes_ims,
    resolve_channel_indices,
)


def _char_array(value: str) -> np.ndarray:
    return np.array(list(value), dtype="S1")


def _write_image_attrs(group: h5py.Group, x: int, y: int, z: int, ext: dict[str, float]) -> None:
    group.attrs["X"] = _char_array(str(x))
    group.attrs["Y"] = _char_array(str(y))
    group.attrs["Z"] = _char_array(str(z))
    for key, value in ext.items():
        group.attrs[key] = _char_array(str(value))
    group.attrs["NumericalAperture"] = _char_array("1.2")


def _write_channel_attrs(group: h5py.Group, name: str, em: str, ex: str) -> None:
    group.attrs["Name"] = _char_array(name)
    group.attrs["LSMEmissionWavelength"] = _char_array(em)
    group.attrs["LSMExcitationWavelength"] = _char_array(ex)
    group.attrs["ColorRange"] = _char_array("0.000 4095.000")


def _make_ims(
    path: Path,
    true_z: int,
    padded_z: int,
    y: int = 32,
    x: int = 32,
    include_dataset_info: bool = True,
    include_data: bool = True,
    ext2: tuple[float, float] = (100.0, 105.0),
) -> None:
    with h5py.File(path, "w") as f:
        dataset = f.create_group("DataSet")
        for level, scale in enumerate([1, 2, 4, 8]):
            rl = dataset.create_group(f"ResolutionLevel {level}")
            tp0 = rl.create_group("TimePoint 0")
            for c in range(3):
                grp = tp0.create_group(f"Channel {c}")
                # Real .ims files declare the per-level dimensions here; the
                # reader uses them so it never has to assume the halving rule.
                grp.attrs["ImageSizeX"] = _char_array(str(max(1, x // scale)))
                grp.attrs["ImageSizeY"] = _char_array(str(max(1, y // scale)))
                grp.attrs["ImageSizeZ"] = _char_array(str(true_z))
                if include_data:
                    shape = (padded_z, max(1, y // scale), max(1, x // scale))
                    data = np.zeros(shape, dtype=np.uint16)
                    # fill only the true-Z region with a distinguishable value
                    data[:true_z] = c + 1
                    data[true_z:] = 9999  # padded planes: must never be read
                    grp.create_dataset("Data", data=data)

        if include_dataset_info:
            info = f.create_group("DataSetInfo")
            image = info.create_group("Image")
            _write_image_attrs(
                image,
                x,
                y,
                true_z,
                {
                    "ExtMin0": 0.0,
                    "ExtMax0": float(x) * 0.1,
                    "ExtMin1": 0.0,
                    "ExtMax1": float(y) * 0.1,
                    "ExtMin2": ext2[0],
                    "ExtMax2": ext2[1],
                },
            )
            custom = info.create_group("CustomData")
            custom.attrs["DateAndTime"] = _char_array("2026-08-07 13:23:29")
            ch0 = info.create_group("Channel 0")
            _write_channel_attrs(ch0, "561_RFP_WF_Zyla 1", "594 nm", "561 nm")
            ch1 = info.create_group("Channel 1")
            _write_channel_attrs(ch1, "405_DAPI_WF_Zyla 1", "442 nm", "405 nm")
            ch2 = info.create_group("Channel 2")
            _write_channel_attrs(ch2, "488_GFP_WF_Zyla 1", "521 nm", "488 nm")


class CharArrayDecodeTests(unittest.TestCase):
    def test_decode_char_array_attribute(self):
        value = _char_array("2026-08-07 13:23:29")
        self.assertEqual(_decode_attr(value), "2026-08-07 13:23:29")


class ZPaddingTests(unittest.TestCase):
    def test_truncates_to_true_z_not_padded_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16)
            inspection = inspect_ims(path)
            self.assertEqual(inspection["true_z"], 10)

            arrays = read_channel_arrays_ims(path, inspection, mode="3d")
            self.assertEqual(arrays[0].shape[0], 10)
            # padded planes (value 9999) must never appear in the returned stack
            self.assertTrue(np.all(arrays[0] != 9999))
            self.assertTrue(np.all(arrays[0] == 1))

    def test_2d_mode_returns_single_plane(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16)
            inspection = inspect_ims(path)
            arrays = read_channel_arrays_ims(path, inspection, mode="2d")
            self.assertEqual(arrays[0].ndim, 2)

    def test_read_channel_planes_matches_full_stack_slice(self):
        """The single-plane reader must agree with indexing the full stack,
        for an arbitrary (non-zero) z, at a non-zero resolution level."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, y=64, x=64)
            inspection = inspect_ims(path)
            full = read_channel_arrays_ims(path, inspection, mode="3d", resolution_level=1)
            planes = read_channel_planes_ims(path, [0, 1, 2], z_index=3, resolution_level=1)
            for index in (0, 1, 2):
                np.testing.assert_array_equal(planes[index], full[index][3])

    def test_read_channel_planes_skips_missing_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16)
            planes = read_channel_planes_ims(path, [0, 99], z_index=0)
            self.assertIn(0, planes)
            self.assertNotIn(99, planes)


class VoxelSizeTests(unittest.TestCase):
    def test_pixel_size_um_from_ims_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, y=2048, x=2048, ext2=(4173.76, 4180.42))
            inspection = inspect_ims(path)
            z, y, x = pixel_size_um(inspection["metadata"], mode="3d")
            self.assertAlmostEqual(x, 0.1, places=4)
            self.assertAlmostEqual(y, 0.1, places=4)
            self.assertAlmostEqual(z, (4180.42 - 4173.76) / 9, places=4)

    def test_pixel_size_scales_with_resolution_level(self):
        """Level-N reads must report level-N spacing, not level-0 spacing.

        Regression test for the 4x error: DataSetInfo/Image X/Y are always
        the level-0 dimensions, so a level-2 read reported spacing 4x too
        small and every physical-unit measurement derived from it was wrong.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, y=2048, x=2048, ext2=(4173.76, 4180.42))
            metadata = inspect_ims(path)["metadata"]

            _, y0, x0 = pixel_size_um_ims(metadata, mode="3d", resolution_level=0)
            self.assertAlmostEqual(x0, 0.1, places=4)

            for level, expected in ((1, 2.0), (2, 4.0), (3, 8.0)):
                _, y_n, x_n = pixel_size_um_ims(metadata, mode="3d", resolution_level=level)
                self.assertAlmostEqual(x_n / x0, expected, places=6)
                self.assertAlmostEqual(y_n / y0, expected, places=6)

    def test_pixel_size_z_step_is_level_independent(self):
        """The pyramid halves Y/X but not Z, so the z-step must not change."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, y=2048, x=2048, ext2=(4173.76, 4180.42))
            metadata = inspect_ims(path)["metadata"]
            steps = [pixel_size_um_ims(metadata, "3d", lvl)[0] for lvl in range(4)]
            for step in steps[1:]:
                self.assertAlmostEqual(step, steps[0], places=9)

    def test_pixel_size_falls_back_when_level_dims_absent(self):
        """With no per-level attrs the halving rule must still give level-N spacing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, y=2048, x=2048, ext2=(4173.76, 4180.42))
            metadata = inspect_ims(path)["metadata"]
            metadata["levels"] = {}
            _, _, x2 = pixel_size_um_ims(metadata, "3d", resolution_level=2)
            self.assertAlmostEqual(x2, 0.4, places=4)


class ChannelResolutionTests(unittest.TestCase):
    def test_resolves_channels_by_name_not_position(self):
        """DAPI is channel 1 on this instrument, not channel 0 or 2."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16)
            metadata = inspect_ims(path)["metadata"]
            self.assertEqual(resolve_channel_indices(metadata), {"561": 0, "405": 1, "488": 2})

    def test_unexpected_channel_set_raises(self):
        """A channel set that is not 405/488/561 must fail loudly, not mislabel."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16)
            metadata = inspect_ims(path)["metadata"]
            metadata["channels"][2] = {
                "Name": "640_Cy5_WF_Zyla 1",
                "LSMExcitationWavelength": "640 nm",
            }
            with self.assertRaises(ValueError) as ctx:
                resolve_channel_indices(metadata)
            self.assertIn("488", str(ctx.exception))

    def test_dispatch_via_inspect_czi_and_read_channel_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16)
            inspection = inspect_czi(path)
            arrays = read_channel_arrays(path, inspection, mode="3d")
            self.assertEqual(arrays[0].shape[0], 10)


class MissingDataSetInfoTests(unittest.TestCase):
    def test_missing_dataset_info_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, include_dataset_info=False)
            inspection = inspect_ims(path)
            self.assertIn("missing DataSetInfo group", inspection["errors"])
            self.assertIsNone(inspection["true_z"])
            # Pixel data can still be read even without metadata, once true Z
            # is supplied by the caller (e.g. recovered from a sibling tile).
            inspection["true_z"] = 10
            arrays = read_channel_arrays_ims(path, inspection, mode="3d")
            self.assertEqual(arrays[0].shape[0], 10)

    def test_completely_empty_file_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.ims"
            with h5py.File(path, "w"):
                pass  # no DataSet, no DataSetInfo at all - a truncated/corrupt tile
            inspection = inspect_ims(path)
            self.assertIn("missing DataSet group", inspection["errors"])
            self.assertIsNone(inspection["metadata"])

    def test_missing_data_object_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.ims"
            _make_ims(path, true_z=10, padded_z=16, include_data=False)
            inspection = inspect_ims(path)
            self.assertTrue(any("missing Data object" in e for e in inspection["errors"]))


if __name__ == "__main__":
    unittest.main()
