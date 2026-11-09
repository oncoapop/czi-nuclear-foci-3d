from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage0_ims_qc import derive_round


class RoundLabelTests(unittest.TestCase):
    """The round label must depend only on its own folder name.

    Regression test: the label was previously derived from
    os.path.commonprefix over every folder in the root, so an analysis output
    directory sitting beside the acquisitions degenerated every label to the
    full folder name and made two runs impossible to join.
    """

    TRF2 = "3x3 test_2026-08-07_TRF2 x yH2AX"
    TSIV = "3x3 test_2026-08-07_TSIV 2 hr"

    def test_numbered_round(self):
        self.assertEqual(derive_round(f"{self.TRF2}_12_Field_1"), "_12")
        self.assertEqual(derive_round(f"{self.TRF2}_1_Field_1"), "_1")

    def test_unnumbered_round(self):
        self.assertEqual(derive_round(f"{self.TRF2}_Field_1"), "(unnumbered)")

    def test_deconvolved_rounds_are_distinct_from_their_raw_partner(self):
        self.assertEqual(derive_round(f"{self.TRF2}_1_deconvolved_Field_1"), "_1_deconvolved")
        self.assertNotEqual(
            derive_round(f"{self.TRF2}_1_deconvolved_Field_1"),
            derive_round(f"{self.TRF2}_1_Field_1"),
        )
        self.assertEqual(
            derive_round(f"{self.TSIV}_deconvolved_Field_1"), "(unnumbered)_deconvolved"
        )
        self.assertEqual(
            derive_round(f"{self.TSIV}_10_deconvolved_1_Field_1"), "_10_deconvolved_1"
        )

    def test_label_is_independent_of_other_folders(self):
        """The whole point: adding a folder must not relabel anything else."""
        acquisitions = [
            f"{self.TRF2}_Field_1",
            f"{self.TRF2}_1_Field_1",
            f"{self.TRF2}_12_Field_1",
        ]
        before = [derive_round(name) for name in acquisitions]
        # An analysis output directory appears alongside the acquisitions.
        _ = derive_round("_analysis_v2")
        after = [derive_round(name) for name in acquisitions]
        self.assertEqual(before, after)
        self.assertEqual(before, ["(unnumbered)", "_1", "_12"])

    def test_labels_are_unique_across_a_real_folder_set(self):
        names = [f"{self.TRF2}_Field_1", f"{self.TRF2}_1_deconvolved_Field_1"] + [
            f"{self.TRF2}_{n}_Field_1" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        ]
        labels = [derive_round(n) for n in names]
        self.assertEqual(len(labels), len(set(labels)))


if __name__ == "__main__":
    unittest.main()


class ResolutionLevelFallbackTests(unittest.TestCase):
    """Not every .ims has a pyramid.

    Regression test: the single-plane TSIV acquisitions carry
    ResolutionLevel 0 only, so requesting level 2 unconditionally failed 500
    of 501 tiles with "ResolutionLevel 2 not present".
    """

    def test_uses_requested_level_when_available(self):
        from stage0_ims_qc import QC_RESOLUTION_LEVEL, choose_resolution_level

        inspection = {"resolution_levels": [f"ResolutionLevel {i}" for i in range(6)]}
        self.assertEqual(choose_resolution_level(inspection), QC_RESOLUTION_LEVEL)

    def test_falls_back_to_deepest_available(self):
        from stage0_ims_qc import choose_resolution_level

        self.assertEqual(choose_resolution_level({"resolution_levels": ["ResolutionLevel 0"]}), 0)
        self.assertEqual(
            choose_resolution_level(
                {"resolution_levels": ["ResolutionLevel 0", "ResolutionLevel 1"]}
            ),
            1,
        )

    def test_nucleus_constants_rescale_with_level(self):
        """Falling back to a finer level must not change what counts as a nucleus.

        Regression test for the stage0b bug: nucleus-size constants used to be
        hardcoded pixel counts calibrated for one resolution level, so reading
        at a different level (different um/px) silently changed what counted
        as a nucleus. They are now physical (um / um2) and converted to
        pixels from the actual pixel size, so the same physical field must
        count the same number of nuclei regardless of sampling density.
        """
        import numpy as np
        from stage0_ims_qc import (
            MIN_NUCLEUS_AREA_UM2,
            MIN_NUCLEUS_SEPARATION_UM,
            count_nuclei,
        )

        # Two discs of a fixed PHYSICAL size (pixel radius * pixel size is
        # constant), rendered at two different pixel sizes ("resolution
        # levels").
        def field(pixel_size_um: float, radius_um: float = 4.0) -> np.ndarray:
            radius_px = radius_um / pixel_size_um
            size = int(radius_px * 6)
            image = np.zeros((1, size, size), dtype=np.float32)
            yy, xx = np.ogrid[:size, :size]
            for cy, cx in ((size * 0.3, size * 0.3), (size * 0.7, size * 0.7)):
                image[0][(yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px**2] = 1000.0
            return image

        n_coarse, _ = count_nuclei(field(0.4), 0, pixel_size_um=0.4)
        n_fine, _ = count_nuclei(field(0.1), 0, pixel_size_um=0.1)
        self.assertEqual(n_coarse, 2)
        self.assertEqual(n_fine, 2)
        self.assertGreater(MIN_NUCLEUS_AREA_UM2, 0)
        self.assertGreater(MIN_NUCLEUS_SEPARATION_UM, 0)
