from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage1b_flatfield import (
    SMOOTH_BOWL_MAX_SCORE,
    estimate_illumination_field,
    nucleus_density_radial_profile,
    radial_slope_and_edge_pct,
    smooth_bowl_score,
)


class FlatFieldNoOpTests(unittest.TestCase):
    """A perfectly uniform synthetic field must correct to a no-op."""

    def test_uniform_field_is_flat_and_passes_gate(self):
        rng = np.random.default_rng(0)
        # 20 tiles, uniform illumination plus small Poisson-like noise - no
        # vignetting, no cell structure.
        planes = [np.full((256, 256), 1000.0) + rng.normal(scale=5.0, size=(256, 256)) for _ in range(20)]
        planes = [np.clip(p, 0, None).astype(np.uint16) for p in planes]

        result = estimate_illumination_field(planes)
        self.assertTrue(result.passed_gate)
        self.assertLessEqual(result.score, SMOOTH_BOWL_MAX_SCORE)
        # Normalised to median 1.0, and since the input is uniform the field
        # itself should be close to flat (no real bowl to find).
        self.assertAlmostEqual(float(np.median(result.field)), 1.0, places=3)
        self.assertLess(float(result.field.std()), 0.05)

    def test_dividing_by_a_uniform_field_is_a_no_op(self):
        rng = np.random.default_rng(1)
        raw = (np.full((256, 256), 2000.0) + rng.normal(scale=10.0, size=(256, 256))).astype(np.float32)
        uniform_field = np.ones_like(raw)
        corrected = raw / uniform_field
        np.testing.assert_array_almost_equal(corrected, raw)

    def test_a_real_bowl_is_removed_by_correction(self):
        """A synthetic vignette (radial falloff) should be corrected to flat."""
        size = 256
        yy, xx = np.indices((size, size))
        cy, cx = (size - 1) / 2.0, (size - 1) / 2.0
        radius = np.hypot(yy - cy, xx - cx)
        bowl = 1.0 - 0.3 * (radius / radius.max())  # 30% edge falloff

        rng = np.random.default_rng(2)
        planes = []
        for _ in range(20):
            cells = rng.normal(scale=20.0, size=(size, size))
            plane = np.clip((1000.0 * bowl) + cells, 0, None).astype(np.uint16)
            planes.append(plane)

        result = estimate_illumination_field(planes)
        self.assertTrue(result.passed_gate)

        # Correcting a fresh noisy plane with this field should flatten its
        # radial profile much more than leaving it uncorrected.
        test_plane = np.clip((1000.0 * bowl) + rng.normal(scale=20.0, size=(size, size)), 0, None).astype(np.float32)
        corrected = test_plane / result.field

        def edge_vs_center(plane):
            center = plane[cy.__int__() - 5 : cy.__int__() + 5, cx.__int__() - 5 : cx.__int__() + 5].mean()
            edge = plane[:10, :10].mean()
            return (edge - center) / center

        before = edge_vs_center(test_plane)
        after = edge_vs_center(corrected)
        self.assertLess(abs(after), abs(before))


class SmoothBowlScoreTests(unittest.TestCase):
    def test_identical_fields_score_zero(self):
        field = np.ones((64, 64)) * 3.0
        self.assertEqual(smooth_bowl_score(field, field), 0.0)

    def test_noisy_raw_vs_smooth_scores_above_zero(self):
        rng = np.random.default_rng(0)
        # A real bowl (non-zero range) so the score denominator isn't the
        # near-flat-field floor.
        yy, xx = np.indices((64, 64))
        smoothed = 1.0 - 0.3 * (np.hypot(yy - 31.5, xx - 31.5) / 45.0)
        raw = smoothed + rng.normal(scale=0.5, size=(64, 64))
        self.assertGreater(smooth_bowl_score(raw, smoothed), 0.0)


class RadialHelperTests(unittest.TestCase):
    def test_nucleus_density_uniform_centroids_gives_roughly_flat_profile(self):
        rng = np.random.default_rng(0)
        shape = (512, 512)
        # uniformly scattered centroids across the whole tile
        centroids = rng.uniform(0, 512, size=(400, 2))
        density = nucleus_density_radial_profile(centroids, shape, pixel_size_um=0.2)
        slope, _ = radial_slope_and_edge_pct(density)
        # Should be small relative to the mean density, not necessarily exactly 0.
        self.assertLess(abs(slope), 0.5 * np.nanmean(density) / 10)

    def test_radial_slope_handles_too_few_valid_bins(self):
        profile = np.array([1.0] + [np.nan] * 7)
        slope, edge = radial_slope_and_edge_pct(profile)
        self.assertTrue(np.isnan(slope))
        self.assertTrue(np.isnan(edge))


if __name__ == "__main__":
    unittest.main()
