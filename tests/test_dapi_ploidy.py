from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dapi_ploidy import SATURATION_LEVEL, integrated_dna


def _stack(n_z=12, size=40, level=1000.0):
    return np.full((n_z, size, size), level, dtype=np.float32)


def _square_coords(y0, x0, side):
    ys, xs = np.mgrid[y0 : y0 + side, x0 : x0 + side]
    return np.column_stack([ys.ravel(), xs.ravel()])


class IntegratedDnaTests(unittest.TestCase):
    def test_background_is_removed(self):
        stack = _stack(level=500.0)
        coords = _square_coords(5, 5, 6)
        background = np.full(stack.shape[0], 500.0)
        dna, _ = integrated_dna(stack, coords, background)
        self.assertAlmostEqual(dna, 0.0, places=3)

    def test_scales_linearly_with_signal_above_background(self):
        coords = _square_coords(5, 5, 6)
        background = np.full(12, 100.0)
        single = _stack(level=100.0)
        single[:, 5:11, 5:11] += 50.0
        double = _stack(level=100.0)
        double[:, 5:11, 5:11] += 100.0
        a, _ = integrated_dna(single, coords, background)
        b, _ = integrated_dna(double, coords, background)
        self.assertAlmostEqual(b / a, 2.0, places=5)

    def test_scales_with_the_number_of_voxels(self):
        background = np.full(12, 100.0)
        stack = _stack(level=100.0)
        stack[:, 5:17, 5:17] += 40.0
        small, _ = integrated_dna(stack, _square_coords(5, 5, 6), background)
        large, _ = integrated_dna(stack, _square_coords(5, 5, 12), background)
        self.assertAlmostEqual(large / small, 4.0, places=5)

    def test_integrates_the_whole_stack_not_one_slab(self):
        """DNA content is a property of the whole nucleus, not the analysis slab."""
        background = np.zeros(12)
        tall = _stack(n_z=12, level=0.0)
        tall[:, 5:11, 5:11] = 10.0
        short = _stack(n_z=12, level=0.0)
        short[4:8, 5:11, 5:11] = 10.0
        coords = _square_coords(5, 5, 6)
        a, _ = integrated_dna(tall, coords, background)
        b, _ = integrated_dna(short, coords, background)
        self.assertAlmostEqual(a / b, 3.0, places=5)

    def test_saturation_is_detected(self):
        """A clipped nucleus under-reports DNA and must be excluded, not used."""
        stack = _stack(level=100.0)
        stack[:, 5:11, 5:11] = SATURATION_LEVEL + 100
        _, fraction = integrated_dna(stack, _square_coords(5, 5, 6), np.zeros(12))
        self.assertAlmostEqual(fraction, 1.0)

    def test_unsaturated_nucleus_reports_no_saturation(self):
        stack = _stack(level=100.0)
        stack[:, 5:11, 5:11] = SATURATION_LEVEL - 1000
        _, fraction = integrated_dna(stack, _square_coords(5, 5, 6), np.zeros(12))
        self.assertEqual(fraction, 0.0)

    def test_partial_saturation_is_reported_as_a_fraction(self):
        stack = _stack(level=100.0)
        stack[:, 5:8, 5:11] = SATURATION_LEVEL + 100  # half the rows
        _, fraction = integrated_dna(stack, _square_coords(5, 5, 6), np.zeros(12))
        self.assertAlmostEqual(fraction, 0.5, places=6)

    def test_per_slice_background_handles_axial_drift(self):
        """Background varies slice to slice; a single scalar would bias the sum."""
        stack = _stack(n_z=10, level=0.0)
        drift = np.linspace(100.0, 500.0, 10).astype(np.float32)
        stack += drift[:, None, None]
        coords = _square_coords(5, 5, 6)
        dna, _ = integrated_dna(stack, coords, drift)
        self.assertAlmostEqual(dna, 0.0, places=2)


class PloidyDiscriminationTests(unittest.TestCase):
    """The test's own logic: normalising by DNA must behave as claimed."""

    def test_pure_ploidy_variation_collapses_under_normalisation(self):
        rng = np.random.default_rng(0)
        ploidy = rng.normal(1.0, 0.45, 4000).clip(0.2)
        dna = ploidy * 1e6
        counts = ploidy * 20.0  # telomeres strictly proportional to DNA
        cv_raw = counts.std() / counts.mean()
        cv_norm = (counts / dna).std() / (counts / dna).mean()
        self.assertGreater(cv_raw, 0.3)
        self.assertLess(cv_norm, 0.01, "proportional counts must normalise away")

    def test_telomere_specific_variation_survives_normalisation(self):
        rng = np.random.default_rng(1)
        ploidy = rng.normal(1.0, 0.15, 4000).clip(0.2)
        dna = ploidy * 1e6
        counts = ploidy * 20.0 * rng.normal(1.0, 0.5, 4000).clip(0.05)
        cv_raw = counts.std() / counts.mean()
        cv_norm = (counts / dna).std() / (counts / dna).mean()
        self.assertGreater(cv_norm, 0.8 * cv_raw, "independent variation must persist")


if __name__ == "__main__":
    unittest.main()
