from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nuclear_thickness import (
    axial_fwhm_by_fit,
    PSF_AXIAL_FWHM_UM,
    axial_fwhm,
    deconvolve_psf,
    sampling_cv_component,
)


def _gaussian_profile(n_z, centre, fwhm_slices, amplitude=1000.0, floor=0.0):
    """Profiles reach axial_fwhm already background-subtracted, so floor is 0."""
    sigma = fwhm_slices / (2 * np.sqrt(2 * np.log(2)))
    z = np.arange(n_z)
    return floor + amplitude * np.exp(-((z - centre) ** 2) / (2 * sigma**2))


class AxialFwhmTests(unittest.TestCase):
    def test_recovers_a_known_width(self):
        profile = _gaussian_profile(31, centre=15, fwhm_slices=6.0)
        fwhm, censored = axial_fwhm(profile, z_step_um=0.5)
        self.assertFalse(censored)
        self.assertAlmostEqual(fwhm, 6.0 * 0.5, delta=0.1)

    def test_width_scales_with_z_step(self):
        profile = _gaussian_profile(31, centre=15, fwhm_slices=6.0)
        coarse, _ = axial_fwhm(profile, z_step_um=0.74)
        fine, _ = axial_fwhm(profile, z_step_um=0.5)
        self.assertAlmostEqual(coarse / fine, 0.74 / 0.5, places=5)

    def test_interpolates_between_slices(self):
        """Sub-slice precision matters: a 1.5 um slab is only 3-4 slices."""
        profile = _gaussian_profile(41, centre=20, fwhm_slices=5.3)
        fwhm, _ = axial_fwhm(profile, z_step_um=0.5)
        self.assertNotAlmostEqual(fwhm / 0.5, round(fwhm / 0.5), places=3)
        self.assertAlmostEqual(fwhm, 5.3 * 0.5, delta=0.1)

    def test_profile_running_off_the_stack_is_censored(self):
        """A nucleus wider than the acquired stack must be dropped, not clipped.

        Keeping these would truncate the thick tail - biasing exactly the
        between-line comparison this measurement exists to settle.
        """
        profile = _gaussian_profile(9, centre=4, fwhm_slices=40.0)
        fwhm, censored = axial_fwhm(profile, z_step_um=0.5)
        self.assertTrue(censored)
        self.assertTrue(np.isnan(fwhm))

    def test_profile_peaking_at_the_stack_edge_is_censored(self):
        profile = _gaussian_profile(21, centre=0, fwhm_slices=6.0)
        _, censored = axial_fwhm(profile, z_step_um=0.5)
        self.assertTrue(censored)

    def test_flat_and_short_profiles_are_censored_not_crashed(self):
        self.assertTrue(axial_fwhm(np.zeros(21), 0.5)[1])
        self.assertTrue(axial_fwhm(np.array([1.0, 2.0]), 0.5)[1])

    def test_a_residual_baseline_slope_cannot_fake_a_narrow_nucleus(self):
        """Regression: subtracting the in-stack minimum invented a peak.

        A nucleus far thicker than the stack gives a nearly flat profile. An
        earlier version re-zeroed on the in-stack minimum, turning that
        residual into an apparent narrow peak and reporting a small width for
        the thickest nuclei - biasing the between-line comparison the wrong
        way. It must be censored instead.
        """
        profile = _gaussian_profile(9, centre=4, fwhm_slices=60.0)
        self.assertLess(profile.max() / profile.min(), 1.05, "profile should be near-flat")
        fwhm, censored = axial_fwhm(profile, z_step_um=0.5)
        self.assertTrue(censored)
        self.assertTrue(np.isnan(fwhm))


class PsfDeconvolutionTests(unittest.TestCase):
    def test_removes_the_psf_in_quadrature(self):
        true_um = 4.0
        measured = np.sqrt(true_um**2 + PSF_AXIAL_FWHM_UM**2)
        self.assertAlmostEqual(deconvolve_psf(measured), true_um, places=6)

    def test_object_below_the_resolution_limit_returns_zero_not_imaginary(self):
        self.assertEqual(deconvolve_psf(PSF_AXIAL_FWHM_UM * 0.5), 0.0)

    def test_matters_less_as_the_object_grows(self):
        thin = deconvolve_psf(2.0) / 2.0
        thick = deconvolve_psf(8.0) / 8.0
        self.assertLess(thin, thick)


class SamplingVarianceTests(unittest.TestCase):
    def test_cv_falls_as_more_of_the_nucleus_is_sampled(self):
        """The whole point: a better-sampled nucleus has LESS sampling noise."""
        self.assertGreater(sampling_cv_component(80, 0.2), sampling_cv_component(80, 0.6))

    def test_cv_is_zero_when_the_whole_nucleus_is_sampled(self):
        self.assertAlmostEqual(sampling_cv_component(80, 1.0), 0.0)

    def test_cv_falls_as_the_object_count_rises(self):
        self.assertGreater(sampling_cv_component(20, 0.3), sampling_cv_component(200, 0.3))

    def test_matches_the_binomial_formula(self):
        n, p = 100, 0.25
        expected = np.sqrt(n * p * (1 - p)) / (n * p)
        self.assertAlmostEqual(sampling_cv_component(n, p), expected, places=10)

    def test_invalid_inputs_return_nan(self):
        self.assertTrue(np.isnan(sampling_cv_component(0, 0.5)))
        self.assertTrue(np.isnan(sampling_cv_component(80, 0.0)))


class GaussianFitWidthTests(unittest.TestCase):
    """The crossing method censored 99.9% of nuclei; the fit must survive truncation."""

    def test_recovers_a_known_width_on_a_clean_profile(self):
        profile = _gaussian_profile(31, centre=15, fwhm_slices=6.0)
        fwhm, failed = axial_fwhm_by_fit(profile, z_step_um=0.5)
        self.assertFalse(failed)
        self.assertAlmostEqual(fwhm, 6.0 * 0.5, delta=0.1)

    def test_recovers_width_when_one_flank_is_clipped(self):
        """The case that breaks half-max crossing: nucleus runs past the stack."""
        full = _gaussian_profile(41, centre=20, fwhm_slices=14.0)
        clipped = full[16:]  # apex 4 slices in; lower flank cut before half-max
        self.assertTrue(axial_fwhm(clipped, 0.5)[1], "crossing should censor this")
        fwhm, failed = axial_fwhm_by_fit(clipped, z_step_um=0.5)
        self.assertFalse(failed, "fit should still measure it")
        self.assertAlmostEqual(fwhm, 14.0 * 0.5, delta=0.5)

    def test_fails_when_the_apex_sits_at_a_stack_edge(self):
        """With the peak at the edge neither flank constrains the width."""
        profile = _gaussian_profile(21, centre=0, fwhm_slices=6.0)
        self.assertTrue(axial_fwhm_by_fit(profile, z_step_um=0.5)[1])

    def test_rejects_an_implausible_extrapolation(self):
        profile = _gaussian_profile(11, centre=5, fwhm_slices=200.0)
        fwhm, failed = axial_fwhm_by_fit(profile, z_step_um=0.5)
        self.assertTrue(failed or fwhm <= 11 * 0.5 * 3.0)

    def test_flat_profile_fails_rather_than_returning_a_width(self):
        self.assertTrue(axial_fwhm_by_fit(np.zeros(21), 0.5)[1])


if __name__ == "__main__":
    unittest.main()
