from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage2_registration import (
    DZ_MAX_NARROW_UM,
    PSF_SIGMA_Z_UM,
    DetectionStats,
    choose_primary_k,
    detect_foci_3d,
    difference_of_gaussians,
    disc_offsets,
    extract_triggered_profiles,
    fit_triggered_profile,
    fixed_sigma_um,
    nucleus_pixel_index,
    permuted_centers,
    qualifying_mask,
    xy_offset_by_crosscorr,
)

PIXEL_UM = 0.1005
Z_STEP_UM = 0.5


def _gaussian_blob(shape, center_zyx, sigma_z_px, sigma_xy_px, amplitude):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center_zyx
    return amplitude * np.exp(
        -(((z - cz) ** 2) / (2 * sigma_z_px**2) + ((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma_xy_px**2))
    )


class FixedSigmaTests(unittest.TestCase):
    def test_matches_the_plan_value_at_half_micron_steps(self):
        """sigma = sqrt(0.38^2 + (z_step/sqrt(12))^2) ~ 0.41 um at a 0.5 um step."""
        self.assertAlmostEqual(fixed_sigma_um(0.5), 0.41, places=2)

    def test_grows_with_coarser_z_sampling(self):
        """Fields 2 and 3 were acquired at 0.74 um; sigma must follow the data."""
        self.assertGreater(fixed_sigma_um(0.74), fixed_sigma_um(0.5))

    def test_never_falls_below_the_optical_psf(self):
        self.assertGreaterEqual(fixed_sigma_um(0.001), PSF_SIGMA_Z_UM)


class ProfileFitTests(unittest.TestCase):
    def test_recovers_a_known_offset_from_a_synthetic_profile(self):
        dz = np.arange(-5, 6) * Z_STEP_UM
        sigma = fixed_sigma_um(Z_STEP_UM)
        truth = 0.35
        profile = 4.0 + 0.2 * dz + 0.05 * dz**2 + 3.0 * np.exp(-((dz - truth) ** 2) / (2 * sigma**2))
        fit = fit_triggered_profile(dz, profile, sigma, pedestal_order=2)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["mu_um"], truth, places=2)
        self.assertAlmostEqual(fit["amplitude"], 3.0, places=1)

    def test_a_sloped_pedestal_does_not_drag_mu(self):
        """A linear pedestal term is fitted explicitly, so tilt must not shift mu."""
        dz = np.arange(-3, 4) * Z_STEP_UM
        sigma = fixed_sigma_um(Z_STEP_UM)
        flat = 10.0 + 2.0 * np.exp(-(dz**2) / (2 * sigma**2))
        sloped = flat + 1.5 * dz
        mu_flat = fit_triggered_profile(dz, flat, sigma, pedestal_order=1)["mu_um"]
        mu_sloped = fit_triggered_profile(dz, sloped, sigma, pedestal_order=1)["mu_um"]
        self.assertAlmostEqual(mu_flat, mu_sloped, places=2)

    def test_narrow_window_linear_pedestal_agrees_with_wide_window_quadratic(self):
        """The licence for using the narrow window at positions 5 and 6."""
        dz_wide = np.arange(-5, 6) * Z_STEP_UM
        sigma = fixed_sigma_um(Z_STEP_UM)
        truth = -0.4
        curve = lambda d: 6.0 + 0.3 * d + 0.08 * d**2 + 2.5 * np.exp(-((d - truth) ** 2) / (2 * sigma**2))
        wide = fit_triggered_profile(dz_wide, curve(dz_wide), sigma, pedestal_order=2)
        dz_narrow = dz_wide[np.abs(dz_wide) <= DZ_MAX_NARROW_UM + 1e-9]
        narrow = fit_triggered_profile(dz_narrow, curve(dz_narrow), sigma, pedestal_order=1)
        self.assertAlmostEqual(wide["mu_um"], truth, places=2)
        self.assertAlmostEqual(narrow["mu_um"], truth, places=1)

    def test_wing_only_pedestal_recipe_agrees_with_the_joint_fit(self):
        dz = np.arange(-5, 6) * Z_STEP_UM
        sigma = fixed_sigma_um(Z_STEP_UM)
        truth = 0.25
        profile = 5.0 + 0.1 * dz + 3.0 * np.exp(-((dz - truth) ** 2) / (2 * sigma**2))
        joint = fit_triggered_profile(dz, profile, sigma, pedestal_order=2)
        wing = fit_triggered_profile(dz, profile, sigma, pedestal_order=2, wing_only_pedestal=True)
        self.assertAlmostEqual(joint["mu_um"], wing["mu_um"], places=1)

    def test_wide_window_quadratic_pedestal_must_not_be_fitted_jointly(self):
        """Regression: the degeneracy that made the wide window return a wrong mu.

        A free quadratic pedestal over +-2.5 um can mimic the narrow bump,
        absorb it, and leave the Gaussian fitting a negative dip somewhere
        else entirely. On this measured-shape profile (true peak at +1.0 um)
        the joint fit returned mu=-0.55 with a NEGATIVE amplitude, while the
        wing-only pedestal recovered +1.04. The pipeline must use wing-only
        for the wide window, and a negative amplitude must be treated as a
        failed fit rather than a measurement.
        """
        dz = np.arange(-5, 6) * Z_STEP_UM
        profile = np.array(
            [99.92, 100.10, 100.12, 100.25, 102.29, 118.87, 168.53, 205.41, 169.03, 119.36, 102.48]
        )
        sigma = fixed_sigma_um(Z_STEP_UM)
        joint = fit_triggered_profile(dz, profile, sigma, pedestal_order=2)
        wing = fit_triggered_profile(dz, profile, sigma, pedestal_order=2, wing_only_pedestal=True)
        self.assertAlmostEqual(wing["mu_um"], 1.0, delta=0.1)
        self.assertGreater(wing["amplitude"], 0.0)
        self.assertLess(joint["amplitude"], 0.0, "the known degenerate failure mode")

    def test_too_few_points_fails_loudly_rather_than_returning_a_number(self):
        fit = fit_triggered_profile(np.array([0.0, 0.5]), np.array([1.0, 2.0]), 0.41, pedestal_order=2)
        self.assertFalse(fit["ok"])
        self.assertTrue(np.isnan(fit["mu_um"]))


class TriggeredAverageEndToEndTests(unittest.TestCase):
    """A synthetic two-channel volume with a known axial offset between channels."""

    def _two_channel_volume(self, offset_slices: int, n_foci: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        shape = (21, 160, 160)
        trigger = np.zeros(shape, dtype=np.float32)
        target = np.zeros(shape, dtype=np.float32)
        sigma_xy_px = 0.15 / PIXEL_UM
        sigma_z_px = (0.15 * 3.6) / Z_STEP_UM
        for _ in range(n_foci):
            cz = rng.integers(8, 13)
            cy = rng.integers(20, 140)
            cx = rng.integers(20, 140)
            trigger += _gaussian_blob(shape, (cz, cy, cx), sigma_z_px, sigma_xy_px, 400.0)
            target += _gaussian_blob(shape, (cz + offset_slices, cy, cx), sigma_z_px, sigma_xy_px, 250.0)
        trigger += 100.0 + rng.normal(0, 2.0, shape).astype(np.float32)
        target += 100.0 + rng.normal(0, 2.0, shape).astype(np.float32)
        return trigger, target

    def _fit_offset(self, trigger, target, dz_slices):
        """Runs the wide window exactly as the pipeline does: wing-only pedestal."""
        mask = np.ones(trigger.shape[1:], dtype=bool)
        dog = difference_of_gaussians(trigger, PIXEL_UM, Z_STEP_UM)
        coords, amplitudes, _ = detect_foci_3d(dog, mask, PIXEL_UM, Z_STEP_UM, mad_k=6.0)
        keep = qualifying_mask(coords.astype(float), dz_slices, target.shape, edge_margin_px=5)
        coords = coords[keep]
        self.assertGreater(len(coords), 10, "synthetic detection found too few foci to fit")
        dy, dx = disc_offsets(0.3, PIXEL_UM)
        profiles = extract_triggered_profiles(target, coords.astype(float), dz_slices, dy, dx)
        dz_um = dz_slices * Z_STEP_UM
        return fit_triggered_profile(
            dz_um, profiles.mean(axis=0), fixed_sigma_um(Z_STEP_UM),
            pedestal_order=2, wing_only_pedestal=True,
        )

    def test_recovers_a_known_positive_axial_offset(self):
        trigger, target = self._two_channel_volume(offset_slices=2)
        dz_slices = np.arange(-5, 6)
        fit = self._fit_offset(trigger, target, dz_slices)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["mu_um"], 2 * Z_STEP_UM, delta=0.15)

    def test_reverse_direction_flips_the_sign(self):
        """The plan's internal validation: no beads exist, so this is the check."""
        trigger, target = self._two_channel_volume(offset_slices=2)
        dz_slices = np.arange(-5, 6)
        forward = self._fit_offset(trigger, target, dz_slices)
        reverse = self._fit_offset(target, trigger, dz_slices)
        self.assertLess(forward["mu_um"] * reverse["mu_um"], 0.0)
        self.assertAlmostEqual(forward["mu_um"], -reverse["mu_um"], delta=0.2)

    def test_perfectly_registered_channels_give_mu_near_zero(self):
        trigger, target = self._two_channel_volume(offset_slices=0)
        fit = self._fit_offset(trigger, target, np.arange(-5, 6))
        self.assertAlmostEqual(fit["mu_um"], 0.0, delta=0.12)


class SamplingAndNullTests(unittest.TestCase):
    def test_disc_mean_reads_the_disc_not_the_whole_plane(self):
        volume = np.zeros((5, 50, 50), dtype=np.float32)
        volume[2, 24:27, 24:27] = 10.0
        dy, dx = disc_offsets(0.3, PIXEL_UM)
        centers = np.array([[2.0, 25.0, 25.0]])
        profile = extract_triggered_profiles(volume, centers, np.array([0]), dy, dx)
        self.assertGreater(profile[0, 0], 0.0)
        self.assertLess(profile[0, 0], 10.0)

    def test_out_of_stack_offsets_are_nan_not_clamped(self):
        """Clamping would fabricate a value from the wrong plane.

        A focus near the stack edge must report NaN beyond the stack so the
        caller can exclude it from that window, rather than silently reusing
        the end plane and flattening the profile toward it.
        """
        volume = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((10, 40, 40), dtype=np.float32)
        dy, dx = disc_offsets(0.3, PIXEL_UM)
        centers = np.array([[1.0, 20.0, 20.0]])
        profile = extract_triggered_profiles(volume, centers, np.arange(-3, 4), dy, dx)
        self.assertTrue(np.isnan(profile[0, 0]), "dz=-3 from z=1 is outside the stack")
        self.assertTrue(np.isnan(profile[0, 1]))
        self.assertFalse(np.isnan(profile[0, 2]))
        self.assertAlmostEqual(float(profile[0, 3]), 1.0, places=5)

    def test_a_focus_covering_only_the_narrow_window_still_counts_there(self):
        """The two-window design: shallow-slab fields must not be silently lost."""
        volume = np.ones((9, 40, 40), dtype=np.float32)
        dy, dx = disc_offsets(0.3, PIXEL_UM)
        centers = np.array([[3.0, 20.0, 20.0]])
        wide = extract_triggered_profiles(volume, centers, np.arange(-5, 6), dy, dx)
        narrow_cols = np.abs(np.arange(-5, 6)) <= 3
        self.assertFalse(np.isfinite(wide[0]).all(), "cannot cover the wide window")
        self.assertTrue(np.isfinite(wide[0, narrow_cols]).all(), "must cover the narrow window")

    def test_qualifying_mask_rejects_foci_without_the_full_dz_range(self):
        centers = np.array([[1.0, 50.0, 50.0], [10.0, 50.0, 50.0], [19.0, 50.0, 50.0]])
        keep = qualifying_mask(centers, np.arange(-5, 6), (21, 100, 100), edge_margin_px=5)
        self.assertFalse(keep[0])
        self.assertTrue(keep[1])
        self.assertFalse(keep[2])

    def test_qualifying_mask_rejects_foci_too_close_to_the_border(self):
        centers = np.array([[10.0, 2.0, 50.0], [10.0, 50.0, 98.0]])
        keep = qualifying_mask(centers, np.arange(-2, 3), (21, 100, 100), edge_margin_px=5)
        self.assertFalse(keep.any())

    def test_permuted_centers_stay_in_the_same_nucleus_and_keep_z(self):
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[5:15, 5:15] = 1
        labels[25:35, 25:35] = 2
        index = nucleus_pixel_index(labels)
        centers = np.array([[7.0, 10.0, 10.0], [7.0, 30.0, 30.0]])
        focus_labels = np.array([1, 2])
        rng = np.random.default_rng(0)
        for _ in range(20):
            permuted = permuted_centers(centers, focus_labels, index, rng)
            self.assertTrue(np.array_equal(permuted[:, 0], centers[:, 0]), "z must be preserved")
            for i, label in enumerate(focus_labels):
                y, x = int(permuted[i, 1]), int(permuted[i, 2])
                self.assertEqual(labels[y, x], label)

    def test_nucleus_pixel_index_matches_a_brute_force_scan(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 4, size=(30, 30)).astype(np.int32)
        index = nucleus_pixel_index(labels)
        for label in (1, 2, 3):
            expected = np.argwhere(labels == label)
            got = index[label]
            self.assertEqual(len(got), len(expected))
            self.assertTrue(set(map(tuple, got.tolist())) == set(map(tuple, expected.tolist())))


class DetectionThresholdTests(unittest.TestCase):
    def test_stricter_k_is_an_exact_subset_of_the_looser_detection(self):
        """How mu-vs-threshold is checked without re-running detection."""
        stats = DetectionStats(median=0.0, mad_scaled=1.0)
        amplitudes = np.array([5.0, 7.0, 9.0, 13.0])
        loose = amplitudes >= stats.threshold(6.0)
        strict = amplitudes >= stats.threshold(12.0)
        self.assertTrue(np.all(strict <= loose))
        self.assertEqual(loose.sum(), 3)
        self.assertEqual(strict.sum(), 1)

    def test_detection_finds_planted_foci_and_amplitude_tracks_brightness(self):
        shape = (15, 120, 120)
        volume = np.full(shape, 100.0, dtype=np.float32)
        sigma_xy_px = 0.15 / PIXEL_UM
        sigma_z_px = (0.15 * 3.6) / Z_STEP_UM
        volume += _gaussian_blob(shape, (7, 40, 40), sigma_z_px, sigma_xy_px, 200.0)
        volume += _gaussian_blob(shape, (7, 80, 80), sigma_z_px, sigma_xy_px, 800.0)
        dog = difference_of_gaussians(volume, PIXEL_UM, Z_STEP_UM)
        coords, amplitudes, _ = detect_foci_3d(dog, np.ones(shape[1:], bool), PIXEL_UM, Z_STEP_UM, 6.0)
        self.assertGreaterEqual(len(coords), 2)
        brightest = coords[np.argmax(amplitudes)]
        self.assertLess(abs(brightest[1] - 80) + abs(brightest[2] - 80), 3)


class DecompositionTests(unittest.TestCase):
    """Splitting forward/reverse into registration vs common-mode."""

    @staticmethod
    def _decompose(mu_forward, mu_reverse):
        return (mu_forward - mu_reverse) / 2.0, (mu_forward + mu_reverse) / 2.0

    def test_a_pure_channel_offset_is_recovered_entirely_as_delta(self):
        delta, common = self._decompose(+0.4, -0.4)
        self.assertAlmostEqual(delta, 0.4)
        self.assertAlmostEqual(common, 0.0)

    def test_a_pure_common_mode_gradient_yields_zero_delta(self):
        """Both directions shifted the same way is NOT a registration offset."""
        delta, common = self._decompose(-0.2, -0.2)
        self.assertAlmostEqual(delta, 0.0)
        self.assertAlmostEqual(common, -0.2)

    def test_superposition_separates_the_two(self):
        true_delta, true_common = 0.05, -0.15
        delta, common = self._decompose(true_delta + true_common, -true_delta + true_common)
        self.assertAlmostEqual(delta, true_delta)
        self.assertAlmostEqual(common, true_common)

    def test_choose_primary_k_prefers_the_strictest_threshold_with_enough_foci(self):
        def entry(n):
            return (np.zeros(3), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n))

        plenty = {("forward", k): entry(5000) for k in (6.0, 8.0, 12.0)}
        plenty.update({("reverse", k): entry(5000) for k in (6.0, 8.0, 12.0)})
        self.assertEqual(choose_primary_k(plenty), 12.0)

    def test_choose_primary_k_backs_off_when_strict_thresholds_starve(self):
        """Field _8's failure mode: dense foci inflate the MAD and k=8 starves."""
        def entry(n):
            return (np.zeros(3), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n))

        cache = {
            ("forward", 6.0): entry(900), ("reverse", 6.0): entry(900),
            ("forward", 8.0): entry(20), ("reverse", 8.0): entry(15),
        }
        self.assertEqual(choose_primary_k(cache), 6.0)

    def test_choose_primary_k_requires_both_directions(self):
        def entry(n):
            return (np.zeros(3), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n))

        cache = {
            ("forward", 12.0): entry(5000),
            ("forward", 6.0): entry(5000), ("reverse", 6.0): entry(5000),
        }
        self.assertEqual(choose_primary_k(cache), 6.0)


class LateralOffsetTests(unittest.TestCase):
    def test_cross_correlation_recovers_a_planted_lateral_shift(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, (6, 128, 128)).astype(np.float32)
        base = np.stack([np.clip(p, 0, None) for p in base])
        shifted = np.roll(base, shift=4, axis=2)  # +4 px in x
        dy, dx, peak = xy_offset_by_crosscorr(base, shifted, z_shift_slices=0, pixel_size_um=PIXEL_UM)
        self.assertAlmostEqual(dy, 0.0, places=6)
        self.assertAlmostEqual(dx, -4 * PIXEL_UM, places=6)
        self.assertGreater(peak, 0.1)


if __name__ == "__main__":
    unittest.main()
