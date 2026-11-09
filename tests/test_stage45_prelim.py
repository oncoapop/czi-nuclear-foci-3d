from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage45_prelim_foci_coloc import (
    COLOC_TOLERANCE_XY_UM,
    COLOC_TOLERANCE_Z_UM,
    DESIGN,
    candidate_pool_null,
    ellipsoidal_coloc,
    permutation_null_coloc,
    place_slab,
    robust_noise_sigma,
    scale_for_ellipsoid,
)


class SlabPlacementTests(unittest.TestCase):
    """Constant thickness is what makes counts comparable between tiles."""

    def test_slab_is_centred_on_focus_when_there_is_room(self):
        self.assertEqual(place_slab(10, 5, 15, 20, thickness_slices=3), (9, 11))

    def test_slab_slides_to_fit_an_asymmetric_usable_range(self):
        """Field _8's case: usable range sits to one side of the focus peak.

        Insisting on centring excluded every tile of the densest field; the
        slab must slide while keeping its thickness.
        """
        slab = place_slab(10, 10, 14, 20, thickness_slices=3)
        self.assertEqual(slab, (10, 12))
        self.assertEqual(slab[1] - slab[0] + 1, 3)

    def test_slab_is_rejected_when_usable_range_is_too_thin(self):
        self.assertIsNone(place_slab(10, 9, 11, 20, thickness_slices=5))

    def test_slab_never_leaves_the_stack(self):
        slab = place_slab(0, 0, 19, 20, thickness_slices=3)
        self.assertIsNotNone(slab)
        self.assertGreaterEqual(slab[0], 0)
        self.assertLessEqual(slab[1], 19)

    def test_thickness_is_always_exactly_as_requested(self):
        for focus in range(0, 20):
            slab = place_slab(focus, 2, 17, 20, thickness_slices=4)
            self.assertIsNotNone(slab)
            self.assertEqual(slab[1] - slab[0] + 1, 4)
            self.assertGreaterEqual(slab[0], 2)
            self.assertLessEqual(slab[1], 17)


class RobustNoiseTests(unittest.TestCase):
    """The density-coupling fix: the noise estimate must not grow with foci."""

    def test_matches_the_true_sigma_on_clean_noise(self):
        rng = np.random.default_rng(0)
        values = rng.normal(100.0, 5.0, 200_000)
        median, sigma = robust_noise_sigma(values)
        self.assertAlmostEqual(median, 100.0, delta=0.1)
        self.assertAlmostEqual(sigma, 5.0, delta=0.15)

    def test_sigma_is_stable_as_focus_density_rises(self):
        """Stage 2's bug: a two-sided MAD inflates with the positive tail.

        Well position 4 had the most nuclei and the fewest detections because
        of exactly this. The estimate must be flat across densities, and the
        threshold derived from it must not tighten as signal is added.
        """
        rng = np.random.default_rng(1)
        base = rng.normal(100.0, 5.0, 200_000)
        estimates, two_sided, thresholds = [], [], []
        for fraction in (0.0, 0.01, 0.05, 0.10):
            values = base.copy()
            n_spikes = int(len(values) * fraction)
            if n_spikes:
                values[:n_spikes] += rng.uniform(40, 200, n_spikes)
            centre, sigma = robust_noise_sigma(values)
            estimates.append(sigma)
            thresholds.append(centre + 8.0 * sigma)
            median = np.median(values)
            two_sided.append(np.median(np.abs(values - median)) * 1.4826)
        spread = max(estimates) - min(estimates)
        self.assertLess(spread, 0.3, f"lower-quantile sigma drifted with density: {estimates}")
        self.assertLess(
            max(thresholds) - min(thresholds), 3.0,
            f"detection threshold moved with density: {thresholds}",
        )
        self.assertGreater(
            max(two_sided) - min(two_sided), spread,
            "the two-sided MAD should be the one that drifts - otherwise this test proves nothing",
        )

    def test_centre_is_recovered_despite_a_heavy_positive_tail(self):
        rng = np.random.default_rng(7)
        values = rng.normal(100.0, 5.0, 200_000)
        values[:20_000] += rng.uniform(50, 300, 20_000)
        centre, sigma = robust_noise_sigma(values)
        self.assertAlmostEqual(centre, 100.0, delta=0.6)
        self.assertAlmostEqual(sigma, 5.0, delta=0.3)

    def test_empty_input_does_not_raise(self):
        median, sigma = robust_noise_sigma(np.zeros(0))
        self.assertEqual((median, sigma), (0.0, 0.0))


class EllipsoidalColocTests(unittest.TestCase):
    def test_tolerance_is_anisotropic_not_a_sphere(self):
        """0.5 um axial separation is INSIDE tolerance; 0.5 um lateral is outside."""
        a = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        axial = scale_for_ellipsoid(np.array([0.5]), np.array([0.0]), np.array([0.0]))
        lateral = scale_for_ellipsoid(np.array([0.0]), np.array([0.5]), np.array([0.0]))
        self.assertEqual(ellipsoidal_coloc(a, axial), (1, 1))
        self.assertEqual(ellipsoidal_coloc(a, lateral), (0, 0))

    def test_axial_tolerance_matches_the_configured_semi_axis(self):
        a = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        just_in = scale_for_ellipsoid(np.array([COLOC_TOLERANCE_Z_UM * 0.95]), np.array([0.0]), np.array([0.0]))
        just_out = scale_for_ellipsoid(np.array([COLOC_TOLERANCE_Z_UM * 1.05]), np.array([0.0]), np.array([0.0]))
        self.assertEqual(ellipsoidal_coloc(a, just_in)[0], 1)
        self.assertEqual(ellipsoidal_coloc(a, just_out)[0], 0)

    def test_lateral_tolerance_matches_the_configured_semi_axis(self):
        a = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        just_in = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([COLOC_TOLERANCE_XY_UM * 0.95]))
        just_out = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([COLOC_TOLERANCE_XY_UM * 1.05]))
        self.assertEqual(ellipsoidal_coloc(a, just_in)[0], 1)
        self.assertEqual(ellipsoidal_coloc(a, just_out)[0], 0)

    def test_reciprocal_counts_differ_when_one_focus_serves_many(self):
        """One RFP focus can satisfy several GFP foci; the counts are not symmetric."""
        a = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        b = scale_for_ellipsoid(np.array([0.0, 0.1, -0.1]), np.zeros(3), np.zeros(3))
        n_a, n_b = ellipsoidal_coloc(a, b)
        self.assertEqual(n_a, 1)
        self.assertEqual(n_b, 3)

    def test_empty_sets_give_no_colocalization(self):
        empty = np.zeros((0, 3))
        a = scale_for_ellipsoid(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        self.assertEqual(ellipsoidal_coloc(a, empty), (0, 0))
        self.assertEqual(ellipsoidal_coloc(empty, empty), (0, 0))


class PermutationNullTests(unittest.TestCase):
    def _square_nucleus(self, size=60):
        yy, xx = np.mgrid[:size, :size]
        return np.column_stack([yy.ravel(), xx.ravel()])

    def test_null_rises_with_the_number_of_random_foci(self):
        pixels = self._square_nucleus()
        a = scale_for_ellipsoid(np.array([2.0]), np.array([3.0]), np.array([3.0]))
        rng = np.random.default_rng(0)
        few = permutation_null_coloc(a, 2, pixels, (4, 8), 0.1005, 0.5, rng, n_permutations=200)
        many = permutation_null_coloc(a, 40, pixels, (4, 8), 0.1005, 0.5, rng, n_permutations=200)
        self.assertGreater(many, few)

    def test_null_is_higher_in_a_small_nucleus_than_a_large_one(self):
        """Why the null must be per nucleus: chance coincidence scales with crowding."""
        small = self._square_nucleus(20)
        large = self._square_nucleus(120)
        rng = np.random.default_rng(2)
        a_small = scale_for_ellipsoid(np.array([2.0]), np.array([1.0]), np.array([1.0]))
        n_small = permutation_null_coloc(a_small, 10, small, (4, 8), 0.1005, 0.5, rng, n_permutations=300)
        n_large = permutation_null_coloc(a_small, 10, large, (4, 8), 0.1005, 0.5, rng, n_permutations=300)
        self.assertGreater(n_small, n_large)

    def test_null_is_zero_without_foci_on_one_side(self):
        pixels = self._square_nucleus()
        rng = np.random.default_rng(3)
        a = scale_for_ellipsoid(np.array([2.0]), np.array([3.0]), np.array([3.0]))
        self.assertEqual(permutation_null_coloc(a, 0, pixels, (4, 8), 0.1005, 0.5, rng), 0.0)
        self.assertEqual(
            permutation_null_coloc(np.zeros((0, 3)), 5, pixels, (4, 8), 0.1005, 0.5, rng), 0.0
        )

    def test_observed_equals_null_when_foci_really_are_random(self):
        """Sanity floor: random placement must not produce apparent enrichment."""
        rng = np.random.default_rng(5)
        pixels = self._square_nucleus(80)
        slab, px, dz = (4, 8), 0.1005, 0.5
        a_pix = pixels[rng.integers(len(pixels), size=15)]
        a_z = rng.uniform(slab[0], slab[1] + 1, size=15)
        a = scale_for_ellipsoid(a_z * dz, a_pix[:, 0] * px, a_pix[:, 1] * px)
        expected = permutation_null_coloc(a, 15, pixels, slab, px, dz, rng, n_permutations=400)
        observed = []
        for _ in range(40):
            b_pix = pixels[rng.integers(len(pixels), size=15)]
            b_z = rng.uniform(slab[0], slab[1] + 1, size=15)
            b = scale_for_ellipsoid(b_z * dz, b_pix[:, 0] * px, b_pix[:, 1] * px)
            observed.append(ellipsoidal_coloc(a, b)[1])
        self.assertAlmostEqual(float(np.mean(observed)), expected, delta=1.0)


class CandidatePoolNullTests(unittest.TestCase):
    """The fix for the artifact the no-primary wells exposed."""

    def _hotspot_landscape(self, rng, n, size=60):
        """Detections clustered into a few 'bright' sub-regions of the nucleus.

        This is what both channels actually look like: detections - signal or
        noise - concentrate where the image is locally bright, so the two
        channels share a landscape even when nothing is colocalized.
        """
        centres = np.array([[10.0, 12.0], [40.0, 45.0], [20.0, 50.0]])
        pick = centres[rng.integers(len(centres), size=n)]
        return np.clip(pick + rng.normal(0, 3.0, (n, 2)), 0, size - 1)

    def _scaled(self, yx, rng, slab=(4, 8), px=0.1005, dz=0.5):
        z = rng.uniform(slab[0], slab[1] + 1, size=len(yx)) * dz
        return scale_for_ellipsoid(z, yx[:, 0] * px, yx[:, 1] * px)

    def test_uniform_null_falsely_enriches_on_a_shared_landscape(self):
        """Reproduces the failure: no real colocalization, yet enrichment >> 1.

        Both channels drawn independently from the same hotspot landscape -
        by construction there is NO colocalization signal, only a common
        cause. The uniform-in-nucleus null must be shown to fail here, or the
        replacement test below proves nothing.
        """
        rng = np.random.default_rng(11)
        pixels = np.column_stack([g.ravel() for g in np.mgrid[:60, :60]])
        a = self._scaled(self._hotspot_landscape(rng, 25), rng)
        b_yx = self._hotspot_landscape(rng, 15)
        b = self._scaled(b_yx, rng)
        observed = ellipsoidal_coloc(a, b)[1]
        uniform = permutation_null_coloc(a, 15, pixels, (4, 8), 0.1005, 0.5, rng, n_permutations=300)
        self.assertGreater(observed, uniform * 1.5,
                           "the uniform null should under-predict on a shared landscape")

    def test_candidate_pool_null_reads_about_one_on_a_shared_landscape(self):
        """The replacement: same data, enrichment must collapse to ~1."""
        rng = np.random.default_rng(11)
        a = self._scaled(self._hotspot_landscape(rng, 25), rng)
        pool_yx = self._hotspot_landscape(rng, 120)
        pool = self._scaled(pool_yx, rng)
        chosen = rng.choice(len(pool), size=15, replace=False)
        observed = ellipsoidal_coloc(a, pool[chosen])[1]
        expected = candidate_pool_null(a, 15, pool, rng, n_permutations=400)
        self.assertGreater(expected, 0.0)
        self.assertAlmostEqual(observed / expected, 1.0, delta=0.6)

    def test_candidate_pool_null_still_detects_real_colocalization(self):
        """It must not be so conservative that true signal disappears."""
        rng = np.random.default_rng(3)
        a_yx = self._hotspot_landscape(rng, 25)
        a = self._scaled(a_yx, rng)
        pool = self._scaled(self._hotspot_landscape(rng, 120), rng)
        # B sits exactly on A - genuine colocalization
        b = a[: 15].copy()
        observed = ellipsoidal_coloc(a, b)[1]
        expected = candidate_pool_null(a, 15, pool, rng, n_permutations=400)
        self.assertGreater(observed / max(expected, 1e-9), 2.0)

    def test_returns_nan_when_the_pool_cannot_support_a_draw(self):
        rng = np.random.default_rng(0)
        a = self._scaled(self._hotspot_landscape(rng, 5), rng)
        pool = self._scaled(self._hotspot_landscape(rng, 3), rng)
        self.assertTrue(np.isnan(candidate_pool_null(a, 5, pool, rng)))
        self.assertTrue(np.isnan(candidate_pool_null(a, 0, pool, rng)))


class DesignTests(unittest.TestCase):
    def test_design_covers_all_eight_positions_exactly_once(self):
        self.assertEqual(len(DESIGN), 8)
        self.assertEqual(sorted(d["well"] for d in DESIGN.values()), list(range(1, 9)))

    def test_the_two_no_primary_wells_are_four_and_eight(self):
        blanks = sorted(d["well"] for d in DESIGN.values() if not d["has_primary"])
        self.assertEqual(blanks, [4, 8])

    def test_one_no_primary_control_per_cell_line(self):
        lines = [d["cell_line"] for d in DESIGN.values() if not d["has_primary"]]
        self.assertEqual(len(set(lines)), 2, "each cell line needs its own secondary-only control")

    def test_each_cell_line_has_two_palbociclib_and_two_untreated_wells(self):
        for line in {d["cell_line"] for d in DESIGN.values()}:
            wells = [d for d in DESIGN.values() if d["cell_line"] == line]
            self.assertEqual(sum(w["treatment"] == "Palbociclib" for w in wells), 2)
            self.assertEqual(sum(w["treatment"] == "untreated" for w in wells), 2)


if __name__ == "__main__":
    unittest.main()
