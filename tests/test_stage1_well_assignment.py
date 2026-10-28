from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage1_well_assignment import (
    EXPECTED_N_CLUSTERS,
    REFERENCE_POSITIONS,
    cluster_positions,
    match_cluster_to_reference,
)


class ClusteringTests(unittest.TestCase):
    def _synthetic_8_cluster_points(self, rng: np.random.Generator, n_per_cluster: int = 9, jitter_mm: float = 0.3):
        points = []
        truth = []
        for pos_id, cx, cy in REFERENCE_POSITIONS:
            jitter = rng.normal(scale=jitter_mm, size=(n_per_cluster, 2))
            cluster_points = jitter + np.array([cx, cy])
            points.append(cluster_points)
            truth.extend([pos_id] * n_per_cluster)
        return np.vstack(points), truth

    def test_recovers_exactly_8_clusters(self):
        rng = np.random.default_rng(0)
        xy, _ = self._synthetic_8_cluster_points(rng)
        labels = cluster_positions(xy)
        self.assertEqual(len(set(labels)), EXPECTED_N_CLUSTERS)

    def test_each_cluster_matches_its_true_reference_position(self):
        rng = np.random.default_rng(1)
        xy, truth = self._synthetic_8_cluster_points(rng)
        labels = cluster_positions(xy)

        # Every point in a given synthetic cluster must map (via its
        # cluster's centroid) to the same reference position it was jittered
        # around - i.e. clustering + matching reconstructs the true layout.
        for cluster_id in set(labels):
            member_idx = np.where(labels == cluster_id)[0]
            centroid = xy[member_idx].mean(axis=0)
            matched = match_cluster_to_reference((centroid[0], centroid[1]))
            true_positions = {truth[i] for i in member_idx}
            self.assertEqual(len(true_positions), 1, "a cluster must not mix two true positions")
            self.assertEqual(matched, true_positions.pop())

    def test_within_tile_scatter_does_not_split_a_field(self):
        """Tiles within one field are sub-mm apart - well under the 5 mm tolerance."""
        rng = np.random.default_rng(2)
        base = np.array([-19.2, -45.9])
        xy = base + rng.normal(scale=0.05, size=(25, 2))  # one field, 25 tiles
        labels = cluster_positions(xy)
        self.assertEqual(len(set(labels)), 1)

    def test_far_outlier_gets_its_own_cluster(self):
        xy = np.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.1], [50.0, 50.0]])
        labels = cluster_positions(xy)
        self.assertEqual(len(set(labels)), 2)

    def test_match_cluster_to_reference_rejects_far_centroid(self):
        self.assertIsNone(match_cluster_to_reference((1000.0, 1000.0)))

    def test_match_cluster_to_reference_finds_nearest(self):
        self.assertEqual(match_cluster_to_reference((5.4, -33.8)), 1)
        self.assertEqual(match_cluster_to_reference((-6.2, -46.5)), 8)


if __name__ == "__main__":
    unittest.main()
