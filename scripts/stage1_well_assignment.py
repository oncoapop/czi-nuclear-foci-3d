"""Stage 1: well assignment from stage coordinates (not filenames).

Andor filenames carry no well, and the user's field->well list does not
reconcile with folder numbering or acquisition order, so
``config.filename.regex`` cannot be used here. Instead this clusters
``stage_x_mm``/``stage_y_mm`` (``DataSetInfo/Image`` ``ExtMin0``/``ExtMin1``)
at a 5 mm tolerance, which the TRF2 and TSIV datasets both reduce to exactly
8 physical stage positions in a 2x4 layout (see ANALYSIS_PLAN.md).

This writes one row per *discovered* tile (both datasets), including tiles
that failed to read and so have no usable stage coordinates - those are
listed with an empty well/well_position_id and flagged as unclustered.

Plate orientation (which position is "well 1") is NOT guessed here. Only a
neutral ``well_position_id`` (matching the position numbering already
recorded in ANALYSIS_PLAN.md) is assigned; the ``well`` column is left blank
pending user confirmation of orientation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_ims_qc import derive_round, discover_tiles  # noqa: E402

from czi_foci.ims_reader import inspect_ims, recover_metadata_from_sibling, stage_position_mm  # noqa: E402

CLUSTER_TOLERANCE_MM = 5.0
EXPECTED_N_CLUSTERS = 8

# The 8 physical stage positions already identified in ANALYSIS_PLAN.md (2x4
# layout). This is reference data to reproduce/confirm against, not a
# plate-orientation guess: it only numbers *positions*, not biological wells.
REFERENCE_POSITIONS = [
    (1, 5.5, -33.9),
    (2, -7.9, -33.9),
    (3, -21.2, -33.9),
    (4, -35.3, -34.9),
    (5, 5.0, -45.3),
    (6, -19.2, -45.9),
    (7, -33.5, -46.5),
    (8, -6.3, -46.6),
]


def load_trf2_positions(trf2_root: Path) -> pd.DataFrame:
    """Every TRF2 tile from stage0_tile_qc.csv, read or not."""
    df = pd.read_csv(trf2_root / "_analysis" / "stage0_tile_qc.csv")
    out = df[["round", "tile", "read_ok", "stage_x_mm", "stage_y_mm"]].copy()
    out["dataset"] = "TRF2"
    out.loc[out["read_ok"] != True, ["stage_x_mm", "stage_y_mm"]] = float("nan")  # noqa: E712
    return out[["dataset", "round", "tile", "stage_x_mm", "stage_y_mm"]]


def load_tsiv_positions(tsiv_root: Path) -> pd.DataFrame:
    """Stage position for every TSIV tile, from metadata only (no pixel read).

    TSIV was never run through Stage 0, so this does its own lightweight
    metadata-only scan (inspect_ims never touches the Data arrays), reusing
    the same folder discovery and round-labelling as Stage 0 for consistency.
    """
    output_dir = tsiv_root / "_analysis"
    tiles, _, _ = discover_tiles(tsiv_root, output_dir)
    rows = []
    for path, folder in tiles:
        round_label = derive_round(folder)
        inspection = inspect_ims(path)
        metadata = inspection.get("metadata")
        if metadata is None or not metadata.get("has_dataset_info"):
            metadata = recover_metadata_from_sibling(path)
        if metadata is None:
            rows.append({"dataset": "TSIV", "round": round_label, "tile": path.name, "stage_x_mm": float("nan"), "stage_y_mm": float("nan")})
            continue
        x_mm, y_mm = stage_position_mm(metadata)
        rows.append({"dataset": "TSIV", "round": round_label, "tile": path.name, "stage_x_mm": x_mm, "stage_y_mm": y_mm})
    return pd.DataFrame(rows)


def cluster_positions(xy: np.ndarray, tolerance_mm: float = CLUSTER_TOLERANCE_MM) -> np.ndarray:
    """Cluster (x, y) points at a fixed distance tolerance.

    Single-linkage hierarchical clustering with a distance cutoff: two points
    merge if *some* chain of neighbours each within ``tolerance_mm`` connects
    them. This is the right reading of "5 mm tolerance" for this data: TRF2's
    and TSIV's own tiles for the same physical field position are tightly
    packed (sub-mm) around two centres that are themselves ~3.5-5.1 mm apart
    (different slide/session), while every *different* position is tens of mm
    away. Complete linkage (cluster diameter <= tolerance) was tried first and
    splits that single field position into two, because it demands every
    cross-dataset pair be within tolerance rather than just the nearest ones -
    too strict for two co-located but offset tile grids.

    Returns 1-indexed cluster labels, one per row of ``xy``. A single point
    gets its own cluster of size 1.
    """
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([1], dtype=int)
    distances = pdist(xy, metric="euclidean")
    links = linkage(distances, method="single")
    return fcluster(links, t=tolerance_mm, criterion="distance")


def match_cluster_to_reference(centroid: tuple[float, float], tolerance_mm: float = CLUSTER_TOLERANCE_MM) -> int | None:
    best_id, best_dist = None, float("inf")
    for pos_id, rx, ry in REFERENCE_POSITIONS:
        dist = float(np.hypot(centroid[0] - rx, centroid[1] - ry))
        if dist < best_dist:
            best_id, best_dist = pos_id, dist
    if best_dist <= tolerance_mm:
        return best_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trf2-root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TRF2"))
    parser.add_argument("--tsiv-root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TSIV 2 hr"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.trf2_root / "_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    trf2 = load_trf2_positions(args.trf2_root)
    tsiv = load_tsiv_positions(args.tsiv_root)
    combined = pd.concat([trf2, tsiv], ignore_index=True)

    has_coords = combined["stage_x_mm"].notna() & combined["stage_y_mm"].notna()
    valid = combined[has_coords].copy()
    missing = combined[~has_coords].copy()
    print(f"{len(combined)} tiles discovered ({len(trf2)} TRF2, {len(tsiv)} TSIV); "
          f"{len(valid)} with usable stage coordinates, {len(missing)} without")

    xy = valid[["stage_x_mm", "stage_y_mm"]].to_numpy(dtype=float)
    labels = cluster_positions(xy)
    valid["cluster_id"] = labels

    n_clusters = len(set(labels))
    if n_clusters != EXPECTED_N_CLUSTERS:
        print(f"WARNING: expected {EXPECTED_N_CLUSTERS} clusters, found {n_clusters}")

    centroids = valid.groupby("cluster_id")[["stage_x_mm", "stage_y_mm"]].mean()
    cluster_to_position: dict[int, int | None] = {}
    for cluster_id, row in centroids.iterrows():
        cluster_to_position[cluster_id] = match_cluster_to_reference((row["stage_x_mm"], row["stage_y_mm"]))

    unmatched_clusters = [cid for cid, pos in cluster_to_position.items() if pos is None]
    if unmatched_clusters:
        print(f"WARNING: {len(unmatched_clusters)} cluster(s) did not match any reference position: {unmatched_clusters}")

    valid["well_position_id"] = valid["cluster_id"].map(cluster_to_position)
    valid["well"] = pd.NA  # plate orientation not guessed - pending user confirmation

    missing["cluster_id"] = pd.NA
    missing["well_position_id"] = pd.NA
    missing["well"] = pd.NA

    well_map = pd.concat([valid, missing], ignore_index=True)
    well_map = well_map[["tile", "round", "dataset", "well", "well_position_id", "stage_x_mm", "stage_y_mm", "cluster_id"]]
    well_map = well_map.sort_values(["well_position_id", "dataset", "round", "tile"], na_position="last")

    well_map_csv = output_dir / "well_map.csv"
    well_map.to_csv(well_map_csv, index=False)
    print(f"Wrote {well_map_csv}")

    # Reproduce the ANALYSIS_PLAN.md reference table from the data itself.
    print("\nReproduced position table (well identity/orientation NOT assigned):")
    print(f"{'pos':>3} {'stage_x_mm':>11} {'stage_y_mm':>11} {'TRF2 fields':<30} {'TSIV fields':<30} {'n_tiles':>8}")
    for pos_id, rx, ry in REFERENCE_POSITIONS:
        rows = valid[valid["well_position_id"] == pos_id]
        trf2_fields = sorted(set(rows[rows["dataset"] == "TRF2"]["round"]))
        tsiv_fields = sorted(set(rows[rows["dataset"] == "TSIV"]["round"]))
        print(f"{pos_id:>3} {rx:>11.1f} {ry:>11.1f} {','.join(trf2_fields):<30} {','.join(tsiv_fields):<30} {len(rows):>8}")

    unclustered = missing
    if len(unclustered):
        print(f"\n{len(unclustered)} tiles not clustered (no usable stage coordinates):")
        for _, row in unclustered.iterrows():
            print(f"  {row['dataset']}/{row['round']}/{row['tile']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
