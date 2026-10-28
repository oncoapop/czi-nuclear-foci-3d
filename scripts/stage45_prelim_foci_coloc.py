"""Preliminary per-nucleus foci counts and colocalization (Stages 4-6, provisional).

Run ahead of Stage 3, deliberately, to answer one question: are these data
good enough to carry the planned endpoints at all? Nothing here is frozen,
nothing here is a result. Detection thresholds are NOT calibrated (that is
Stage 3) and the nuclei come from Stage 0's 2.5D rule rather than Stage 4's
guarded segmentation.

WHAT IS FIXED RELATIVE TO STAGE 2
  Stage 2 thresholded each channel at median + k*1.4826*MAD of the DoG
  response inside nuclei. Foci are a positive tail of that distribution, so
  in a dense field the foci inflate the MAD and raise their own detection
  bar. Measured: well position 4 has the MOST nuclei (174/tile) and the
  FEWEST detections (4/tile), while position 7 has 27 nuclei and 15/tile.
  A count endpoint built on that would be anti-correlated with cell density -
  a technical artifact that, under blinding, is indistinguishable from a
  treatment effect.

  Fixed here by estimating the noise from the LOWER half of the DoG
  distribution only (see robust_noise_sigma). Foci can only push values up,
  so the lower half is focus-free and its spread is a density-independent
  estimate of the noise. This is a prerequisite for counting, not a tuning
  choice.

COMPARABILITY
  Stage 0 found the slide is tilted 4.5-7.0 um/mm, so each tile's in-focus
  slab differs (1.5-5.0 um) and sits at a different depth. A raw per-nucleus
  count is therefore partly a measure of how much of the nucleus was in
  focus. Counts are made inside a FIXED-THICKNESS slab centred on each
  tile's own focus plane, and reported both per nucleus and per um3 of
  sampled nuclear volume. Tiles whose usable slab cannot cover the fixed
  thickness are excluded and reported.

COLOCALIZATION
  No axial shift is applied: Stage 2 measured the channel offset at
  <= 0.093 um, within its own 0 +- 0.1 um acceptance band. Matching uses the
  plan's PSF-matched ellipsoid, sqrt((dx/0.3)^2+(dy/0.3)^2+(dz/0.9)^2) <= 1,
  not an isotropic radius. Every colocalization count is paired with a
  within-nucleus permutation null, and only observed-minus-expected and
  enrichment are reported - never the raw percentage, which is meaningless
  when it is dominated by chance coincidence at these densities.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_ims_qc import CHANNEL_LABELS, nucleus_labels  # noqa: E402
from stage2_registration import (  # noqa: E402
    ANALYSIS_RESOLUTION_LEVEL,
    MIN_FOCUS_SEPARATION_XY_UM,
    MIN_FOCUS_SEPARATION_Z_UM,
    _channel_index_map,
    difference_of_gaussians,
    flatten_background,
    load_flatfield,
    refine_centroids_xy,
)

from czi_foci.ims_reader import inspect_ims, read_channel_arrays_ims  # noqa: E402

BACKGROUND_SIGMA_UM = 2.5
SLAB_THICKNESS_UM = 1.5  # fixed analysis slab; 141/164 tiles and all 8 positions support it
# The lowest entry is the candidate-pool threshold, not an analysis
# threshold: it exists only to map where a detection CAN occur, so the
# permutation null has somewhere to move points to. It must sit far below
# the analysis thresholds - at k=3 the pool was only 1.7-2.6x the selected
# set in stained wells, so the null could barely randomize and every
# enrichment was squeezed toward 1.0. Counts reported at this threshold are
# noise and are labelled as such.
POOL_K = 1.5
K_SWEEP = (POOL_K, 4.0, 6.0, 8.0)
COLOC_TOLERANCE_XY_UM = 0.3  # PSF-matched ellipsoid semi-axes
COLOC_TOLERANCE_Z_UM = 0.9
N_NULL_PERMUTATIONS = 100
# NO expected telomere count is assumed. The textbook 92 (46 chromosomes x 2
# ends) describes a diploid cell with a normal karyotype, and nothing here
# qualifies: these are hTERT-immortalised p53-/- BRCA1-/- cells, which are
# aneuploid, and BRCA1 loss destabilises telomeres specifically. Sister
# telomeres and clustered telomeres also merge into single diffraction-
# limited foci, so even a known karyotype would not fix the expected count.
# Anchoring a detection threshold to 92 would therefore be calibrating to a
# number no publication establishes for this system. The threshold is chosen
# instead from the no-antibody control wells, which measure the false
# positive rate directly in these samples.
MIN_SPECIFICITY_FOR_PRIMARY_K = 10.0  # stained:control foci ratio required
SATURATED_DAPI_ROUNDS = ("_6", "_11")  # Stage 0: DAPI saturation blooms area and z-extent

# Confirmed design (lab notebook "Optimization of TRF2 + gH2AX Coloc on
# hTERT + BRCA-/-"). Stage positions are arbitrary cluster labels from Stage
# 1; the well they correspond to was fixed by two independent routes that
# agree on all eight: the chamber row/column geometry, and the student's
# well->field map. Wells 4 and 8 carry NO primary antibody in either channel
# and are the empirical false-positive floor, one per cell line.
# TRF2 (rabbit) 1:1000 -> RedX -> 561/594; gH2AX (mouse) 1:100 -> FITC -> 488.
DESIGN = {
    1.0: {"well": 1, "cell_line": "WT_hTERT", "treatment": "Palbociclib", "has_primary": True},
    2.0: {"well": 2, "cell_line": "WT_hTERT", "treatment": "Palbociclib", "has_primary": True},
    3.0: {"well": 3, "cell_line": "WT_hTERT", "treatment": "untreated", "has_primary": True},
    4.0: {"well": 4, "cell_line": "WT_hTERT", "treatment": "untreated", "has_primary": False},
    5.0: {"well": 5, "cell_line": "p53-/-BRCA1-/-", "treatment": "Palbociclib", "has_primary": True},
    8.0: {"well": 6, "cell_line": "p53-/-BRCA1-/-", "treatment": "Palbociclib", "has_primary": True},
    6.0: {"well": 7, "cell_line": "p53-/-BRCA1-/-", "treatment": "untreated", "has_primary": True},
    7.0: {"well": 8, "cell_line": "p53-/-BRCA1-/-", "treatment": "untreated", "has_primary": False},
}


# Standard-normal quantiles used to rebuild the noise scale and centre from
# the lower quartile alone: z(0.25) = -0.6745, z(0.025) = -1.9600.
_Z25, _Z2_5 = -0.6744897501960817, -1.9599639845400545
_LOWER_QUANTILE_SPAN = _Z25 - _Z2_5  # 1.2855 sigma


def robust_noise_sigma(values: np.ndarray) -> tuple[float, float]:
    """Noise centre and sigma estimated from the LOWER QUARTILE only.

    Foci are a one-sided positive tail, so any statistic that touches the
    upper half grows with focus density. A two-sided MAD does (that was the
    Stage 2 bug), and so does a "lower half about the median" estimate,
    because the tail drags the median itself upward and the anchor moves.

    Taking the spread between the 2.5th and 25th percentiles avoids the
    problem entirely: those quantiles sit below any plausible amount of
    signal, so the estimate is unaffected by contamination anywhere above
    the lower quartile. Both are rescaled to a Gaussian to recover sigma and
    the clean centre.
    """
    if values.size == 0:
        return 0.0, 0.0
    p2_5, p25 = np.percentile(values, [2.5, 25.0])
    sigma = float((p25 - p2_5) / _LOWER_QUANTILE_SPAN)
    if not np.isfinite(sigma) or sigma <= 0:
        return float(np.median(values)), 0.0
    centre = float(p25 - _Z25 * sigma)
    return centre, sigma


def place_slab(
    focus_idx: int, usable_lo: int, usable_hi: int, n_z: int, thickness_slices: int
) -> tuple[int, int] | None:
    """Fixed-thickness slab inside the usable range, as near the focus plane as it fits.

    The slab must be a constant thickness for counts to be comparable between
    tiles, but it must NOT be forced to centre on the focus plane: the slide
    is tilted, so Stage 0's usable range is often asymmetric about the focus
    peak, and insisting on centring throws away tiles whose usable range is
    plenty thick - it excluded every tile of field _8, the densest field in
    the set. Sliding the slab to fit keeps them at the same thickness.
    """
    usable_lo = max(usable_lo, 0)
    usable_hi = min(usable_hi, n_z - 1)
    if usable_hi - usable_lo + 1 < thickness_slices:
        return None
    lo = focus_idx - thickness_slices // 2
    lo = max(usable_lo, min(lo, usable_hi - thickness_slices + 1))
    return lo, lo + thickness_slices - 1


def detect_foci_in_slab(
    dog: np.ndarray,
    nuclear_labels_2d: np.ndarray,
    slab: tuple[int, int],
    pixel_size_um: float,
    z_step_um: float,
    mad_k: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Local maxima above a density-independent threshold, inside nuclei and slab."""
    z0, z1 = slab
    sub = dog[z0 : z1 + 1]
    mask = nuclear_labels_2d > 0
    inside = sub[:, mask]
    median, sigma = robust_noise_sigma(inside.ravel())
    threshold = median + mad_k * sigma

    rz = max(1, int(round(MIN_FOCUS_SEPARATION_Z_UM / z_step_um)))
    rxy = max(1, int(round(MIN_FOCUS_SEPARATION_XY_UM / pixel_size_um)))
    local_max = ndi.maximum_filter(sub, size=(2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), mode="nearest")
    peaks = (sub >= local_max) & (sub > threshold) & mask[None, :, :]
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3)), np.zeros(0), threshold, sigma
    amplitudes = sub[coords[:, 0], coords[:, 1], coords[:, 2]]
    centers = refine_centroids_xy(sub, coords)
    centers[:, 0] += z0  # back to absolute slice index
    return centers, amplitudes, threshold, sigma


def ellipsoidal_coloc(
    a_zyx_um: np.ndarray, b_zyx_um: np.ndarray
) -> tuple[int, int]:
    """Count foci of each set having a partner within the PSF-matched ellipsoid.

    Coordinates are pre-scaled by the ellipsoid semi-axes so that a plain
    unit-radius query implements
    sqrt((dx/0.3)^2 + (dy/0.3)^2 + (dz/0.9)^2) <= 1. An isotropic tolerance
    would be wrong in both directions at once on 5:1 anisotropic voxels:
    too permissive laterally, too strict axially.
    """
    if len(a_zyx_um) == 0 or len(b_zyx_um) == 0:
        return 0, 0
    tree_a = cKDTree(a_zyx_um)
    tree_b = cKDTree(b_zyx_um)
    a_has = tree_a.query_ball_tree(tree_b, r=1.0)
    b_has = tree_b.query_ball_tree(tree_a, r=1.0)
    return sum(1 for m in a_has if m), sum(1 for m in b_has if m)


def scale_for_ellipsoid(z_um: np.ndarray, y_um: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [z_um / COLOC_TOLERANCE_Z_UM, y_um / COLOC_TOLERANCE_XY_UM, x_um / COLOC_TOLERANCE_XY_UM]
    )


def permutation_null_coloc(
    a_scaled: np.ndarray,
    n_b: int,
    nucleus_pixels_yx: np.ndarray,
    slab: tuple[int, int],
    pixel_size_um: float,
    z_step_um: float,
    rng: np.random.Generator,
    n_permutations: int = N_NULL_PERMUTATIONS,
) -> float:
    """Expected colocalization if the second channel's foci were placed at random.

    Positions are redrawn inside THIS nucleus's own footprint and slab, so the
    null carries the same nuclear size, shape and slab thickness as the
    observation. A tile-wide or image-wide null would understate chance
    coincidence in small dense nuclei and overstate it in large sparse ones.

    The footprint is 2D propagated through the slab because the segmentation
    is 2.5D (Stage 0's rule); the null is therefore uniform in z within the
    slab, which is the honest null for the segmentation actually used.
    """
    if len(a_scaled) == 0 or n_b == 0 or len(nucleus_pixels_yx) == 0:
        return 0.0
    tree_a = cKDTree(a_scaled)
    z0, z1 = slab
    total = 0
    for _ in range(n_permutations):
        picks = nucleus_pixels_yx[rng.integers(len(nucleus_pixels_yx), size=n_b)]
        z = rng.uniform(z0, z1 + 1, size=n_b) * z_step_um
        b_scaled = scale_for_ellipsoid(z, picks[:, 0] * pixel_size_um, picks[:, 1] * pixel_size_um)
        hits = tree_a.query_ball_point(b_scaled, r=1.0)
        total += sum(1 for h in hits if h)
    return total / n_permutations


def candidate_pool_null(
    a_scaled: np.ndarray,
    n_b: int,
    pool_scaled: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = N_NULL_PERMUTATIONS,
) -> float:
    """Expected colocalization if the B foci were a random subset of the
    positions where a B detection could have occurred at all.

    THE UNIFORM NULL IS WRONG HERE, and the no-primary wells proved it: with
    no antibody in either channel they returned 4.8x and 7.8x "enrichment",
    higher than the stained wells. The cause is a common one, not
    colocalization - detections of any kind, signal or noise, concentrate
    where the local image is bright and textured, and both channels inherit
    that same landscape from the same nucleus and the same out-of-focus
    haze. A null that scatters points uniformly through the nuclear volume
    has no such structure, so it under-predicts coincidence and every well
    looks enriched.

    This null instead draws from the channel's OWN loose-threshold detections
    in that same nucleus. That pool traces the detectability landscape
    directly, so the comparison becomes the question actually worth asking:
    given where 488 signals can appear in this nucleus, is the set that
    survived the analysis threshold closer to the telomeres than an equally
    sized random draw from those candidates?

    Degenerate when the pool is no larger than the selection, which is why
    the analysis thresholds must be stricter than the pool's.
    """
    if len(a_scaled) == 0 or n_b == 0 or len(pool_scaled) < n_b + 1:
        return float("nan")
    tree_a = cKDTree(a_scaled)
    total = 0
    for _ in range(n_permutations):
        picks = rng.choice(len(pool_scaled), size=n_b, replace=False)
        hits = tree_a.query_ball_point(pool_scaled[picks], r=1.0)
        total += sum(1 for h in hits if h)
    return total / n_permutations


@dataclass
class TileResult:
    round_label: str
    tile: str
    well_position_id: float
    per_nucleus: pd.DataFrame
    slab_thickness_um: float
    n_nuclei_total: int
    n_nuclei_scored: int
    thresholds: dict
    error: str | None = None


def _process_tile(args) -> TileResult:
    row, input_dir, detect_k, seed = args
    try:
        return _process_tile_inner(row, input_dir, detect_k, seed)
    except Exception as exc:  # noqa: BLE001
        return TileResult(
            round_label=row["round"], tile=row["tile"],
            well_position_id=row.get("well_position_id", np.nan),
            per_nucleus=pd.DataFrame(), slab_thickness_um=np.nan,
            n_nuclei_total=0, n_nuclei_scored=0, thresholds={},
            error=f"{type(exc).__name__}: {exc}",
        )


def _process_tile_inner(row, input_dir: Path, detect_k: float, seed: int) -> TileResult:
    path = Path(row["source_path"])
    inspection = inspect_ims(path)
    arrays = read_channel_arrays_ims(path, inspection, mode="3d", resolution_level=ANALYSIS_RESOLUTION_LEVEL)
    index_by_name = _channel_index_map(row["channel_map"])
    pixel_size_um = float(row["pixel_size_um_level0"])
    z_step_um = float(row["z_step_um"])
    focus_idx = int(row["focus_peak_idx"])

    thickness_slices = max(1, int(round(SLAB_THICKNESS_UM / z_step_um)))
    usable_lo, usable_hi = (int(v) for v in str(row["usable_z_range"]).split("-"))
    n_z = arrays[index_by_name["DAPI_405"]].shape[0]
    slab = place_slab(focus_idx, usable_lo, usable_hi, n_z, thickness_slices)
    if slab is None:
        raise ValueError(
            f"usable slab {usable_lo}-{usable_hi} is thinner than {SLAB_THICKNESS_UM} um "
            f"({thickness_slices} slices at {z_step_um} um)"
        )
    slab_thickness_um = (slab[1] - slab[0] + 1) * z_step_um

    dapi = arrays[index_by_name["DAPI_405"]]
    labels_2d, _ = nucleus_labels(dapi, focus_idx, pixel_size_um)
    n_total = int(labels_2d.max())
    # Nuclei clipped by the tile border are partial objects; counting foci in
    # them would bias every per-nucleus number downward.
    labels_2d = segmentation.clear_border(labels_2d)
    labels_2d = measure.label(labels_2d > 0).astype(np.int32)
    n_scored = int(labels_2d.max())
    if n_scored == 0:
        raise ValueError("no interior nuclei after border exclusion")
    del dapi

    volumes = {}
    for name in ("RFP_561", "GFP_488"):
        volume = arrays[index_by_name[name]].astype(np.float32)
        field = load_flatfield(input_dir, row.get(f"flatfield_path_{name}"))
        if field is not None and field.shape == volume.shape[1:]:
            volume /= field[None, :, :]
        volumes[name] = volume
    arrays.clear()

    # Detect once at the loosest threshold in the sweep and carry each focus's
    # DoG amplitude, so every stricter k is an exact subset - no re-detection.
    k_min = min(K_SWEEP)
    foci = {}
    thresholds = {}
    for name, volume in volumes.items():
        dog = difference_of_gaussians(volume, pixel_size_um, z_step_um)
        centers, amplitudes, threshold, sigma = detect_foci_in_slab(
            dog, labels_2d, slab, pixel_size_um, z_step_um, k_min
        )
        centre = threshold - k_min * sigma
        foci[name] = (centers, amplitudes)
        thresholds[name] = {"centre": centre, "noise_sigma": sigma}
        del dog

    # Threshold-free damage burden: integrated background-flattened 488 in the
    # nucleus. 53BP1 sits near the noise floor and can go pan-nuclear, so a
    # count alone misses both extremes (ANALYSIS_PLAN Stage 5).
    # Threshold-free integrated signal per channel. For 488 this is the
    # damage burden that a count misses when 53BP1/gH2AX goes pan-nuclear.
    # For 561 it answers "TRF2 signal vs telomere-bound": the detected foci
    # are the bound fraction, while this total includes the diffuse
    # nucleoplasmic pool, so their ratio estimates how much of the TRF2
    # signal is actually at telomeres.
    slab_sums = {}
    for name in ("RFP_561", "GFP_488"):
        flat = flatten_background(volumes[name], BACKGROUND_SIGMA_UM, pixel_size_um)
        slab_sums[name] = flat[slab[0] : slab[1] + 1].sum(axis=0)
        del flat
    gfp_slab_sum = slab_sums["GFP_488"]
    del volumes

    voxel_um3 = pixel_size_um * pixel_size_um * z_step_um
    rng = np.random.default_rng(seed)
    regions = {r.label: r for r in measure.regionprops(labels_2d)}

    owners = {}
    for name in ("RFP_561", "GFP_488"):
        centers, amplitudes = foci[name]
        if len(centers):
            owners[name] = labels_2d[np.rint(centers[:, 1]).astype(int), np.rint(centers[:, 2]).astype(int)]
        else:
            owners[name] = np.zeros(0, dtype=int)

    # Per nucleus, every GFP detection at the loosest threshold: the
    # detectability landscape the null draws from.
    pool_by_label = {}
    centers_g, _ = foci["GFP_488"]
    for label in regions:
        p = centers_g[owners["GFP_488"] == label] if len(centers_g) else centers_g
        pool_by_label[label] = (
            scale_for_ellipsoid(p[:, 0] * z_step_um, p[:, 1] * pixel_size_um, p[:, 2] * pixel_size_um)
            if len(p) else np.zeros((0, 3))
        )

    design = DESIGN.get(float(row.get("well_position_id", float("nan"))), {})
    rows = []
    for detect_k in K_SWEEP:
        cuts = {
            name: thresholds[name]["centre"] + detect_k * thresholds[name]["noise_sigma"]
            for name in ("RFP_561", "GFP_488")
        }
        for label, region in regions.items():
            per_channel = {}
            for name in ("RFP_561", "GFP_488"):
                centers, amplitudes = foci[name]
                mine = (owners[name] == label) & (amplitudes >= cuts[name])
                per_channel[name] = centers[mine]
            a, b = per_channel["RFP_561"], per_channel["GFP_488"]
            a_scaled = (
                scale_for_ellipsoid(a[:, 0] * z_step_um, a[:, 1] * pixel_size_um, a[:, 2] * pixel_size_um)
                if len(a) else np.zeros((0, 3))
            )
            b_scaled = (
                scale_for_ellipsoid(b[:, 0] * z_step_um, b[:, 1] * pixel_size_um, b[:, 2] * pixel_size_um)
                if len(b) else np.zeros((0, 3))
            )
            n_rfp_coloc, n_gfp_coloc = ellipsoidal_coloc(a_scaled, b_scaled)
            expected = permutation_null_coloc(
                a_scaled, len(b), region.coords, slab, pixel_size_um, z_step_um, rng
            )
            expected_pool = candidate_pool_null(a_scaled, len(b), pool_by_label[label], rng)
            rows.append(
                {
                    "round": row["round"],
                    "tile": row["tile"],
                    "well_position_id": row.get("well_position_id", np.nan),
                    "well": design.get("well"),
                    "cell_line": design.get("cell_line"),
                    "treatment": design.get("treatment"),
                    "has_primary": design.get("has_primary"),
                    "detect_k": detect_k,
                    "nucleus_id": int(label),
                    "nucleus_area_um2": region.area * pixel_size_um**2,
                    "sampled_volume_um3": region.area * voxel_um3 * (slab[1] - slab[0] + 1),
                    "slab_thickness_um": slab_thickness_um,
                    "n_foci_RFP_561": len(a),
                    "n_foci_GFP_488": len(b),
                    "integrated_GFP_488": float(
                        gfp_slab_sum[region.coords[:, 0], region.coords[:, 1]].sum()
                    ),
                    "integrated_RFP_561": float(
                        slab_sums["RFP_561"][region.coords[:, 0], region.coords[:, 1]].sum()
                    ),
                    "n_coloc_GFP_with_RFP": n_gfp_coloc,
                    "n_coloc_RFP_with_GFP": n_rfp_coloc,
                    "coloc_expected_null": expected,
                    "coloc_expected_pool_null": expected_pool,
                    "flatfield_applied": bool(row.get("flatfield_applied", False)),
                    "dapi_saturated_round": row["round"] in SATURATED_DAPI_ROUNDS,
                }
            )
    return TileResult(
        round_label=row["round"], tile=row["tile"],
        well_position_id=row.get("well_position_id", np.nan),
        per_nucleus=pd.DataFrame(rows), slab_thickness_um=slab_thickness_um,
        n_nuclei_total=n_total, n_nuclei_scored=n_scored, thresholds=thresholds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TRF2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rounds", type=str, default=None)
    parser.add_argument("--include-deconvolved", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "_analysis")
    input_dir = args.input_dir or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_df = pd.read_csv(input_dir / "stage0_tile_qc.csv")
    usable = tile_df[(tile_df["read_ok"] == True) & tile_df["usable_z_range"].notna()].copy()  # noqa: E712
    if not args.include_deconvolved:
        usable = usable[~usable["round"].str.contains("deconvolved")]
    if args.rounds:
        usable = usable[usable["round"].isin({r.strip() for r in args.rounds.split(",")})]
    if args.limit:
        usable = usable.head(args.limit)
    print(f"Prelim foci/coloc: {len(usable)} tiles, k sweep={K_SWEEP}, "
          f"slab={SLAB_THICKNESS_UM} um ({args.workers} workers)")

    tasks = [(row, input_dir, min(K_SWEEP), 7000 + i) for i, (_, row) in enumerate(usable.iterrows())]
    results = []
    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_tile, tasks, chunksize=1):
            results.append(result)
    excluded = [r for r in results if r.error]
    good = [r for r in results if r.error is None and len(r.per_nucleus)]
    slab_failures = [r for r in excluded if "cannot cover" in (r.error or "")]
    print(f"Scored {len(good)}/{len(results)} tiles "
          f"({len(slab_failures)} excluded: usable slab too thin for a {SLAB_THICKNESS_UM} um slab)")
    for r in excluded:
        if r not in slab_failures:
            print(f"  WARNING {r.round_label}/{r.tile}: {r.error}")
    if not good:
        print("Nothing scored.")
        return 1

    nuclei = pd.concat([r.per_nucleus for r in good], ignore_index=True)
    nuclei["foci_RFP_per_um3"] = nuclei["n_foci_RFP_561"] / nuclei["sampled_volume_um3"]
    nuclei["foci_GFP_per_um3"] = nuclei["n_foci_GFP_488"] / nuclei["sampled_volume_um3"]
    nuclei["coloc_minus_null"] = nuclei["n_coloc_GFP_with_RFP"] - nuclei["coloc_expected_null"]

    nuclei.to_csv(output_dir / "stage45_prelim_per_nucleus.csv", index=False)

    per_tile = nuclei.groupby(["round", "tile", "well_position_id", "detect_k"], as_index=False).agg(
        n_nuclei=("nucleus_id", "size"),
        mean_RFP=("n_foci_RFP_561", "mean"),
        mean_GFP=("n_foci_GFP_488", "mean"),
        mean_area_um2=("nucleus_area_um2", "mean"),
        mean_coloc=("n_coloc_GFP_with_RFP", "mean"),
        mean_expected=("coloc_expected_null", "mean"),
        mean_integrated_GFP=("integrated_GFP_488", "mean"),
    )
    per_tile.to_csv(output_dir / "stage45_prelim_per_tile.csv", index=False)

    per_sample = _summarize_per_sample(nuclei, per_tile)
    per_sample.to_csv(output_dir / "stage45_prelim_per_sample.csv", index=False)

    _print_summary(nuclei, per_tile, per_sample, good)
    _write_pdf(output_dir / "stage45_prelim.pdf", nuclei, per_tile, per_sample)
    print(f"\nWrote {output_dir / 'stage45_prelim.pdf'}")
    return 0


def _summarize_per_sample(nuclei: pd.DataFrame, per_tile: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to sample level with TILE-level bootstrap CIs.

    Nuclei within a tile are not independent (shared focus, illumination,
    staining and segmentation), so the bootstrap resamples tiles. 11,000
    nuclei are not 11,000 observations.
    """
    rows = []
    for (position, detect_k), group in nuclei.groupby(["well_position_id", "detect_k"]):
        tiles = group["tile"].unique()
        rng = np.random.default_rng(int(position) * 13 + 1)
        boots = []
        by_tile = {t: group[group["tile"] == t] for t in tiles}
        for _ in range(400):
            drawn = rng.choice(tiles, size=len(tiles), replace=True)
            sample = pd.concat([by_tile[t] for t in drawn])
            observed = sample["n_coloc_GFP_with_RFP"].sum()
            expected = sample["coloc_expected_pool_null"].sum()
            boots.append(observed - expected)
        observed_total = group["n_coloc_GFP_with_RFP"].sum()
        expected_total = group["coloc_expected_pool_null"].sum()
        uniform_total = group["coloc_expected_null"].sum()
        rows.append(
            {
                "well_position_id": position,
                "well": group["well"].iloc[0],
                "cell_line": group["cell_line"].iloc[0],
                "treatment": group["treatment"].iloc[0],
                "has_primary": group["has_primary"].iloc[0],
                "detect_k": detect_k,
                "rounds": ",".join(sorted(group["round"].unique())),
                "n_tiles": len(tiles),
                "n_nuclei": len(group),
                "mean_foci_RFP_561": group["n_foci_RFP_561"].mean(),
                "mean_foci_GFP_488": group["n_foci_GFP_488"].mean(),
                "median_foci_RFP_561": group["n_foci_RFP_561"].median(),
                "median_foci_GFP_488": group["n_foci_GFP_488"].median(),
                "mean_nucleus_area_um2": group["nucleus_area_um2"].mean(),
                "RFP_per_um3": group["foci_RFP_per_um3"].mean(),
                "GFP_per_um3": group["foci_GFP_per_um3"].mean(),
                "mean_integrated_GFP_488": group["integrated_GFP_488"].mean(),
                "coloc_observed_total": observed_total,
                "coloc_expected_total": expected_total,
                "coloc_expected_uniform_null": uniform_total,
                "coloc_excess_total": observed_total - expected_total,
                "coloc_excess_ci_lo": float(np.percentile(boots, 2.5)),
                "coloc_excess_ci_hi": float(np.percentile(boots, 97.5)),
                "coloc_enrichment": (observed_total / expected_total) if expected_total > 0 else np.nan,
                "coloc_enrichment_uniform_null": (observed_total / uniform_total) if uniform_total > 0 else np.nan,
                "pct_of_GFP_coloc": 100.0 * observed_total / max(group["n_foci_GFP_488"].sum(), 1),
                "frac_nuclei_with_any_GFP": float((group["n_foci_GFP_488"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _print_summary(nuclei, per_tile, per_sample, good) -> None:
    slab = nuclei["slab_thickness_um"].median()
    print(f"\nFixed analysis slab {slab:.2f} um. No expected telomere count is assumed - these are")
    print("aneuploid p53-/-BRCA1-/- cells and no published figure covers this system, so the")
    print("threshold is set from the no-antibody controls below, not from a target count.")

    print("\n=== THRESHOLD SWEEP (no threshold is calibrated yet - that is Stage 3) ===")
    print(f"{'k':>4} {'RFP/nuc':>9} {'GFP/nuc':>9} {'nuc w/ GFP':>11} {'coloc obs':>10} "
          f"{'exp null':>9} {'enrich':>7}")
    for k in sorted(nuclei["detect_k"].unique()):
        sub = nuclei[nuclei["detect_k"] == k]
        observed = sub["n_coloc_GFP_with_RFP"].sum()
        expected = sub["coloc_expected_null"].sum()
        print(f"{k:>4.0f} {sub['n_foci_RFP_561'].mean():>9.1f} {sub['n_foci_GFP_488'].mean():>9.2f} "
              f"{(sub['n_foci_GFP_488'] > 0).mean():>11.2%} {observed:>10.0f} {expected:>9.1f} "
              f"{(observed / expected if expected > 0 else np.nan):>7.2f}")

    print("\n=== (1) PER-NUCLEUS FOCI COUNTS, each channel independently ===")
    for k in sorted(per_sample["detect_k"].unique()):
        print(f"\n-- detect k = {k:.0f} --")
        print(f"{'pos':>4} {'rounds':>14} {'tiles':>6} {'nuclei':>7} {'RFP/nuc':>8} {'GFP/nuc':>8} "
              f"{'RFP/um3':>9} {'area um2':>9}")
        for _, r in per_sample[per_sample["detect_k"] == k].sort_values("well_position_id").iterrows():
            print(f"{r['well_position_id']:>4.0f} {r['rounds']:>14} {r['n_tiles']:>6.0f} {r['n_nuclei']:>7.0f} "
                  f"{r['mean_foci_RFP_561']:>8.1f} {r['mean_foci_GFP_488']:>8.2f} "
                  f"{r['RFP_per_um3']:>9.4f} {r['mean_nucleus_area_um2']:>9.1f}")

    print("\nDensity-coupling check (the Stage 2 bug): foci-per-nucleus vs nuclei-per-tile")
    print("A strong negative r would mean the threshold is still starving dense fields.")
    for k in sorted(per_tile["detect_k"].unique()):
        sub = per_tile[per_tile["detect_k"] == k]
        if sub["n_nuclei"].nunique() < 3:
            continue
        r_rfp = np.corrcoef(sub["n_nuclei"], sub["mean_RFP"])[0, 1]
        r_gfp = np.corrcoef(sub["n_nuclei"], sub["mean_GFP"])[0, 1]
        print(f"  k={k:>4.0f}  RFP r = {r_rfp:+.3f}   GFP r = {r_gfp:+.3f}")

    print("\n=== (1b) TELOMERE-COUNT STABILITY, not absolute number ===")
    print("The absolute count is not interpretable here (aneuploid line, merged sister/clustered")
    print("telomeres, antibody at 1:1000). The DISPERSION is, and it maps onto the biology:")
    print("  - Palbociclib arrests in G1, fixing chromosome number, so it should TIGHTEN the")
    print("    distribution relative to cycling cells spanning 2N-4N. If it does not, the counts")
    print("    are not tracking telomeres at all.")
    print("  - BRCA1 loss destabilises telomeres, so p53-/-BRCA1-/- is expected to be MORE")
    print("    variable than WT. Compare CV within matched treatment.")
    stained_nuc = nuclei[nuclei["has_primary"] == True]  # noqa: E712
    print(f"\n{'cell line':>16} {'treatment':>12} {'k':>3} {'nuclei':>7} {'mean':>7} {'SD':>7} "
          f"{'CV':>6} {'bound frac':>11}")
    for k in sorted(stained_nuc["detect_k"].unique()):
        for (line, treat), g in stained_nuc[stained_nuc["detect_k"] == k].groupby(["cell_line", "treatment"]):
            counts = g["n_foci_RFP_561"]
            cv = counts.std() / counts.mean() if counts.mean() > 0 else float("nan")
            bound = (counts / g["integrated_RFP_561"].replace(0, np.nan)).mean() * 1e6
            print(f"{str(line):>16} {str(treat):>12} {k:>3.0f} {len(g):>7} {counts.mean():>7.2f} "
                  f"{counts.std():>7.2f} {cv:>6.2f} {bound:>11.3f}")

    print("\n=== (2) COLOCALIZATION - CONTROL VALIDATION FIRST ===")
    print("Wells 4 and 8 have no primary antibody in either channel. Any enrichment")
    print("measure MUST read ~1.0 there. The uniform null failed this; the")
    print("candidate-pool null is the replacement. Both shown so the fix is auditable.")
    print(f"\n{'well':>5} {'cell line':>16} {'treatment':>12} {'k':>3} {'obs':>6} "
          f"{'pool null':>10} {'enrich':>7} | {'unif null':>10} {'enrich':>7}")
    for primary in (False, True):
        for _, r in per_sample[per_sample["has_primary"] == primary].sort_values(
            ["well", "detect_k"]
        ).iterrows():
            tag = "  <- CONTROL" if not primary else ""
            print(f"{r['well']:>5.0f} {str(r['cell_line']):>16} {str(r['treatment']):>12} "
                  f"{r['detect_k']:>3.0f} {r['coloc_observed_total']:>6.0f} "
                  f"{r['coloc_expected_total']:>10.1f} {r['coloc_enrichment']:>7.2f} | "
                  f"{r['coloc_expected_uniform_null']:>10.1f} "
                  f"{r['coloc_enrichment_uniform_null']:>7.2f}{tag}")

    print("\n=== (3) BY DESIGN: cell line x treatment (stained wells only) ===")
    stained = per_sample[per_sample["has_primary"] == True]  # noqa: E712
    for k in sorted(stained["detect_k"].unique()):
        sub = stained[stained["detect_k"] == k]
        if not len(sub):
            continue
        print(f"\n-- detect k = {k:.0f} --")
        print(f"{'cell line':>16} {'treatment':>12} {'wells':>7} {'nuclei':>7} {'TRF2/nuc':>9} "
              f"{'gH2AX/nuc':>10} {'coloc obs':>10} {'pool null':>10} {'enrich':>7}")
        for (line, treat), g in sub.groupby(["cell_line", "treatment"]):
            obs = g["coloc_observed_total"].sum()
            exp = g["coloc_expected_total"].sum()
            n_nuc = g["n_nuclei"].sum()
            trf2 = (g["mean_foci_RFP_561"] * g["n_nuclei"]).sum() / n_nuc
            h2ax = (g["mean_foci_GFP_488"] * g["n_nuclei"]).sum() / n_nuc
            wells = ",".join(str(int(w)) for w in sorted(g["well"].unique()))
            print(f"{str(line):>16} {str(treat):>12} {wells:>7} {n_nuc:>7.0f} {trf2:>9.2f} "
                  f"{h2ax:>10.2f} {obs:>10.0f} {exp:>10.1f} "
                  f"{(obs / exp if exp > 0 else float('nan')):>7.2f}")


def _write_pdf(path: Path, nuclei_all, per_tile_all, per_sample_all) -> None:
    # Plot at the threshold whose median RFP count comes closest to the
    # telomere plausibility bound - a provisional operating point for looking
    # at the data, NOT a calibration. Stage 3 sets the real one.
    slab = nuclei_all["slab_thickness_um"].median()
    # Pick the plotting threshold from measured specificity, NOT from an
    # assumed telomere count: the lowest analysis threshold at which stained
    # wells exceed the no-antibody controls by MIN_SPECIFICITY_FOR_PRIMARY_K.
    by_k = nuclei_all.groupby("detect_k")["n_foci_RFP_561"].mean()
    plot_k = float(max(k for k in K_SWEEP if k > POOL_K))
    for k in sorted(x for x in K_SWEEP if x > POOL_K):
        at_k = nuclei_all[nuclei_all["detect_k"] == k]
        control = at_k[at_k["has_primary"] == False]["n_foci_RFP_561"].mean()  # noqa: E712
        stained = at_k[at_k["has_primary"] == True]["n_foci_RFP_561"].mean()  # noqa: E712
        if control > 0 and stained / control >= MIN_SPECIFICITY_FOR_PRIMARY_K:
            plot_k = float(k)
            break
    nuclei = nuclei_all[nuclei_all["detect_k"] == plot_k]
    per_tile = per_tile_all[per_tile_all["detect_k"] == plot_k]
    per_sample = per_sample_all[per_sample_all["detect_k"] == plot_k]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        fig.suptitle("Threshold sweep: counts and colocalization vs detection threshold", fontsize=12)
        ax = fig.add_subplot(2, 2, 1)
        ax.plot(by_k.index, by_k.values, marker="o")
        control_by_k = nuclei_all[nuclei_all["has_primary"] == False].groupby(  # noqa: E712
            "detect_k")["n_foci_RFP_561"].mean()
        ax.plot(control_by_k.index, control_by_k.values, marker="s", lw=1, color="crimson",
                label="no-antibody control (false positives)")
        ax.set_xlabel("detection threshold k (sigma above noise)")
        ax.set_ylabel("mean RFP_561 foci per nucleus")
        ax.legend(fontsize=6)
        ax.set_title("plausibility bound", fontsize=8)

        ax2 = fig.add_subplot(2, 2, 2)
        gfp = nuclei_all.groupby("detect_k")["n_foci_GFP_488"].mean()
        ax2.plot(gfp.index, gfp.values, marker="o", color="tab:green")
        ax2.set_xlabel("detection threshold k")
        ax2.set_ylabel("mean GFP_488 foci per nucleus")
        ax2.set_title("53BP1 detection vs threshold", fontsize=8)

        ax3 = fig.add_subplot(2, 2, 3)
        agg = nuclei_all.groupby("detect_k").agg(
            obs=("n_coloc_GFP_with_RFP", "sum"), exp=("coloc_expected_null", "sum")
        )
        ax3.plot(agg.index, agg["obs"], marker="o", label="observed")
        ax3.plot(agg.index, agg["exp"], marker="s", label="permuted null")
        ax3.set_xlabel("detection threshold k")
        ax3.set_ylabel("colocalization events (pooled)")
        ax3.legend(fontsize=7)
        ax3.set_title("observed vs chance", fontsize=8)

        ax4 = fig.add_subplot(2, 2, 4)
        enrichment = agg["obs"] / agg["exp"].replace(0, np.nan)
        ax4.plot(enrichment.index, enrichment.values, marker="o", color="tab:purple")
        ax4.axhline(1.0, color="grey", lw=0.8)
        ax4.set_xlabel("detection threshold k")
        ax4.set_ylabel("enrichment (observed / null)")
        ax4.set_title("1.0 = indistinguishable from chance", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        fig.suptitle(f"Preliminary per-nucleus counts and colocalization at k={plot_k:.0f} "
                     f"(provisional, NOT calibrated)", fontsize=12)
        ax = fig.add_axes((0.04, 0.55, 0.92, 0.33))
        ax.axis("off")
        show = per_sample[["well_position_id", "rounds", "n_tiles", "n_nuclei", "mean_foci_RFP_561",
                           "mean_foci_GFP_488", "RFP_per_um3", "GFP_per_um3", "coloc_observed_total",
                           "coloc_expected_total", "coloc_excess_total", "coloc_enrichment"]].copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].map(lambda v: f"{v:.3g}" if pd.notna(v) else "n/a")
        table = ax.table(cellText=show.values, colLabels=list(show.columns), loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(5.5)
        table.scale(1, 1.3)

        ax2 = fig.add_axes((0.08, 0.08, 0.38, 0.38))
        ax2.scatter(per_tile["n_nuclei"], per_tile["mean_RFP"], s=14, alpha=0.7, label="RFP_561")
        ax2.scatter(per_tile["n_nuclei"], per_tile["mean_GFP"], s=14, alpha=0.7, label="GFP_488")
        ax2.set_xlabel("nuclei per tile")
        ax2.set_ylabel("mean foci per nucleus")
        ax2.set_title("density-coupling check: should show no trend", fontsize=8)
        ax2.legend(fontsize=7)

        ax3 = fig.add_axes((0.57, 0.08, 0.38, 0.38))
        positions = per_sample.sort_values("well_position_id")
        lo = (positions["coloc_excess_total"] - positions["coloc_excess_ci_lo"]).abs()
        hi = (positions["coloc_excess_ci_hi"] - positions["coloc_excess_total"]).abs()
        ax3.errorbar(positions["well_position_id"], positions["coloc_excess_total"],
                     yerr=[lo, hi], fmt="o", capsize=3)
        ax3.axhline(0, color="grey", lw=0.8)
        ax3.set_xlabel("well position id")
        ax3.set_ylabel("colocalization excess over null")
        ax3.set_title("observed - expected, tile-level bootstrap CI", fontsize=8)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        ax = fig.add_subplot(2, 2, 1)
        ax.hist(nuclei["n_foci_RFP_561"], bins=50)
        ax.set_xlabel("RFP_561 foci per nucleus")
        ax.set_ylabel("nuclei")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(nuclei["n_foci_GFP_488"], bins=50)
        ax2.set_xlabel("GFP_488 foci per nucleus")
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(nuclei["n_foci_RFP_561"], nuclei["n_foci_GFP_488"], s=3, alpha=0.2)
        ax3.set_xlabel("RFP_561 foci")
        ax3.set_ylabel("GFP_488 foci")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(nuclei["coloc_minus_null"], bins=50)
        ax4.axvline(0, color="crimson", lw=1)
        ax4.set_xlabel("per-nucleus colocalization minus null")
        fig.suptitle("Distributions across all scored nuclei (pooled, provisional)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
