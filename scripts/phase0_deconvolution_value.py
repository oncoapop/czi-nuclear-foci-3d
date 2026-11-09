"""Phase 0: does Andor Fusion's deconvolution add information, or just change the numbers?

Deconvolution on this instrument ran in real time during acquisition (~0.4 s per
plane), so the cost is not time but storage: every deconvolved acquisition is a
second full copy, 18.6 GB of the 111 GB archive. The question is therefore
whether it earns that, not whether it is worth waiting for.

The decisive test is matched-detection adjudication. Detect foci in the raw and
deconvolved copies of the SAME field, match them, and classify each detection as
both / raw-only / deconvolved-only. Then adjudicate every disagreement against
the *other* image's pixel values:

  - a deconvolved-only focus sitting where the raw image has genuinely elevated
    (if sub-threshold) signal means deconvolution RECOVERED a real focus;
  - one sitting at raw background is an ARTEFACT.

That is decidable without ground truth, which matters here because no bead slide
was acquired.

Two optical facts constrain how far the conclusion generalises, and both are
printed in the report:

  - This is widefield (``MicroscopeMode = Widefield``, ``LSMPinhole = 0``), so
    there is a lot of out-of-focus light to remove but also a missing cone.
  - Z is sampled BELOW Nyquist in every acquisition (axial Nyquist 292-393 nm
    depending on channel; actual dz is 380-740 nm). Deconvolution cannot restore
    axial frequencies that were never sampled, so some of what it adds in Z is
    interpolation. This is the leading candidate explanation for any excess of
    deconvolved-only detections.

Nuclear masks come from the RAW DAPI channel and are shared between the two
images, so detection differences are not confounded with segmentation
differences. Whether segmentation itself changes is measured separately.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from czi_foci.ims_reader import (  # noqa: E402
    inspect_ims,
    pixel_size_um_ims,
    read_channel_arrays_ims,
    resolve_channel_indices,
)

# Detection parameters. Deliberately the same for raw and deconvolved — the
# whole point is to hold the detector fixed and vary only the image.
MAD_MULTIPLIER = 5.0
DOG_SIGMA_XY_SMALL = 1.0
DOG_SIGMA_XY_LARGE = 3.0
DOG_SIGMA_Z_SMALL = 0.6
DOG_SIGMA_Z_LARGE = 2.0
PEAK_MIN_DISTANCE_PX = 3

# A detection in one image is "the same focus" as one in the other if their
# centres are within this physical distance. Voxels are strongly anisotropic
# (XY 0.1 um, Z 0.38-0.74 um), so a fixed 0.3 um radius is SUB-VOXEL in Z and
# any one-plane shift of a peak breaks the match. The effective radius is
# therefore widened to at least ~1.2 voxels on the coarsest axis.
MATCH_RADIUS_UM = 0.3
MATCH_RADIUS_VOXEL_FACTOR = 1.2

# Adjudication is against an empirical null: the DoG response at random
# positions inside the same nuclear mask. A detection present in only one
# image is "recovered" if the other image's response at that position exceeds
# what background alone produces, and "artefact" if it does not. A fixed
# MAD-sigma cut cannot do this job — inside a nucleus almost every position is
# elevated relative to the global median.
NULL_SAMPLE_SIZE = 20000
NULL_PERCENTILE = 95.0

NUCLEUS_MIN_SIZE_VOX_3D = 20000
NUCLEUS_MIN_SIZE_VOX_2D = 3000

# An "isolated" punctum for FWHM measurement has no other detection within this
# radius, so its profile is not contaminated by a neighbour.
ISOLATION_RADIUS_UM = 1.0


@dataclass
class PairResult:
    rows: list[dict] = field(default_factory=list)
    fwhm_rows: list[dict] = field(default_factory=list)


def nuclear_mask(dapi: np.ndarray) -> np.ndarray:
    """Otsu on DAPI, opened, with small objects removed. 2-D-safe."""
    is_3d = dapi.ndim == 3 and dapi.shape[0] > 1
    mask = dapi > threshold_otsu(dapi)
    footprint = np.ones((1, 5, 5)) if dapi.ndim == 3 else np.ones((5, 5))
    mask = ndi.binary_opening(mask, footprint)
    min_size = NUCLEUS_MIN_SIZE_VOX_3D if is_3d else NUCLEUS_MIN_SIZE_VOX_2D
    return remove_small_objects(mask, min_size)


def difference_of_gaussians(image: np.ndarray) -> np.ndarray:
    """Band-pass that suppresses background without the top-hat's flat-top bias."""
    if image.ndim == 3 and image.shape[0] > 1:
        small = (DOG_SIGMA_Z_SMALL, DOG_SIGMA_XY_SMALL, DOG_SIGMA_XY_SMALL)
        large = (DOG_SIGMA_Z_LARGE, DOG_SIGMA_XY_LARGE, DOG_SIGMA_XY_LARGE)
    elif image.ndim == 3:
        small = (0.0, DOG_SIGMA_XY_SMALL, DOG_SIGMA_XY_SMALL)
        large = (0.0, DOG_SIGMA_XY_LARGE, DOG_SIGMA_XY_LARGE)
    else:
        small = (DOG_SIGMA_XY_SMALL,) * 2
        large = (DOG_SIGMA_XY_LARGE,) * 2
    return ndi.gaussian_filter(image, small) - ndi.gaussian_filter(image, large)


def mad_stats(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-sigma (1.4826 * MAD), the scale used by the threshold."""
    median = float(np.median(values))
    mad_sigma = float(1.4826 * np.median(np.abs(values - median)))
    return median, max(mad_sigma, 1e-9)


def detect(dog: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (peak coordinates, median, mad_sigma) measured inside the mask."""
    inside = dog[mask]
    median, mad_sigma = mad_stats(inside)
    threshold = median + MAD_MULTIPLIER * mad_sigma
    peaks = peak_local_max(
        dog,
        min_distance=PEAK_MIN_DISTANCE_PX,
        threshold_abs=threshold,
        labels=mask,
        exclude_border=False,
    )
    return peaks, median, mad_sigma


def to_physical(peaks: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if peaks.size == 0:
        return peaks.reshape(0, len(spacing))
    return peaks * np.asarray(spacing, dtype=float)


def zscore_at(dog: np.ndarray, peaks: np.ndarray, median: float, mad_sigma: float) -> np.ndarray:
    """DoG response of one image at the other image's peak positions, in MAD-sigma."""
    if peaks.size == 0:
        return np.zeros(0)
    idx = tuple(peaks[:, i] for i in range(peaks.shape[1]))
    return (dog[idx] - median) / mad_sigma


def null_zscores(
    dog: np.ndarray,
    mask: np.ndarray,
    median: float,
    mad_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """DoG response at random positions inside the nuclear mask.

    This is the baseline a one-image-only detection must beat to count as
    recovered rather than invented. Without it, "elevated relative to the
    global median" is nearly vacuous inside a nucleus.
    """
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return np.zeros(0)
    take = min(NULL_SAMPLE_SIZE, coords.shape[0])
    sample = coords[rng.choice(coords.shape[0], size=take, replace=False)]
    return zscore_at(dog, sample, median, mad_sigma)


FWHM_WINDOW_UM = 1.5


def half_max_width(profile: np.ndarray, centre: int, spacing_um: float) -> float:
    """FWHM of a 1-D profile, linearly interpolated, measured in a bounded window.

    The window matters. Taken over a whole 2048-px line, a dim punctum sitting
    on bright nuclear background never falls to half maximum, and the returned
    "width" is then the width of the nucleus rather than of the punctum —
    which is how a 0.1 um focus can appear to be 9 um across. Restricting the
    search to +/- FWHM_WINDOW_UM and returning nan when no crossing is found
    inside it reports "not measurable" instead of a wrong number.
    """
    if spacing_um <= 0 or not np.isfinite(spacing_um):
        return float("nan")
    half_window = max(3, int(round(FWHM_WINDOW_UM / spacing_um)))
    lo = max(0, centre - half_window)
    hi = min(profile.size, centre + half_window + 1)
    window = profile[lo:hi]
    local_centre = centre - lo

    peak = float(window[local_centre])
    baseline = float(window.min())
    if peak <= baseline:
        return float("nan")
    half = baseline + (peak - baseline) / 2.0

    def crossing(direction: int) -> float | None:
        position = local_centre
        while 0 <= position + direction < window.size:
            nxt = position + direction
            if window[nxt] <= half:
                span = window[position] - window[nxt]
                frac = (window[position] - half) / span if span > 0 else 0.0
                return abs(nxt - local_centre) - (1.0 - frac)
            position = nxt
        return None

    left = crossing(-1)
    right = crossing(1)
    if left is None or right is None:
        # Never reaches half maximum within the window: not a resolvable
        # punctum on this profile.
        return float("nan")
    return (left + right) * spacing_um


def measure_fwhm(
    image: np.ndarray,
    peaks: np.ndarray,
    spacing: tuple[float, float, float],
    max_puncta: int = 40,
) -> list[dict]:
    """FWHM in XY and Z for isolated puncta. Axial narrowing is what 3-D
    deconvolution claims to do, so it is measured even though Z is undersampled."""
    if peaks.shape[0] < 2:
        return []
    physical = to_physical(peaks, spacing)
    tree = cKDTree(physical)
    neighbours = tree.query_ball_point(physical, r=ISOLATION_RADIUS_UM)
    isolated = [i for i, n in enumerate(neighbours) if len(n) == 1]
    if not isolated:
        return []

    intensities = image[tuple(peaks[:, i] for i in range(peaks.shape[1]))]
    isolated.sort(key=lambda i: -intensities[i])
    rows: list[dict] = []
    is_3d = image.ndim == 3 and image.shape[0] > 1

    for i in isolated[:max_puncta]:
        coord = peaks[i]
        if is_3d:
            z, y, x = (int(v) for v in coord)
            fwhm_x = half_max_width(image[z, y, :].astype(float), x, spacing[2])
            fwhm_y = half_max_width(image[z, :, x].astype(float), y, spacing[1])
            fwhm_z = half_max_width(image[:, y, x].astype(float), z, spacing[0])
        else:
            if coord.size == 3:
                _, y, x = (int(v) for v in coord)
                plane = image[0] if image.ndim == 3 else image
            else:
                y, x = (int(v) for v in coord)
                plane = image
            fwhm_x = half_max_width(plane[y, :].astype(float), x, spacing[2])
            fwhm_y = half_max_width(plane[:, x].astype(float), y, spacing[1])
            fwhm_z = float("nan")
        rows.append({"fwhm_x_um": fwhm_x, "fwhm_y_um": fwhm_y, "fwhm_z_um": fwhm_z})
    return rows


def compare_pair(raw_path: Path, dec_path: Path, label: str) -> PairResult:
    """Run the full battery on one matched raw/deconvolved field."""
    result = PairResult()

    rng = np.random.default_rng(20261002)
    raw_inspection = inspect_ims(raw_path)
    dec_inspection = inspect_ims(dec_path)
    if raw_inspection.get("errors") and not raw_inspection.get("channels"):
        return result
    if dec_inspection.get("errors") and not dec_inspection.get("channels"):
        return result

    metadata = raw_inspection["metadata"]
    channels = resolve_channel_indices(metadata)
    spacing = pixel_size_um_ims(metadata, mode="3d", resolution_level=0)
    true_z = int(raw_inspection["true_z"])
    is_3d = true_z > 1
    if not is_3d:
        # Single plane: no z-step exists. Use the XY spacing for the degenerate
        # axis so physical distances stay well defined in the matcher.
        spacing = (spacing[1], spacing[1], spacing[2])

    raw_arrays = read_channel_arrays_ims(raw_path, raw_inspection, mode="3d", resolution_level=0)
    dec_arrays = read_channel_arrays_ims(dec_path, dec_inspection, mode="3d", resolution_level=0)

    dapi_raw = raw_arrays[channels["405"]].astype(np.float32)
    dapi_dec = dec_arrays[channels["405"]].astype(np.float32)

    # Shared mask from the RAW DAPI: isolates the detection effect.
    mask = nuclear_mask(dapi_raw)
    if mask.sum() < 5000:
        return result

    # Segmentation sensitivity, measured separately (test 5).
    mask_dec = nuclear_mask(dapi_dec)
    n_raw_objects = int(ndi.label(mask)[1])
    n_dec_objects = int(ndi.label(mask_dec)[1])

    for wavelength in ("561", "488"):
        index = channels[wavelength]
        raw = raw_arrays[index].astype(np.float32)
        dec = dec_arrays[index].astype(np.float32)

        dog_raw = difference_of_gaussians(raw)
        dog_dec = difference_of_gaussians(dec)
        peaks_raw, med_raw, sig_raw = detect(dog_raw, mask)
        peaks_dec, med_dec, sig_dec = detect(dog_dec, mask)

        phys_raw = to_physical(peaks_raw, spacing)
        phys_dec = to_physical(peaks_dec, spacing)

        # Match radius must be at least ~1 voxel on the coarsest axis, or a
        # single-plane shift in Z counts as a different focus.
        match_radius = max(MATCH_RADIUS_UM, MATCH_RADIUS_VOXEL_FACTOR * max(spacing))

        if peaks_raw.size and peaks_dec.size:
            tree_raw = cKDTree(phys_raw)
            matched_dec = np.array(
                [len(n) > 0 for n in tree_raw.query_ball_point(phys_dec, r=match_radius)]
            )
            tree_dec = cKDTree(phys_dec)
            matched_raw = np.array(
                [len(n) > 0 for n in tree_dec.query_ball_point(phys_raw, r=match_radius)]
            )
        else:
            matched_dec = np.zeros(len(peaks_dec), dtype=bool)
            matched_raw = np.zeros(len(peaks_raw), dtype=bool)

        dec_only = peaks_dec[~matched_dec] if peaks_dec.size else peaks_dec
        raw_only = peaks_raw[~matched_raw] if peaks_raw.size else peaks_raw

        # Adjudicate each disagreement against the other image, relative to
        # what that image's own background produces at random in-mask points.
        z_dec_only_in_raw = zscore_at(dog_raw, dec_only, med_raw, sig_raw)
        z_raw_only_in_dec = zscore_at(dog_dec, raw_only, med_dec, sig_dec)
        null_raw = null_zscores(dog_raw, mask, med_raw, sig_raw, rng)
        null_dec = null_zscores(dog_dec, mask, med_dec, sig_dec, rng)
        cut_raw = float(np.percentile(null_raw, NULL_PERCENTILE)) if null_raw.size else np.inf
        cut_dec = float(np.percentile(null_dec, NULL_PERCENTILE)) if null_dec.size else np.inf

        def classify(scores: np.ndarray, cut: float) -> tuple[int, int]:
            """Above the background's 95th percentile = real but sub-threshold."""
            recovered = int(np.sum(scores >= cut))
            return recovered, int(scores.size - recovered)

        dec_recovered, dec_artefact = classify(z_dec_only_in_raw, cut_raw)
        raw_recovered, raw_artefact = classify(z_raw_only_in_dec, cut_dec)

        # Ringing test. The z-score adjudication above can only detect foci
        # invented in flat background; it cannot catch deconvolution ringing,
        # which appears as a halo at a characteristic distance from a genuinely
        # bright object and therefore sits on elevated raw signal too. If the
        # added detections are ringing, they will cluster closer to existing
        # matched foci than chance allows.
        matched_phys = phys_raw[matched_raw] if peaks_raw.size else phys_raw
        dec_only_phys = to_physical(dec_only, spacing)
        ring_observed = ring_expected = float("nan")
        if matched_phys.shape[0] > 1 and dec_only_phys.shape[0] > 0:
            tree_matched = cKDTree(matched_phys)
            ring_observed = float(np.median(tree_matched.query(dec_only_phys)[0]))
            coords = np.argwhere(mask)
            take = min(dec_only_phys.shape[0] * 4, coords.shape[0])
            sample = coords[rng.choice(coords.shape[0], size=take, replace=False)]
            ring_expected = float(np.median(tree_matched.query(to_physical(sample, spacing))[0]))

        result.rows.append(
            {
                "label": label,
                "channel": wavelength,
                "mode": "3d" if is_3d else "2d",
                "true_z": true_z,
                "z_step_um": round(spacing[0], 4) if is_3d else "",
                "n_raw": int(len(peaks_raw)),
                "n_deconv": int(len(peaks_dec)),
                "n_both": int(matched_raw.sum()),
                "n_raw_only": int(len(raw_only)),
                "n_deconv_only": int(len(dec_only)),
                # Of the detections deconvolution ADDS, how many are real?
                "deconv_only_recovered": dec_recovered,
                "deconv_only_artefact": dec_artefact,
                "deconv_only_median_z_in_raw": round(float(np.median(z_dec_only_in_raw)), 3)
                if z_dec_only_in_raw.size
                else "",
                "raw_null_p95_z": round(cut_raw, 3),
                # Of the detections deconvolution LOSES, how many were real?
                "raw_only_recovered": raw_recovered,
                "raw_only_artefact": raw_artefact,
                "raw_only_median_z_in_deconv": round(float(np.median(z_raw_only_in_dec)), 3)
                if z_raw_only_in_dec.size
                else "",
                "deconv_null_p95_z": round(cut_dec, 3),
                "match_radius_um": round(match_radius, 3),
                # Ringing: added detections closer to existing foci than
                # chance would put them indicates deconvolution haloes.
                "deconv_only_dist_to_matched_um": round(ring_observed, 3)
                if np.isfinite(ring_observed)
                else "",
                "random_dist_to_matched_um": round(ring_expected, 3)
                if np.isfinite(ring_expected)
                else "",
                "raw_mad_sigma": round(sig_raw, 3),
                "deconv_mad_sigma": round(sig_dec, 3),
                "raw_p999": round(float(np.percentile(raw[mask], 99.9)), 1),
                "deconv_p999": round(float(np.percentile(dec[mask], 99.9)), 1),
                "n_nuclei_objects_raw": n_raw_objects,
                "n_nuclei_objects_deconv": n_dec_objects,
                "nuclear_voxels_raw": int(mask.sum()),
                "nuclear_voxels_deconv": int(mask_dec.sum()),
            }
        )

        for source, peaks, image in (("raw", peaks_raw, raw), ("deconv", peaks_dec, dec)):
            for row in measure_fwhm(image, peaks, spacing):
                row.update({"label": label, "channel": wavelength, "source": source,
                            "mode": "3d" if is_3d else "2d"})
                result.fwhm_rows.append(row)

    return result


def find_pairs(root: Path, limit_per_folder: int | None) -> list[tuple[Path, Path, str]]:
    """Match each deconvolved folder to its raw partner, tile by tile."""
    pairs: list[tuple[Path, Path, str]] = []
    for dec_folder in sorted(root.glob("*_deconvolved_Field_1")):
        raw_folder = dec_folder.with_name(
            dec_folder.name.replace("_deconvolved_Field_1", "_Field_1")
        )
        if not raw_folder.is_dir():
            continue
        dec_tiles = sorted(dec_folder.glob("*.ims"))
        if limit_per_folder:
            dec_tiles = dec_tiles[:limit_per_folder]
        for dec_tile in dec_tiles:
            raw_tile = raw_folder / dec_tile.name.replace("_deconvolved_", "_")
            if raw_tile.exists():
                pairs.append((raw_tile, dec_tile, f"{raw_folder.name}/{raw_tile.stem.split('_')[-1]}"))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-folder", type=int, default=None)
    args = parser.parse_args()

    pairs = find_pairs(args.root, args.limit_per_folder)
    print(f"Matched {len(pairs)} raw/deconvolved tile pairs under {args.root}", flush=True)
    if not pairs:
        return 1

    all_rows: list[dict] = []
    all_fwhm: list[dict] = []
    for raw_path, dec_path, label in pairs:
        try:
            result = compare_pair(raw_path, dec_path, label)
        except Exception as exc:  # noqa: BLE001 - one bad tile must not stop the pass
            print(f"  ERR {label}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not result.rows:
            print(f"  skip {label} (no usable nuclear mask)", flush=True)
            continue
        all_rows.extend(result.rows)
        all_fwhm.extend(result.fwhm_rows)
        summary = ", ".join(
            f"{r['channel']}: {r['n_raw']}->{r['n_deconv']} "
            f"(+{r['n_deconv_only']} / -{r['n_raw_only']})"
            for r in result.rows
        )
        print(f"  {label}  {summary}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections = args.output_dir / "phase0_detection_adjudication.csv"
    with detections.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {detections} ({len(all_rows)} rows)")

    if all_fwhm:
        fwhm_path = args.output_dir / "phase0_fwhm.csv"
        with fwhm_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_fwhm[0].keys()))
            writer.writeheader()
            writer.writerows(all_fwhm)
        print(f"wrote {fwhm_path} ({len(all_fwhm)} puncta)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
