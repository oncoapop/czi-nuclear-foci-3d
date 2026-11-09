"""Per-nucleus axial thickness from DAPI, and what it does to the slab confound.

WHY THIS EXISTS
  The counting pass measures foci inside a FIXED 1.5 um slab, because the
  tilted slide gives every tile a different in-focus range. That makes counts
  comparable between tiles, but it samples a different FRACTION of a 83 um2
  WT nucleus than of a 189 um2 p53-/-BRCA1-/- nucleus - and the fraction
  sampled drives how much sampling variance lands in the per-nucleus count.
  Since the headline result is that BRCA1-/- counts are MORE dispersed, the
  slab is the main threat to it.

  Thickness decides the direction. If BRCA1-/- nuclei are thicker, the slab
  catches proportionally less of them, sampling variance is higher, and some
  of the observed dispersion gap is artifact. If they are FLATTER - which is
  common for large spread cells - the slab catches proportionally more, their
  sampling variance is LOWER, and the gap is understated rather than inflated.

METHOD
  Per nucleus, the mean DAPI over its 2D footprint is profiled across the
  whole acquired stack (not just the usable slab - out-of-focus DAPI is still
  DAPI), background-subtracted per slice, and the FWHM taken with linear
  interpolation between slices. The measured width is the true extent
  convolved with the axial PSF, so the PSF is removed in quadrature.

  Nuclei whose profile does not fall to half-maximum on BOTH sides inside the
  stack are CENSORED and excluded - keeping them would silently truncate the
  thick tail and bias exactly the comparison being made.

  Read at ResolutionLevel 1: the pyramid halves Y/X but never Z, so axial
  resolution is untouched and the read is 4x cheaper. Nucleus segmentation is
  recomputed at that level using Stage 0's physical-units rule.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_ims_qc import CHANNEL_LABELS, nucleus_labels  # noqa: E402
from stage2_registration import _channel_index_map  # noqa: E402
from stage45_prelim_foci_coloc import DESIGN, SATURATED_DAPI_ROUNDS, SLAB_THICKNESS_UM  # noqa: E402

from czi_foci.ims_reader import inspect_ims, pixel_size_um_ims, read_channel_arrays_ims  # noqa: E402

THICKNESS_RESOLUTION_LEVEL = 1
PSF_AXIAL_FWHM_UM = 0.9  # 60x/1.2NA; removed in quadrature from the measured width
MIN_SLICES_FOR_FWHM = 5


def axial_fwhm(profile: np.ndarray, z_step_um: float) -> tuple[float, bool]:
    """FWHM of an axial profile, interpolated between slices.

    Returns (fwhm_um, censored). Censored means the profile never drops to
    half-maximum on one or both sides inside the acquired stack, so the true
    width is only bounded below - those nuclei must be dropped, not clipped.
    """
    if len(profile) < MIN_SLICES_FOR_FWHM:
        return float("nan"), True
    # The profile arrives already background-subtracted per slice, so its
    # baseline is ~0. Do NOT re-subtract the in-stack minimum: when a nucleus
    # fills the stack the minimum sits on the nucleus itself, which
    # manufactures a peak out of the residual and returns a small, wrong
    # width. That would bias thick nuclei downward - the exact direction that
    # would corrupt this comparison.
    p = np.asarray(profile, dtype=float)
    peak = float(p.max())
    if peak <= 0:
        return float("nan"), True
    half = peak / 2.0
    # If the profile has not fallen to half-maximum by either end of the
    # stack, the nucleus extends past what was acquired and its width is only
    # bounded below.
    if p[0] > half or p[-1] > half:
        return float("nan"), True
    apex = int(np.argmax(p))

    left = None
    for i in range(apex, 0, -1):
        if p[i - 1] <= half <= p[i]:
            span = p[i] - p[i - 1]
            left = (i - 1) + ((half - p[i - 1]) / span if span > 0 else 0.5)
            break
    right = None
    for i in range(apex, len(p) - 1):
        if p[i + 1] <= half <= p[i]:
            span = p[i] - p[i + 1]
            right = i + ((p[i] - half) / span if span > 0 else 0.5)
            break
    if left is None or right is None:
        return float("nan"), True
    return float((right - left) * z_step_um), False


_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def axial_fwhm_by_fit(profile: np.ndarray, z_step_um: float, edge_margin: int = 2) -> tuple[float, bool]:
    """Axial FWHM from a Gaussian fit, which tolerates truncated flanks.

    Half-max crossing fails on this data: the one nucleus it could measure is
    7.6 um thick while the stacks are only 6.5-9 um tall and the focal plane
    is not centred in them (Stage 0: stacks started too high), so nearly every
    nucleus runs past one end and gets censored. A parametric fit recovers the
    width from curvature instead of from the crossing points, so it still
    works when one flank is clipped.

    A Gaussian is an approximation - a flattened ellipsoid's axial DAPI
    profile is nearer a parabola convolved with the PSF - so the absolute
    value is a proxy. It is used only to compare cell lines measured the same
    way, which is what the slab question needs.

    Returns (fwhm_um, failed). Fails when the profile's apex sits within
    ``edge_margin`` slices of either end, since then neither flank constrains
    the width, or when the fit does not converge to a physically plausible
    width.
    """
    from scipy.optimize import curve_fit

    if len(profile) < MIN_SLICES_FOR_FWHM:
        return float("nan"), True
    p = np.asarray(profile, dtype=float)
    peak = float(p.max())
    if peak <= 0:
        return float("nan"), True
    apex = int(np.argmax(p))
    if apex < edge_margin or apex > len(p) - 1 - edge_margin:
        return float("nan"), True

    z = np.arange(len(p), dtype=float)

    def model(zz, amplitude, centre, sigma, floor):
        return floor + amplitude * np.exp(-((zz - centre) ** 2) / (2.0 * sigma**2))

    try:
        popt, _ = curve_fit(
            model, z, p,
            p0=[peak, float(apex), max(len(p) / 4.0, 1.0), 0.0],
            bounds=([0.0, -len(p), 0.3, -peak], [peak * 10, 2 * len(p), len(p) * 4.0, peak]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return float("nan"), True
    sigma = float(popt[2])
    fwhm = sigma * _FWHM_PER_SIGMA * z_step_um
    # A width many times the acquired stack is an extrapolation, not a
    # measurement.
    if not np.isfinite(fwhm) or fwhm <= 0 or fwhm > len(p) * z_step_um * 3.0:
        return float("nan"), True
    return fwhm, False


def deconvolve_psf(fwhm_um: float, psf_fwhm_um: float = PSF_AXIAL_FWHM_UM) -> float:
    """Remove the axial PSF from a measured width, in quadrature.

    A nucleus thinner than the PSF cannot be resolved axially at all; those
    return 0.0 rather than an imaginary width, and should be read as
    "at or below the axial resolution limit".
    """
    if not np.isfinite(fwhm_um):
        return float("nan")
    return float(np.sqrt(max(fwhm_um**2 - psf_fwhm_um**2, 0.0)))


def sampling_cv_component(n_total: float, fraction_sampled: float) -> float:
    """CV contributed by binomial slab sampling alone.

    Counting n of N objects, each independently inside the slab with
    probability p, gives variance N*p*(1-p) on a mean of N*p, so
    CV_sampling = sqrt((1-p)/(N*p)). This is the part of the observed
    dispersion the slab is responsible for; anything above it is biology or
    detection.
    """
    if n_total <= 0 or not (0 < fraction_sampled <= 1):
        return float("nan")
    return float(np.sqrt((1.0 - fraction_sampled) / (n_total * fraction_sampled)))


def _process_tile(args) -> pd.DataFrame:
    row = args
    try:
        path = Path(row["source_path"])
        inspection = inspect_ims(path)
        arrays = read_channel_arrays_ims(
            path, inspection, mode="3d", resolution_level=THICKNESS_RESOLUTION_LEVEL
        )
        index_by_name = _channel_index_map(row["channel_map"])
        dapi = arrays[index_by_name["DAPI_405"]].astype(np.float32)
        arrays.clear()

        metadata = inspection.get("metadata") or {}
        z_step_um, _, pixel_size_um = pixel_size_um_ims(
            metadata, mode="3d", resolution_level=THICKNESS_RESOLUTION_LEVEL
        )
        if not np.isfinite(z_step_um):
            raise ValueError("no z-step")

        focus_idx = min(int(row["focus_peak_idx"]), dapi.shape[0] - 1)
        labels, mask = nucleus_labels(dapi, focus_idx, pixel_size_um)
        labels = segmentation.clear_border(labels)
        labels = measure.label(labels > 0).astype(np.int32)
        if labels.max() == 0:
            raise ValueError("no interior nuclei")

        # Per-slice background from non-nuclear pixels, so the profile floor
        # is the tile's own haze rather than zero.
        background = np.array([np.median(plane[~mask]) if (~mask).any() else 0.0 for plane in dapi])

        design = DESIGN.get(float(row.get("well_position_id", float("nan"))), {})
        out = []
        for region in measure.regionprops(labels):
            ys, xs = region.coords[:, 0], region.coords[:, 1]
            profile = dapi[:, ys, xs].mean(axis=1) - background
            fwhm_cross, censored = axial_fwhm(profile, z_step_um)
            fwhm, fit_failed = axial_fwhm_by_fit(profile, z_step_um)
            out.append(
                {
                    "fwhm_crossing_um": fwhm_cross,
                    "crossing_censored": censored,
                    "fit_failed": fit_failed,
                    "round": row["round"],
                    "tile": row["tile"],
                    "well_position_id": row.get("well_position_id", np.nan),
                    "well": design.get("well"),
                    "cell_line": design.get("cell_line"),
                    "treatment": design.get("treatment"),
                    "has_primary": design.get("has_primary"),
                    "nucleus_id": int(region.label),
                    "nucleus_area_um2": region.area * pixel_size_um**2,
                    "z_step_um": z_step_um,
                    "n_z": int(dapi.shape[0]),
                    "fwhm_um": fwhm,
                    "thickness_um": deconvolve_psf(fwhm),
                    "censored": censored,
                    "dapi_saturated_round": row["round"] in SATURATED_DAPI_ROUNDS,
                }
            )
        return pd.DataFrame(out)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"round": row["round"], "tile": row["tile"], "error": f"{type(exc).__name__}: {exc}"}])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("/Volumes/Backup/Andor imaging/TRF2/_analysis"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rounds", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tiles = pd.read_csv(args.analysis_dir / "stage0_tile_qc.csv")
    tiles = tiles[(tiles["read_ok"] == True) & tiles["usable_z_range"].notna()]  # noqa: E712
    tiles = tiles[~tiles["round"].str.contains("deconvolved")]
    if args.rounds:
        tiles = tiles[tiles["round"].isin({r.strip() for r in args.rounds.split(",")})]
    if args.limit:
        tiles = tiles.head(args.limit)
    print(f"Nuclear thickness: {len(tiles)} tiles at ResolutionLevel {THICKNESS_RESOLUTION_LEVEL} "
          f"({args.workers} workers)")

    frames = []
    with Pool(processes=args.workers) as pool:
        for frame in pool.imap_unordered(_process_tile, [r for _, r in tiles.iterrows()], chunksize=1):
            frames.append(frame)
    nuclei = pd.concat(frames, ignore_index=True)
    if "error" in nuclei:
        for _, e in nuclei[nuclei["error"].notna()].iterrows():
            print(f"  WARNING {e['round']}/{e['tile']}: {e['error']}")
        nuclei = nuclei[nuclei["error"].isna()].drop(columns=["error"])
    # Concatenating the error frames left the boolean columns as dtype
    # object, and `~` on an object column returns -1/-2 - both truthy - so
    # every filter silently passed everything. Restore real booleans before
    # any filtering.
    for column in ("fit_failed", "crossing_censored", "dapi_saturated_round", "has_primary"):
        if column in nuclei:
            nuclei[column] = nuclei[column].astype(bool)
    # Keep failed measurements in the file. Dropping them here is how the
    # first run reported "0 censored" while censoring 99.9% of nuclei.
    out = args.analysis_dir / "nuclear_thickness_per_nucleus.csv"
    nuclei.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(nuclei)} nuclei)")
    _report(nuclei)
    return 0


def _report(nuclei: pd.DataFrame) -> None:
    total = len(nuclei)
    usable = nuclei[(~nuclei["fit_failed"]) & (~nuclei["dapi_saturated_round"])
                    & (nuclei["has_primary"] == True)]  # noqa: E712
    print(f"\n{len(usable)} of {total} nuclei measurable by Gaussian fit")
    print(f"  fit failed (apex too near a stack edge, or no convergence): "
          f"{int(nuclei['fit_failed'].sum())} ({nuclei['fit_failed'].mean():.1%})")
    print(f"  half-max crossing censored (nucleus runs past the stack): "
          f"{int(nuclei['crossing_censored'].sum())} ({nuclei['crossing_censored'].mean():.1%})"
          "  <- why the crossing method was abandoned")
    print(f"  in DAPI-saturated rounds (_6, _11), excluded: "
          f"{int(nuclei['dapi_saturated_round'].sum())}")
    both = nuclei[(~nuclei["fit_failed"]) & (~nuclei["crossing_censored"])]
    if len(both) >= 20:
        ratio = (both["fwhm_um"] / both["fwhm_crossing_um"]).median()
        print(f"  cross-check on the {len(both)} nuclei both methods measured: "
              f"fit/crossing median ratio {ratio:.2f}")
    if len(usable) < 50:
        print("\n  TOO FEW NUCLEI MEASURABLE - the acquired z-range cannot support this")
        print("  measurement. Report as inconclusive rather than quoting these numbers.")

    print("\n=== NUCLEAR THICKNESS by cell line (PSF removed in quadrature) ===")
    print(f"{'cell line':>16} {'treatment':>12} {'n':>6} {'median um':>10} {'mean um':>9} "
          f"{'area um2':>9} {'slab frac':>10}")
    by_line = {}
    for (line, treat), g in usable.groupby(["cell_line", "treatment"]):
        med = g["thickness_um"].median()
        frac = min(SLAB_THICKNESS_UM / med, 1.0) if med > 0 else np.nan
        print(f"{str(line):>16} {str(treat):>12} {len(g):>6} {med:>10.2f} "
              f"{g['thickness_um'].mean():>9.2f} {g['nucleus_area_um2'].mean():>9.1f} {frac:>10.2f}")
    for line, g in usable.groupby("cell_line"):
        med = g["thickness_um"].median()
        by_line[line] = (med, min(SLAB_THICKNESS_UM / med, 1.0) if med > 0 else np.nan,
                         g["nucleus_area_um2"].mean())

    print("\n=== WHAT IT DOES TO THE SLAB CONFOUND ===")
    lines = list(by_line)
    if len(lines) == 2:
        a, b = lines
        t_a, f_a, _ = by_line[a]
        t_b, f_b, _ = by_line[b]
        print(f"  {a}: {t_a:.2f} um thick, slab samples {f_a:.0%}")
        print(f"  {b}: {t_b:.2f} um thick, slab samples {f_b:.0%}")
        thicker = a if t_a > t_b else b
        print(f"\n  {thicker} nuclei are thicker, so the slab samples LESS of them.")
        if thicker.startswith("p53"):
            print("  That inflates their sampling variance -> part of the observed dispersion")
            print("  gap IS slab artifact. Quantified below.")
        else:
            print("  The MORE dispersed line (BRCA1-/-) is the BETTER sampled one, so slab")
            print("  sampling works AGAINST the observed gap - it understates it, not inflates it.")

    print("\n  Sampling CV the slab alone would produce, for a nucleus of N telomeres:")
    print(f"  {'N':>6} " + " ".join(f"{str(l)[:14]:>16}" for l in lines))
    for n_total in (40, 80, 160):
        cells = []
        for line in lines:
            _, frac, _ = by_line[line]
            cells.append(f"{sampling_cv_component(n_total, frac):>16.2f}")
        print(f"  {n_total:>6} " + " ".join(cells))
    print("\n  Compare with the observed CVs at k=6 (WT 0.93-0.98, BRCA1-/- 1.17-1.33).")


if __name__ == "__main__":
    raise SystemExit(main())
