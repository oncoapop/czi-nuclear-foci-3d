"""Integrated DAPI per nucleus as a DNA-content proxy, to separate two explanations.

THE QUESTION
  p53-/-BRCA1-/- nuclei show 2.9-3.9x higher dispersion in telomere-focus
  count than WT. Two explanations fit equally well so far:
    (a) telomere-specific instability - fusion, loss, fragility
    (b) variable DNA content - aneuploidy on a p53-null background
  Both produce cell-to-cell variation in how many telomeres a nucleus holds.
  Normalising each nucleus by its own DNA content separates them: under (b)
  the dispersion gap collapses, under (a) it survives.

  Nuclear footprint area was already tried as a proxy and did not collapse
  the gap - but area is not ploidy. Integrated DAPI is the standard DNA
  measure (the same quantity flow cytometry uses for cell-cycle gating).

WHY THE NUMBERS ARE COMPARABLE BETWEEN WELLS
  PreAmpGain, excitation/emission wavelengths and objective are identical in
  every round, and all wells are one slide stained in one batch and imaged in
  one session. Exposure time is not recorded in the .ims attributes, so that
  part is an assumption; the per-round background DAPI level is reported as
  an empirical check on it.

SEGMENTATION MUST MATCH THE COUNTING PASS
  Nuclei are segmented here exactly as in stage45_prelim_foci_coloc - same
  rule, same resolution level, same border clearing - so ``nucleus_id``
  refers to the same object and the two tables can be joined directly rather
  than matched by position. DAPI is segmented RAW and integrated
  FLAT-FIELDED, which is the order the counting pass uses.

SATURATION IS DISQUALIFYING
  A clipped pixel caps the integral, so a saturated nucleus reports less DNA
  than it has - and would mimic exactly the narrowing this test looks for.
  Rounds _6 and _11 are saturated wholesale (Stage 0) and any individual
  nucleus with clipped voxels is dropped.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from skimage import measure, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_ims_qc import nucleus_labels  # noqa: E402
from stage2_registration import ANALYSIS_RESOLUTION_LEVEL, _channel_index_map, load_flatfield  # noqa: E402
from stage45_prelim_foci_coloc import (  # noqa: E402
    DESIGN,
    SATURATED_DAPI_ROUNDS,
    SLAB_THICKNESS_UM,
    place_slab,
)

SATURATION_LEVEL = 60000  # 16-bit detector; clipping starts below 65535
MAX_SATURATED_FRACTION = 0.001  # per nucleus


def read_dapi_stack(path: Path, channel_index: int, true_z: int, level: int) -> np.ndarray:
    """Read one channel's stack directly, without pulling the other two.

    ``read_channel_arrays_ims`` loads every channel present, which is three
    times the I/O this needs.
    """
    with h5py.File(path, "r") as handle:
        data = handle["DataSet"][f"ResolutionLevel {level}"]["TimePoint 0"][
            f"Channel {channel_index}"
        ]["Data"]
        return data[:true_z].astype(np.float32)


def integrated_dna(
    stack: np.ndarray,
    coords: np.ndarray,
    background: np.ndarray,
    slab: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Background-subtracted DAPI summed over the nucleus, and its saturated fraction.

    ``slab`` restricts the sum to the same z range the telomere counting used.
    That is the PRIMARY measure, and it is the one that is truncation-free:
    nuclei are 6-8 um thick in 7-9 um stacks with an off-centre focal plane,
    so a whole-stack integral loses a position-dependent slice of every
    nucleus and inflates its apparent variability - measured CV 1.25 in
    G1-arrested WT cells, which should be close to uniform 2N.

    Summing DNA over the SAME slab the foci were counted in makes the
    geometry cancel: both numerator and denominator scale with the slab
    volume actually sampled, so counts-per-DNA is a density ratio that does
    not care how much of the nucleus was captured.
    """
    ys, xs = coords[:, 0], coords[:, 1]
    z0, z1 = (0, stack.shape[0] - 1) if slab is None else slab
    column = stack[z0 : z1 + 1, ys, xs]
    saturated = float((column >= SATURATION_LEVEL).mean())
    return float((column - background[z0 : z1 + 1, None]).sum()), saturated


def _process_tile(row) -> pd.DataFrame:
    try:
        path = Path(row["source_path"])
        channel_index = _channel_index_map(row["channel_map"])["DAPI_405"]
        stack = read_dapi_stack(path, channel_index, int(row["true_z"]), ANALYSIS_RESOLUTION_LEVEL)
        pixel_size_um = float(row["pixel_size_um_level0"])
        focus_idx = min(int(row["focus_peak_idx"]), stack.shape[0] - 1)

        # Segment on RAW DAPI, matching the counting pass exactly so that
        # nucleus_id means the same thing in both tables.
        labels, mask = nucleus_labels(stack, focus_idx, pixel_size_um)
        labels = segmentation.clear_border(labels)
        labels = measure.label(labels > 0).astype(np.int32)
        if labels.max() == 0:
            raise ValueError("no interior nuclei")

        field = load_flatfield(Path(row["_analysis_dir"]), row.get("flatfield_path_DAPI_405"))
        corrected = stack / field[None, :, :] if (field is not None and field.shape == stack.shape[1:]) else stack
        background = np.array(
            [np.median(plane[~mask]) if (~mask).any() else 0.0 for plane in corrected]
        )

        # Same slab the counting pass used, so the geometry cancels in
        # counts-per-DNA.
        thickness_slices = max(1, int(round(SLAB_THICKNESS_UM / float(row["z_step_um"]))))
        usable_lo, usable_hi = (int(v) for v in str(row["usable_z_range"]).split("-"))
        slab = place_slab(focus_idx, usable_lo, usable_hi, stack.shape[0], thickness_slices)
        if slab is None:
            raise ValueError("usable slab too thin")

        design = DESIGN.get(float(row.get("well_position_id", float("nan"))), {})
        out = []
        for region in measure.regionprops(labels):
            dna, saturated = integrated_dna(corrected, region.coords, background, slab)
            dna_full, saturated_full = integrated_dna(corrected, region.coords, background)
            out.append(
                {
                    "round": row["round"],
                    "tile": row["tile"],
                    "nucleus_id": int(region.label),
                    "well": design.get("well"),
                    "cell_line": design.get("cell_line"),
                    "treatment": design.get("treatment"),
                    "has_primary": design.get("has_primary"),
                    "integrated_dna": dna,
                    "integrated_dna_fullstack": dna_full,
                    "saturated_fraction": max(saturated, saturated_full),
                    "nucleus_area_um2": region.area * pixel_size_um**2,
                    "background_dapi": float(np.median(background)),
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
    tiles = tiles.assign(_analysis_dir=str(args.analysis_dir))
    print(f"DAPI ploidy: {len(tiles)} tiles ({args.workers} workers)")

    frames = []
    with Pool(processes=args.workers) as pool:
        for frame in pool.imap_unordered(_process_tile, [r for _, r in tiles.iterrows()], chunksize=1):
            frames.append(frame)
    dna = pd.concat(frames, ignore_index=True)
    if "error" in dna:
        for _, e in dna[dna["error"].notna()].iterrows():
            print(f"  WARNING {e['round']}/{e['tile']}: {e['error']}")
        dna = dna[dna["error"].isna()].drop(columns=["error"])
    for column in ("has_primary", "dapi_saturated_round"):
        dna[column] = dna[column].astype(bool)

    out = args.analysis_dir / "dapi_ploidy_per_nucleus.csv"
    dna.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(dna)} nuclei)")
    _report(dna, args.analysis_dir)
    return 0


def _report(dna: pd.DataFrame, analysis_dir: Path) -> None:
    print("\nExposure sanity check - median background DAPI per round "
          "(should be similar if acquisition settings matched):")
    bg = dna.groupby("round")["background_dapi"].median().sort_values()
    print(f"  range {bg.min():.0f} - {bg.max():.0f}  ({bg.max() / max(bg.min(), 1e-9):.1f}x across rounds)")

    clean = dna[(~dna["dapi_saturated_round"])
                & (dna["saturated_fraction"] <= MAX_SATURATED_FRACTION)
                & (dna["has_primary"])]
    print(f"\n{len(clean)} of {len(dna)} nuclei usable "
          f"({int(dna['dapi_saturated_round'].sum())} in saturated rounds, "
          f"{int((dna['saturated_fraction'] > MAX_SATURATED_FRACTION).sum())} individually saturated)")

    print("\n=== DNA CONTENT HETEROGENEITY (the direct ploidy readout) ===")
    print("  slab-integrated is primary (truncation-free); full-stack shown for contrast.")
    print(f"{'group':>34} {'n':>6} {'median DNA':>12} {'CV slab':>9} {'CV fullstack':>13}")
    for (line, treat), g in clean.groupby(["cell_line", "treatment"]):
        d, f = g["integrated_dna"], g["integrated_dna_fullstack"]
        print(f'{line + " / " + treat:>34} {len(g):>6} {d.median():>12.3g} '
              f'{d.std() / d.mean():>9.2f} {f.std() / f.mean():>13.2f}')

    counts = pd.read_csv(analysis_dir / "stage45_prelim_per_nucleus.csv")
    counts = counts[counts["detect_k"] == 6.0]
    merged = counts.merge(
        clean[["round", "tile", "nucleus_id", "integrated_dna", "saturated_fraction"]],
        on=["round", "tile", "nucleus_id"], how="inner",
    )
    print(f"\nJoined {len(merged)} nuclei to the k=6 counts "
          f"({len(merged) / max(len(counts[counts['has_primary']]), 1):.0%} of stained nuclei)")
    if len(merged) < 200:
        print("  TOO FEW JOINED - segmentation did not match; report as inconclusive.")
        return

    merged["foci_per_dna"] = merged["n_foci_RFP_561"] / merged["integrated_dna"]
    print("\n=== DOES NORMALISING BY DNA COLLAPSE THE DISPERSION GAP? ===")
    print(f"{'group':>34} {'n':>6} {'CV(count)':>10} {'CV(count/DNA)':>14}")
    cvs = {}
    for (line, treat), g in merged.groupby(["cell_line", "treatment"]):
        cv_n = g["n_foci_RFP_561"].std() / g["n_foci_RFP_561"].mean()
        cv_d = g["foci_per_dna"].std() / g["foci_per_dna"].mean()
        cvs[(line, treat)] = (cv_n, cv_d)
        print(f'{line + " / " + treat:>34} {len(g):>6} {cv_n:>10.2f} {cv_d:>14.2f}')

    print()
    for treat in ("Palbociclib", "untreated"):
        wt = cvs.get(("WT_hTERT", treat))
        ko = cvs.get(("p53-/-BRCA1-/-", treat))
        if not (wt and ko):
            continue
        raw, norm = ko[0] / wt[0], ko[1] / wt[1]
        verdict = ("gap SURVIVES -> telomere-specific" if norm > 0.8 * raw
                   else "gap COLLAPSES -> explained by DNA content")
        print(f"  {treat:>11}: CV ratio BRCA1/WT  raw {raw:.2f}  ->  per-DNA {norm:.2f}   {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
