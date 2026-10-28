"""Stage 1b: retrospective flat-field correction (BLOCKING for later stages).

Stage 0b found background intensity falling 10-47% centre-to-edge in nearly
every TRF2 folder and channel - a position-dependent detection bias (foci
near tile edges are less likely to be detected) that, under blinding, can
imitate a treatment effect. No flat-field acquisition exists and none is
needed: with 189 successfully-read tiles and randomly placed cells, a
per-pixel robust average across tiles directly estimates the illumination
profile (BaSiC approach, Peng et al. 2017, implemented directly here with no
new dependency).

For each channel, per folder:
  1. Read every tile's DAPI-focus-plane slice at ResolutionLevel 0 (full
     resolution - the illumination field must be estimated at the resolution
     it will be applied at).
  2. Estimate a darkfield offset from the per-pixel minimum across tiles,
     smoothed.
  3. Per-pixel median of the darkfield-subtracted planes, then a large-sigma
     Gaussian smooth -> the illumination field, normalised to median 1.0.

VALIDATION GATE: the field must look like a smooth bowl. A quantitative proxy
(the high-frequency residual left by the final smoothing, relative to the
field's own dynamic range) stands in for "visible cell structure" here, since
no human reviews this script's intermediate output before the PDF does.
Folders with fewer than MIN_TILES_FOR_OWN_FIELD tiles, or whose own field
fails the gate, fall back to the field pooled across every TRF2 folder for
that channel.

Verification divides each tile's own raw focus plane by its selected field
and recomputes the radial background profile and a DAPI nucleus-density
profile before/after, to confirm the gradient is gone. This stage does not
write corrected pixel volumes to disk (that would roughly double TRF2's
on-disk footprint for no immediate consumer); instead it records, per tile,
which field (own-folder or pooled) downstream stages must divide by, so the
correction happens once, when the full z-stack is actually read for
segmentation/detection.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field as dataclass_field
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_ims_qc import (  # noqa: E402
    CHANNEL_LABELS,
    MIN_NUCLEUS_AREA_UM2,
    N_RADIAL_BINS,
    RADIAL_BIN_CENTERS_UM,
    RADIAL_BIN_EDGES_UM,
    background_radial_profile,
)

from czi_foci.ims_reader import read_channel_planes_ims  # noqa: E402

MIN_TILES_FOR_OWN_FIELD = 6
# Fraction of the field's dynamic range explained by the high-frequency
# residual left by the final smoothing step. Above this, the "bowl" likely
# still carries real cell structure (too few tiles, or non-random placement).
# Empirical heuristic, not a tuned detection threshold.
SMOOTH_BOWL_MAX_SCORE = 0.25
MIN_TILES_FOR_RADIAL_METRIC = 3


def darkfield_offset(stack: np.ndarray, max_fraction_of_median: float = 0.05) -> float:
    """A conservative, scalar darkfield estimate.

    A full per-pixel minimum map (the textbook BaSiC darkfield) is only safe
    when the true additive offset is well separated from the illuminated
    signal. In practice (confirmed with a synthetic vignette test), the
    per-pixel minimum across a modest number of tiles tracks the
    *illumination-modulated* background almost everywhere - real fluorescence
    background is still bowl-shaped, the camera's own dark offset is not -
    so subtracting it, even as a single scalar taken from a low percentile of
    that map, partially removes real bowl structure along with any genuine
    camera offset and can distort the illumination estimate far more than it
    corrects it.

    This estimates the offset the same way (a low percentile of the per-pixel
    minimum across tiles), but caps it at a small fraction of the overall
    median tile intensity. A true camera/background offset is a small
    fraction of the illuminated signal, so the cap costs nothing when the
    estimate is genuine and bounds the damage when it is not (e.g. no real
    darkfield exists at all, or too few tiles make the percentile noisy).
    """
    raw_min = stack.min(axis=0)
    estimate = float(np.percentile(raw_min, 1.0))
    cap = float(np.median(stack)) * max_fraction_of_median
    return float(np.clip(estimate, 0.0, cap))


def smooth_bowl_score(raw_field: np.ndarray, smoothed_field: np.ndarray) -> float:
    """High-frequency residual left by smoothing, relative to the field's own range.

    Large values mean the per-pixel median still has structure a smooth
    illumination bowl would not - i.e. the gate failed. The denominator is
    floored at 1% of the field's mean level so a genuinely flat field (no
    real bowl at all) cannot produce a spuriously huge score just because it
    has almost no dynamic range to divide by.
    """
    residual = raw_field - smoothed_field
    field_range = float(smoothed_field.max() - smoothed_field.min())
    mean_level = float(np.mean(smoothed_field))
    denom = max(field_range, abs(mean_level) * 0.01, 1e-9)
    return float(residual.std() / denom)


@dataclass
class IlluminationField:
    field: np.ndarray
    score: float
    n_tiles: int
    passed_gate: bool


def estimate_illumination_field(planes: list[np.ndarray]) -> IlluminationField:
    stack = np.stack(planes).astype(np.float32)
    darkfield = darkfield_offset(stack)
    corrected = np.clip(stack - darkfield, 0, None)
    median_field = np.median(corrected, axis=0)
    sigma = median_field.shape[0] / 8.0
    smoothed = ndi.gaussian_filter(median_field, sigma)
    score = smooth_bowl_score(median_field, smoothed)
    normalized = smoothed / np.median(smoothed)
    return IlluminationField(field=normalized, score=score, n_tiles=len(planes), passed_gate=score <= SMOOTH_BOWL_MAX_SCORE)


def segment_foreground_2d(plane: np.ndarray, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight 2D Otsu foreground/background split and nucleus centroids.

    This is a QC-level stand-in for Stage 0b's full 3D nucleus counting (it
    works on one already-in-memory plane, not a focus-centred sub-MIP), used
    only to define "non-nuclear background" for the radial background check
    and to get approximate nucleus centroids for the before/after
    density-vs-radius comparison. Not a re-derivation of n_nuclei.
    """
    smoothed = ndi.gaussian_filter(plane.astype(np.float32), 2.0)
    if np.unique(smoothed).size <= 1:
        return np.zeros(plane.shape, dtype=bool), np.zeros((0, 2))
    mask = smoothed > filters.threshold_otsu(smoothed)
    min_size_px = max(1, int(round(MIN_NUCLEUS_AREA_UM2 / (pixel_size_um**2))))
    mask = morphology.remove_small_objects(mask, min_size_px)
    mask = ndi.binary_fill_holes(mask)
    if not mask.any():
        return mask, np.zeros((0, 2))
    labels = measure.label(mask)
    centroids = np.array([r.centroid for r in measure.regionprops(labels)]) if labels.max() else np.zeros((0, 2))
    return mask, centroids


_RADIAL_BIN_AREAS_UM2 = np.pi * (RADIAL_BIN_EDGES_UM[1:] ** 2 - RADIAL_BIN_EDGES_UM[:-1] ** 2)


def nucleus_density_radial_profile(centroids_yx: np.ndarray, shape: tuple[int, int], pixel_size_um: float) -> np.ndarray:
    """Nuclei per um2, binned by radial distance from the tile centre."""
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    counts = np.zeros(N_RADIAL_BINS)
    if len(centroids_yx):
        radii = np.hypot(centroids_yx[:, 0] - cy, centroids_yx[:, 1] - cx) * pixel_size_um
        bin_idx = np.clip(np.digitize(radii, RADIAL_BIN_EDGES_UM) - 1, 0, N_RADIAL_BINS - 1)
        for b in bin_idx:
            counts[b] += 1
    return counts / _RADIAL_BIN_AREAS_UM2


def radial_slope_and_edge_pct(pooled_profile: np.ndarray) -> tuple[float, float]:
    valid = ~np.isnan(pooled_profile)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    slope, _ = np.polyfit(RADIAL_BIN_CENTERS_UM[valid], pooled_profile[valid], 1)
    center_val = pooled_profile[0] if not np.isnan(pooled_profile[0]) else pooled_profile[valid][0]
    edge_val = pooled_profile[-1] if not np.isnan(pooled_profile[-1]) else pooled_profile[valid][-1]
    edge_pct = float((edge_val - center_val) / center_val * 100.0) if center_val > 0 else float("nan")
    return float(slope), edge_pct


def _read_tile_planes(args: tuple[Path, str, str, dict, int]) -> dict:
    path, round_label, tile_name, channel_index, focus_peak_idx = args
    try:
        planes = read_channel_planes_ims(path, list(channel_index.values()), z_index=focus_peak_idx, resolution_level=0)
        by_name = {CHANNEL_LABELS[wavelength]: planes[index] for wavelength, index in channel_index.items() if index in planes}
        return {"round": round_label, "tile": tile_name, "path": str(path), "ok": True, "planes": by_name}
    except Exception as exc:  # noqa: BLE001
        return {"round": round_label, "tile": tile_name, "path": str(path), "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def load_tile_planes(tile_df: pd.DataFrame, workers: int) -> list[dict]:
    tasks = []
    for _, row in tile_df.iterrows():
        channel_index = {}
        for part in row["channel_map"].split(";"):
            wavelength, idx = part.split("=C")
            channel_index[wavelength] = int(idx)
        tasks.append((Path(row["source_path"]), row["round"], row["tile"], channel_index, int(row["focus_peak_idx"])))

    results = []
    with Pool(processes=workers) as pool:
        for result in pool.imap(_read_tile_planes, tasks, chunksize=1):
            results.append(result)
    return results


def display_u16_to_u8(plane: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(plane, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((plane.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TRF2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N read_ok tiles (debugging)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "_analysis")
    flatfield_dir = output_dir / "flatfield"
    flatfield_dir.mkdir(parents=True, exist_ok=True)

    tile_csv = output_dir / "stage0_tile_qc.csv"
    tile_df = pd.read_csv(tile_csv)
    ok = tile_df[tile_df["read_ok"] == True].copy()  # noqa: E712
    if args.limit:
        per_folder_cap = max(1, args.limit // ok["round"].nunique())
        ok = ok.groupby("round", as_index=False, group_keys=False).head(per_folder_cap).reset_index(drop=True)
    print(f"Reading RL0 focus planes for {len(ok)} tiles ({args.workers} workers)...")

    read_results = load_tile_planes(ok, args.workers)
    read_failures = [r for r in read_results if not r["ok"]]
    for failure in read_failures:
        print(f"  WARNING: could not read RL0 planes for {failure['round']}/{failure['tile']}: {failure['error']}")
    good = [r for r in read_results if r["ok"]]
    print(f"Read {len(good)}/{len(read_results)} tiles OK at ResolutionLevel 0")

    pixel_size_by_round = ok.set_index("round")["pixel_size_um_level0"].groupby(level=0).first().to_dict()

    channel_names = list(CHANNEL_LABELS.values())
    by_folder_channel: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}
    by_channel_pooled: dict[str, list[np.ndarray]] = {name: [] for name in channel_names}
    for r in good:
        folder_planes = by_folder_channel.setdefault(r["round"], {name: [] for name in channel_names})
        for name, plane in r["planes"].items():
            folder_planes[name].append((r["tile"], plane))
            by_channel_pooled[name].append(plane)

    pooled_fields: dict[str, IlluminationField] = {}
    for name in channel_names:
        pooled_fields[name] = estimate_illumination_field(by_channel_pooled[name])
        tifffile.imwrite(flatfield_dir / f"pooled_{name}.tif", pooled_fields[name].field.astype(np.float32))
        print(
            f"pooled/{name}: n_tiles={pooled_fields[name].n_tiles} smooth_bowl_score={pooled_fields[name].score:.4f} "
            f"gate={'PASS' if pooled_fields[name].passed_gate else 'FAIL'}"
        )

    field_selection: dict[tuple[str, str], dict] = {}  # (round, channel) -> {field, source, score, n_tiles}
    for folder, per_channel in by_folder_channel.items():
        for name in channel_names:
            planes = per_channel[name]
            n_tiles = len(planes)
            own_field = None
            if n_tiles >= MIN_TILES_FOR_OWN_FIELD:
                own_field = estimate_illumination_field([p for _, p in planes])
            if own_field is not None and own_field.passed_gate:
                selection = {"field": own_field.field, "source": "own", "score": own_field.score, "n_tiles": n_tiles}
            else:
                reason = "too few tiles (<6)" if own_field is None else f"smooth-bowl gate failed (score={own_field.score:.4f})"
                selection = {"field": pooled_fields[name].field, "source": "pooled", "score": None if own_field is None else own_field.score, "n_tiles": n_tiles, "fallback_reason": reason}
                print(f"{folder}/{name}: falling back to pooled field - {reason}")
            field_selection[(folder, name)] = selection
            tifffile.imwrite(flatfield_dir / f"{folder}_{name}.tif", selection["field"].astype(np.float32))

    # --- Verification: apply the selected field, recompute radial profiles ---
    bg_before: dict[tuple[str, str], list[np.ndarray]] = {}
    bg_after: dict[tuple[str, str], list[np.ndarray]] = {}
    nuc_before: dict[str, list[np.ndarray]] = {}
    nuc_after: dict[str, list[np.ndarray]] = {}
    example_tiles: dict[tuple[str, str], dict] = {}

    for r in good:
        folder = r["round"]
        pixel_size_um = float(pixel_size_by_round[folder])
        dapi_raw = r["planes"].get("DAPI_405")
        if dapi_raw is None:
            continue
        dapi_field = field_selection[(folder, "DAPI_405")]["field"]
        dapi_corrected = dapi_raw.astype(np.float32) / dapi_field
        mask_before, centroids_before = segment_foreground_2d(dapi_raw, pixel_size_um)
        mask_after, centroids_after = segment_foreground_2d(dapi_corrected, pixel_size_um)

        nuc_before.setdefault(folder, []).append(nucleus_density_radial_profile(centroids_before, dapi_raw.shape, pixel_size_um))
        nuc_after.setdefault(folder, []).append(nucleus_density_radial_profile(centroids_after, dapi_corrected.shape, pixel_size_um))

        for name, raw_plane in r["planes"].items():
            key = (folder, name)
            field = field_selection[key]["field"]
            corrected_plane = raw_plane.astype(np.float32) / field
            bg_before.setdefault(key, []).append(background_radial_profile(raw_plane, mask_before, pixel_size_um))
            bg_after.setdefault(key, []).append(background_radial_profile(corrected_plane, mask_after, pixel_size_um))
            if key not in example_tiles:
                example_tiles[key] = {"tile": r["tile"], "raw": raw_plane, "corrected": corrected_plane}

    verification_rows = []
    for folder in sorted(by_folder_channel):
        for name in channel_names:
            key = (folder, name)
            if key not in bg_before:
                continue
            pooled_before = np.nanmean(np.vstack(bg_before[key]), axis=0) if len(bg_before[key]) >= MIN_TILES_FOR_RADIAL_METRIC else None
            pooled_after = np.nanmean(np.vstack(bg_after[key]), axis=0) if len(bg_after[key]) >= MIN_TILES_FOR_RADIAL_METRIC else None
            slope_before, edge_before = radial_slope_and_edge_pct(pooled_before) if pooled_before is not None else (float("nan"), float("nan"))
            slope_after, edge_after = radial_slope_and_edge_pct(pooled_after) if pooled_after is not None else (float("nan"), float("nan"))
            verification_rows.append(
                {
                    "round": folder,
                    "channel": name,
                    "source": field_selection[key]["source"],
                    "n_tiles": field_selection[key]["n_tiles"],
                    "edge_vs_center_pct_before": edge_before,
                    "edge_vs_center_pct_after": edge_after,
                    "slope_before": slope_before,
                    "slope_after": slope_after,
                }
            )
    verification_df = pd.DataFrame(verification_rows)
    print("\nPost-correction radial background check (edge-vs-centre %, before -> after):")
    for _, row in verification_df.iterrows():
        print(f"  {row['round']:16s} {row['channel']:10s} [{row['source']:6s}] {row['edge_vs_center_pct_before']:+7.1f}% -> {row['edge_vs_center_pct_after']:+6.2f}%")

    nuc_slope_rows = []
    for folder in sorted(nuc_before):
        if len(nuc_before[folder]) < MIN_TILES_FOR_RADIAL_METRIC:
            continue
        pooled_nuc_before = np.nanmean(np.vstack(nuc_before[folder]), axis=0)
        pooled_nuc_after = np.nanmean(np.vstack(nuc_after[folder]), axis=0)
        slope_b, _ = radial_slope_and_edge_pct(pooled_nuc_before)
        slope_a, _ = radial_slope_and_edge_pct(pooled_nuc_after)
        nuc_slope_rows.append({"round": folder, "density_slope_before": slope_b, "density_slope_after": slope_a})
    nuc_df = pd.DataFrame(nuc_slope_rows)
    print("\nPost-correction DAPI nucleus-density-vs-radius slope (nuclei/um2 per um), before -> after:")
    for _, row in nuc_df.iterrows():
        print(f"  {row['round']:16s} {row['density_slope_before']:+.6f} -> {row['density_slope_after']:+.6f}")

    # --- PDF ---
    pdf_path = output_dir / "stage1b_flatfield.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        fig.suptitle("Stage 1b: retrospective flat-field correction - summary", fontsize=13, y=0.98)
        ax = fig.add_axes((0.03, 0.08, 0.94, 0.85))
        ax.axis("off")
        cols = ["round", "channel", "source", "n_tiles", "edge_vs_center_pct_before", "edge_vs_center_pct_after"]
        table_df = verification_df[cols].copy()
        for c in ("edge_vs_center_pct_before", "edge_vs_center_pct_after"):
            table_df[c] = table_df[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "n/a")
        table = ax.table(cellText=table_df.values, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.2)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        for name in channel_names:
            fig = plt.figure(figsize=(11, 8.5), dpi=300)
            fig.suptitle(f"Pooled (fallback) illumination field - {name}", fontsize=12)
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(pooled_fields[name].field, cmap="viridis")
            ax.set_title(
                f"n_tiles={pooled_fields[name].n_tiles}  smooth_bowl_score={pooled_fields[name].score:.4f}  "
                f"gate={'PASS' if pooled_fields[name].passed_gate else 'FAIL'}",
                fontsize=8,
            )
            fig.colorbar(im, ax=ax, fraction=0.046)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        for folder in sorted(by_folder_channel):
            for name in channel_names:
                key = (folder, name)
                if key not in field_selection:
                    continue
                sel = field_selection[key]
                fig = plt.figure(figsize=(11, 8.5), dpi=300)
                grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])

                ax_field = fig.add_subplot(grid[0, 0])
                im = ax_field.imshow(sel["field"], cmap="viridis")
                ax_field.set_title(f"field ({sel['source']})", fontsize=8)
                ax_field.axis("off")
                fig.colorbar(im, ax=ax_field, fraction=0.046)

                vrow = verification_df[(verification_df["round"] == folder) & (verification_df["channel"] == name)]
                ax_radial = fig.add_subplot(grid[0, 1:])
                if len(vrow) and key in bg_before:
                    pooled_before = np.nanmean(np.vstack(bg_before[key]), axis=0)
                    pooled_after = np.nanmean(np.vstack(bg_after[key]), axis=0)
                    ax_radial.plot(RADIAL_BIN_CENTERS_UM, pooled_before, marker="o", label="before")
                    ax_radial.plot(RADIAL_BIN_CENTERS_UM, pooled_after, marker="o", label="after")
                    ax_radial.set_xlabel("radius (um)")
                    ax_radial.set_ylabel("background median intensity")
                    ax_radial.legend(fontsize=7)
                ax_radial.set_title("radial background profile", fontsize=8)

                example = example_tiles.get(key)
                if example is not None:
                    ax_raw = fig.add_subplot(grid[1, 0])
                    ax_raw.imshow(display_u16_to_u8(example["raw"]), cmap="gray")
                    ax_raw.set_title("example tile - raw", fontsize=8)
                    ax_raw.axis("off")

                    ax_corr = fig.add_subplot(grid[1, 1])
                    ax_corr.imshow(display_u16_to_u8(example["corrected"]), cmap="gray")
                    ax_corr.set_title("example tile - corrected", fontsize=8)
                    ax_corr.axis("off")

                vtext = ""
                if len(vrow):
                    row0 = vrow.iloc[0]
                    vtext = (
                        f"edge-vs-centre: {row0['edge_vs_center_pct_before']:.1f}% -> {row0['edge_vs_center_pct_after']:.1f}%  "
                        f"slope: {row0['slope_before']:.3g} -> {row0['slope_after']:.3g}"
                    )
                reason = f"  ({sel.get('fallback_reason', '')})" if sel["source"] == "pooled" else ""
                gate_passed = sel["source"] == "own" or pooled_fields[name].passed_gate
                caption = (
                    f"round={folder}  channel={name}  source={sel['source']} (n_tiles={sel['n_tiles']}){reason}\n"
                    f"field_passed_smooth_bowl_gate={gate_passed}\n"
                    f"example tile={example['tile'] if example else 'n/a'}\n{vtext}"
                )
                fig.suptitle(caption, fontsize=8, y=0.99, ha="center")
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

    print(f"\nWrote {pdf_path}")

    # --- Update stage0_tile_qc.csv: merge well_map + flatfield columns ---
    # Idempotent: drop any columns this stage (or a prior run of it) added
    # before merging/recomputing, so re-running never produces pandas'
    # _x/_y suffix collisions.
    stage1b_added_cols = [c for c in tile_df.columns if c == "well" or c.startswith("well_position_id") or c.startswith("flatfield_")]
    tile_df = tile_df.drop(columns=stage1b_added_cols, errors="ignore")

    backup_path = tile_csv.with_suffix(".csv.pre_stage1b.bak")
    if not backup_path.exists():
        shutil.copy2(tile_csv, backup_path)
        print(f"Backed up {tile_csv} -> {backup_path}")

    well_map_path = output_dir / "well_map.csv"
    if well_map_path.exists():
        well_map = pd.read_csv(well_map_path)
        trf2_wells = well_map[well_map["dataset"] == "TRF2"][["round", "tile", "well", "well_position_id"]]
        tile_df = tile_df.merge(trf2_wells, on=["round", "tile"], how="left")
    else:
        print(f"WARNING: {well_map_path} not found - run Stage 1 first. Skipping well columns.")

    tile_df["flatfield_applied"] = tile_df["read_ok"] == True  # noqa: E712
    for name in channel_names:
        source_map = {(folder, n): sel["source"] for (folder, n), sel in field_selection.items()}
        path_map = {(folder, n): f"flatfield/{folder}_{n}.tif" for (folder, n) in field_selection}
        # The field actually applied to a tile passed the smooth-bowl gate
        # either on its own (an "own" selection only happens when it passed)
        # or via the pooled fallback's own gate result - which is NOT
        # guaranteed to pass (see RFP_561/GFP_488 in the written report).
        # Downstream stages should treat a gate_passed=False correction with
        # more caution than a passing one.
        gate_map = {
            (folder, n): (True if sel["source"] == "own" else pooled_fields[n].passed_gate)
            for (folder, n), sel in field_selection.items()
        }
        tile_df[f"flatfield_source_{name}"] = tile_df.apply(
            lambda row, name=name: source_map.get((row["round"], name)) if row["read_ok"] == True else None, axis=1  # noqa: E712
        )
        tile_df[f"flatfield_gate_passed_{name}"] = tile_df.apply(
            lambda row, name=name: gate_map.get((row["round"], name)) if row["read_ok"] == True else None, axis=1  # noqa: E712
        )
        tile_df[f"flatfield_path_{name}"] = tile_df.apply(
            lambda row, name=name: path_map.get((row["round"], name)) if row["read_ok"] == True else None, axis=1  # noqa: E712
        )

    tile_df.to_csv(tile_csv, index=False)
    print(f"Updated {tile_csv} with well + flatfield columns")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
