"""Stage 0: Imaris (.ims) reader QC/triage pass.

Walks every tile under the TRF2 root, computes a per-tile QC row (focus,
sharpness, saturation, nucleus count, etc.) at ResolutionLevel 1 for speed,
and writes:

  stage0_tile_qc.csv   one row per tile
  stage0_summary.csv   one row per folder ("round")
  stage0_qc.pdf        one reviewable page per successfully-read tile

This is QC/triage only: it does not assign wells, does not run real
segmentation, and does not tune thresholds. See the module docstring in
czi_foci/ims_reader.py for the three .ims traps (char-array attrs, Z padding,
resolution pyramid) this script relies on.

Stage 0b correction (see git history): the nucleus-size constants used to be
hardcoded in pixels and calibrated for ResolutionLevel 1 (0.2 um/px), but the
QC level had been bumped to 2 (0.402 um/px) without rescaling them, silently
turning a 30 um2 minimum nucleus area into 129 um2 and deleting real nuclei.
All segmentation constants are now physical (um / um2) and converted to
pixels at runtime from the pixel size actually measured for each tile, so
they are correct at whatever resolution level is actually read. The
bleaching and vignetting checks were also replaced: the old bleaching metric
(per-slice p99.5 vs z) was confounded with focus position, not photobleaching,
and the old vignetting metric (nucleus count density vs radius) was
confounded by seeding density and border exclusion.
"""

from __future__ import annotations

import argparse
import re
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from skimage import feature, filters, morphology, segmentation

from czi_foci.ims_reader import (
    inspect_ims,
    pixel_size_um_ims,
    read_channel_arrays_ims,
    recover_metadata_from_sibling,
    resolve_channel_indices,
    stage_position_mm,
)

# Stage 0 is a triage pass: it answers "is this tile usable", not "how many
# foci are there". Level 1 (1024x1024 for a 2048 tile, 0.2 um/px here) is the
# level the nucleus-size constants below are calibrated for. It is NOT
# adequate for foci — a ~250 nm diffraction-limited focus is still well below
# Nyquist at 0.2 um/px. Quantitative work reads level 0.
QC_RESOLUTION_LEVEL = 1

# Segmentation constants are physical (um / um2), never pixels, and are
# converted to pixels at runtime from the pixel size actually measured for
# the tile being processed (see count_nuclei). This is the fix for the bug
# where these values were pixel counts calibrated for one resolution level
# and silently wrong at another.
MIN_NUCLEUS_AREA_UM2 = 30.0
MIN_NUCLEUS_SEPARATION_UM = 3.6
# Pre-segmentation smoothing sigma. Physical equivalent of the original 2 px
# at the level-1 calibration (2 * 0.2 um/px).
SMOOTHING_SIGMA_UM = 0.4

# Channel *roles* are resolved per file from the recorded channel names, never
# assumed positionally — DAPI is channel 1 in these .ims files, not 0 (the
# Zeiss .czi convention) or 2. These labels are for column naming only.
CHANNEL_LABELS = {"561": "RFP_561", "405": "DAPI_405", "488": "GFP_488"}

SATURATION_THRESHOLD = 65000
SHARPNESS_CONTAINMENT_THRESHOLD = 0.35

# Matched-sharpness intensity asymmetry: a pre/post-focus slice pair counts as
# "matched" only if their normalised sharpness values are within this much of
# each other.
ASYMMETRY_SHARPNESS_TOLERANCE = 0.05
# Folder-level checks need at least this many usable tiles, else N/A.
MIN_TILES_FOR_FOLDER_METRIC = 3

# Fixed radial bins (um from tile centre) for the background-vignetting
# check, shared across every tile so per-tile profiles can be pooled directly
# within a folder. 205.8 um is the measured field of view for this
# instrument (2048 px * 0.1005 um/px at level 0); it does not change with
# resolution level, only the sampling density does.
FIELD_OF_VIEW_UM = 2048 * 0.1005
N_RADIAL_BINS = 8
RADIAL_BIN_EDGES_UM = np.linspace(0.0, FIELD_OF_VIEW_UM / 2.0 * np.sqrt(2.0), N_RADIAL_BINS + 1)
RADIAL_BIN_CENTERS_UM = (RADIAL_BIN_EDGES_UM[:-1] + RADIAL_BIN_EDGES_UM[1:]) / 2.0


ROUND_SUFFIX_RE = re.compile(
    r"_(?P<number>\d+)(?P<deconvolved>_deconvolved(?:_\d+)?)?$|(?P<deconv_only>_deconvolved(?:_\d+)?)$"
)


def derive_round(folder_name: str) -> str:
    """Derive a short, STABLE 'round' label from a single folder's own name.

    Deliberately depends on nothing but this folder. The previous
    implementation derived the label from ``os.path.commonprefix`` over every
    folder in the root, so adding or removing one folder silently relabelled
    every other folder — and an analysis output directory appearing alongside
    the acquisitions was enough to degenerate every label to the full folder
    name, making two runs of this script impossible to join.

    ``3x3 test_..._TRF2 x yH2AX_12_Field_1``             -> ``_12``
    ``3x3 test_..._TRF2 x yH2AX_1_deconvolved_Field_1``  -> ``_1_deconvolved``
    ``3x3 test_..._TRF2 x yH2AX_Field_1``                -> ``(unnumbered)``
    """
    base = folder_name
    for suffix in ("_Field_1", "_Field"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    match = ROUND_SUFFIX_RE.search(base)
    if not match:
        return "(unnumbered)"
    if match.group("deconv_only"):
        return f"(unnumbered){match.group('deconv_only')}"
    label = f"_{match.group('number')}"
    if match.group("deconvolved"):
        label += match.group("deconvolved")
    return label


@dataclass
class TileResult:
    row: dict
    thumbnail: np.ndarray | None = None
    panels: dict | None = None  # {"DAPI_405": plane, "RFP_561": plane, "GFP_488": plane}
    sharpness: np.ndarray | None = None
    # {channel_name: array of length N_RADIAL_BINS, NaN where no background
    # pixels fell in that bin for this tile}. Not written to the per-tile CSV
    # (too wide); pooled across tiles per folder for the vignetting check.
    radial_background: dict = field(default_factory=dict)


def per_slice_sharpness(dapi_stack: np.ndarray) -> np.ndarray:
    n_z = dapi_stack.shape[0]
    sharp = np.zeros(n_z, dtype=np.float64)
    for z in range(n_z):
        a = dapi_stack[z].astype(np.float32)
        sharp[z] = ndi.laplace(ndi.gaussian_filter(a, 1.0)).var() / max(float(a.mean()) ** 2, 1e-9)
    peak = sharp.max()
    if peak > 0:
        sharp = sharp / peak
    return sharp


def usable_z_range(sharp: np.ndarray, threshold: float = SHARPNESS_CONTAINMENT_THRESHOLD) -> tuple[int, int]:
    """Contiguous slice span around the focus peak where sharpness >= threshold.

    This is the z range downstream stages will actually be able to analyse;
    slices outside it are too far out of focus to trust.
    """
    peak = int(sharp.argmax())
    lo = peak
    while lo - 1 >= 0 and sharp[lo - 1] >= threshold:
        lo -= 1
    hi = peak
    while hi + 1 < len(sharp) and sharp[hi + 1] >= threshold:
        hi += 1
    return lo, hi


def choose_resolution_level(inspection: dict) -> int:
    """Pick the QC level, falling back to the deepest level the file actually has.

    Not every .ims on this instrument carries a pyramid: the single-plane TSIV
    acquisitions have ``ResolutionLevel 0`` only, so asking for level 1
    unconditionally would fail them. The level actually used is recorded per
    row; nucleus-size constants are physical and need no rescaling when the
    level falls back (see count_nuclei).
    """
    available = [
        int(name.rsplit(" ", 1)[-1])
        for name in inspection.get("resolution_levels") or []
    ]
    if not available:
        return QC_RESOLUTION_LEVEL
    return min(QC_RESOLUTION_LEVEL, max(available))


def nucleus_labels(dapi_stack: np.ndarray, peak_idx: int, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Label nuclei on a focus-centred DAPI max sub-projection.

    All size constants are physical (um / um2) and are converted to pixels
    here from ``pixel_size_um``, the actual pixel size of the data being
    read — never a value baked in for a particular resolution level. This is
    the fix for the bug where a pixel-count constant calibrated for level 1
    was silently 4x too large in physical terms when applied at level 2.

    Returns (labels, nucleus_mask). Separated from :func:`count_nuclei` so
    stages that need the per-nucleus label image (Stage 2's within-nucleus
    permutation null) get exactly the segmentation Stage 0 counted, rather
    than a re-implementation that could drift from it.
    """
    lo = max(peak_idx - 2, 0)
    proj = dapi_stack[lo : peak_idx + 3].astype(np.float32).max(0)
    sigma_px = SMOOTHING_SIGMA_UM / pixel_size_um
    proj = ndi.gaussian_filter(proj, sigma_px)
    zeros = np.zeros(proj.shape, dtype=np.int32)
    if np.unique(proj).size <= 1:
        return zeros, np.zeros(proj.shape, dtype=bool)

    min_size_px = max(1, int(round(MIN_NUCLEUS_AREA_UM2 / (pixel_size_um**2))))
    min_distance_px = max(1, int(round(MIN_NUCLEUS_SEPARATION_UM / pixel_size_um)))

    m = proj > filters.threshold_otsu(proj)
    m = morphology.remove_small_objects(m, min_size_px)
    m = ndi.binary_fill_holes(m)
    if not m.any():
        return zeros, m
    dist = ndi.distance_transform_edt(m)
    pks = feature.peak_local_max(dist, min_distance=min_distance_px, labels=m)
    if len(pks) == 0:
        return zeros, m
    mk = np.zeros(m.shape, int)
    mk[tuple(pks.T)] = np.arange(1, len(pks) + 1)
    lab = segmentation.watershed(-dist, mk, mask=m)
    return lab.astype(np.int32), m


def count_nuclei(dapi_stack: np.ndarray, peak_idx: int, pixel_size_um: float) -> tuple[int, np.ndarray]:
    """Count nuclei on a focus-centred DAPI max sub-projection.

    Returns (n_nuclei, nucleus_mask) — the mask is reused by the background
    vignetting check so "non-nuclear background" means the same thing there.
    """
    labels, mask = nucleus_labels(dapi_stack, peak_idx, pixel_size_um)
    return int(labels.max()), mask


def channel_percentiles(stack: np.ndarray) -> dict:
    flat = stack.astype(np.float32)
    p5, p50, p999 = np.percentile(flat, [5, 50, 99.9])
    return {
        "p5": float(p5),
        "p50": float(p50),
        "p999": float(p999),
        "max": float(stack.max()),
        "saturated_frac": float(np.count_nonzero(stack >= SATURATION_THRESHOLD) / stack.size),
    }


def intensity_asymmetry(sharp: np.ndarray, channel_slice_means: np.ndarray, peak_idx: int) -> float:
    """Relative intensity difference between post- and pre-focus slices matched by sharpness.

    The old bleaching metric (per-slice p99.5 vs z-index) was confounded with
    focus position (r=+0.61 DAPI, +0.65 GFP) rather than measuring
    photobleaching. Matching slices on sharpness instead of z-distance from
    the peak holds focus quality constant, so any remaining intensity
    difference between the pre-focus and post-focus side is attributable to
    imaging order (cumulative photon dose), i.e. actual bleaching.

    Positive means the post-focus (imaged later) side is brighter than its
    sharpness-matched pre-focus counterpart; negative is consistent with
    photobleaching. NaN if there are no matched pairs within tolerance.
    """
    pre_idx = np.arange(0, peak_idx)
    post_idx = np.arange(peak_idx + 1, len(sharp))
    if len(pre_idx) == 0 or len(post_idx) == 0:
        return float("nan")

    diffs = []
    for j in post_idx:
        i = pre_idx[int(np.argmin(np.abs(sharp[pre_idx] - sharp[j])))]
        if abs(sharp[i] - sharp[j]) > ASYMMETRY_SHARPNESS_TOLERANCE:
            continue
        denom = (channel_slice_means[j] + channel_slice_means[i]) / 2.0
        if denom <= 1e-9:
            continue
        diffs.append((channel_slice_means[j] - channel_slice_means[i]) / denom)

    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def background_radial_profile(
    plane: np.ndarray, nucleus_mask: np.ndarray, pixel_size_um: float
) -> np.ndarray:
    """Per-bin median intensity of non-nuclear background pixels vs radial distance.

    Bins are fixed physical distances from the tile centre (RADIAL_BIN_EDGES_UM)
    so per-tile profiles can be pooled directly across a folder. NaN where a
    tile has no background pixels in a given bin.
    """
    yy, xx = np.indices(plane.shape)
    cy, cx = (plane.shape[0] - 1) / 2.0, (plane.shape[1] - 1) / 2.0
    radius_um = np.hypot(yy - cy, xx - cx) * pixel_size_um

    background = ~nucleus_mask
    bin_idx = np.clip(np.digitize(radius_um, RADIAL_BIN_EDGES_UM) - 1, 0, N_RADIAL_BINS - 1)

    flat_vals = plane[background].astype(np.float32)
    flat_bins = bin_idx[background]
    result = np.full(N_RADIAL_BINS, np.nan)
    for b in range(N_RADIAL_BINS):
        sel = flat_vals[flat_bins == b]
        if sel.size > 0:
            result[b] = float(np.median(sel))
    return result


def process_tile(args: tuple[Path, str]) -> TileResult:
    path, folder_name = args
    deconvolved = "deconvolved" in path.name.lower() or "deconvolved" in folder_name.lower()
    row: dict = {
        "round": folder_name,
        "tile": path.name,
        "deconvolved": deconvolved,
        "read_ok": False,
        "error": "",
        "recovered_metadata_from_sibling": False,
    }

    try:
        inspection = inspect_ims(path)

        if "missing DataSet group" in inspection.get("errors", []):
            # No pixel data at all (e.g. a truncated/empty .ims) - metadata
            # recovery would be pointless since there is nothing to read.
            row["error"] = "; ".join(inspection["errors"])
            return TileResult(row=row)

        metadata = inspection["metadata"]
        true_z = inspection.get("true_z")

        if metadata is None or not metadata.get("has_dataset_info") or true_z is None:
            sibling_metadata = recover_metadata_from_sibling(path)
            if sibling_metadata is not None:
                metadata = sibling_metadata
                true_z = int(metadata["image"]["Z"])
                row["recovered_metadata_from_sibling"] = True
            else:
                row["error"] = "; ".join(inspection.get("errors") or ["missing DataSetInfo; no sibling to recover from"])
                return TileResult(row=row)

        if any("missing Data object" in e for e in inspection.get("errors", [])) and len(inspection.get("channels", {})) == 0:
            row["error"] = "no channel Data objects present"
            return TileResult(row=row)

        # Resolve channel roles from the recorded names. Never positional:
        # DAPI is channel 1 here, and a folder acquired in a different order
        # must fail loudly rather than be silently mislabelled.
        try:
            channel_index = resolve_channel_indices(metadata)
        except ValueError as exc:
            row["error"] = f"channel resolution failed: {exc}"
            return TileResult(row=row)
        dapi_channel = channel_index["405"]

        # Not every file has a pyramid; use the deepest level it does have.
        level = choose_resolution_level(inspection)

        # resolution_level MUST match the level the pixels are read at below.
        z_step_um, y_um, x_um = pixel_size_um_ims(
            metadata, mode="3d", resolution_level=level
        )
        stage_x_mm, stage_y_mm = stage_position_mm(metadata)
        _, _, x_um_level0 = pixel_size_um_ims(metadata, mode="3d", resolution_level=0)
        acquisition = metadata.get("custom", {}).get("DateAndTime", "")

        arrays = read_channel_arrays_ims(path, {**inspection, "true_z": true_z}, mode="3d", resolution_level=level)
        if dapi_channel not in arrays:
            row["error"] = f"missing DAPI (resolved as channel {dapi_channel}) Data object"
            return TileResult(row=row)

        dapi = arrays[dapi_channel]
        sharp = per_slice_sharpness(dapi)
        focus_peak_idx = int(sharp.argmax())
        axially_contained = bool(sharp[0] < SHARPNESS_CONTAINMENT_THRESHOLD and sharp[-1] < SHARPNESS_CONTAINMENT_THRESHOLD)
        usable_lo, usable_hi = usable_z_range(sharp)
        n_nuclei, nucleus_mask = count_nuclei(dapi, focus_peak_idx, pixel_size_um=x_um)

        row.update(
            {
                "read_ok": True,
                "source_path": str(path),
                "resolution_level": level,
                "resolution_level_requested": QC_RESOLUTION_LEVEL,
                "channel_map": ";".join(
                    f"{wavelength}=C{index}" for wavelength, index in sorted(channel_index.items())
                ),
                "true_z": int(true_z),
                "z_step_um": z_step_um,
                "z_span_um": z_step_um * (true_z - 1) if true_z > 1 else 0.0,
                # Spacing OF THE PIXELS MEASURED, i.e. at resolution_level.
                # level-0 spacing is recorded separately so the two can never
                # be confused again.
                "pixel_size_um": x_um,
                "pixel_size_um_level0": x_um_level0,
                "stage_x_mm": stage_x_mm,
                "stage_y_mm": stage_y_mm,
                "acquisition_datetime": acquisition,
                "focus_peak_idx": focus_peak_idx,
                "sharpness_at_first": float(sharp[0]),
                "sharpness_at_last": float(sharp[-1]),
                "axially_contained": axially_contained,
                "usable_z_range": f"{usable_lo}-{usable_hi}",
                "usable_z_span_slices": usable_hi - usable_lo + 1,
                "n_nuclei": n_nuclei,
            }
        )

        panels = {}
        radial_background: dict[str, np.ndarray] = {}
        for wavelength, index in channel_index.items():
            name = CHANNEL_LABELS[wavelength]
            if index not in arrays:
                continue
            stack = arrays[index]
            stats = channel_percentiles(stack)
            for key, value in stats.items():
                row[f"{name}_{key}"] = value

            slice_means = stack.reshape(stack.shape[0], -1).mean(axis=1).astype(np.float64)
            row[f"intensity_asymmetry_{name}"] = (
                intensity_asymmetry(sharp, slice_means, focus_peak_idx) if axially_contained else float("nan")
            )

            plane_idx = min(focus_peak_idx, stack.shape[0] - 1)
            plane = stack[plane_idx]
            panels[name] = plane
            radial_background[name] = background_radial_profile(plane, nucleus_mask, x_um)

        return TileResult(row=row, thumbnail=panels.get("DAPI_405"), panels=panels, sharpness=sharp, radial_background=radial_background)

    except Exception as exc:  # noqa: BLE001 - QC pass must not crash on a bad tile
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        return TileResult(row=row)


def discover_tiles(root: Path, output_dir: Path) -> tuple[list[tuple[Path, str]], list[str], dict[str, int]]:
    output_dir = output_dir.resolve()
    # Skip analysis output directories. Only the *current* output_dir was
    # excluded before, so a previous run's output directory was treated as an
    # acquisition folder.
    folders = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.resolve() != output_dir
        and not p.name.startswith((".", "_"))
        and any(p.glob("*.ims"))
    )
    tiles: list[tuple[Path, str]] = []
    tile_counts: dict[str, int] = {}
    for folder in folders:
        folder_tiles = sorted(folder.glob("*.ims"))
        tile_counts[folder.name] = len(folder_tiles)
        for tile in folder_tiles:
            tiles.append((tile, folder.name))
    return tiles, [f.name for f in folders], tile_counts


# Failure modes that occur before any resolution-level-dependent read is
# attempted (missing DataSet/DataSetInfo entirely, with no sibling to recover
# from). These will fail identically at any resolution level, so a tile with
# one of these errors in a previous run can be reused without re-reading it.
LEVEL_INDEPENDENT_FAILURE_PREFIXES = ("missing DataSet group", "missing DataSetInfo")


def load_previous_tile_qc(output_dir: Path) -> dict[str, dict]:
    """Load a prior stage0_tile_qc.csv keyed by source_path, if one exists."""
    path = output_dir / "stage0_tile_qc.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "source_path" not in df.columns:
        return {}
    return {row["source_path"]: row.to_dict() for _, row in df.iterrows() if isinstance(row.get("source_path"), str)}


def display_u16_to_u8(plane: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(plane, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((plane.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def render_tile_page(pdf: PdfPages, row: dict, panels: dict, sharpness: np.ndarray) -> None:
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1.4])

    for col, name in enumerate(["DAPI_405", "RFP_561", "GFP_488"]):
        ax = fig.add_subplot(grid[0, col])
        if name in panels:
            ax.imshow(display_u16_to_u8(panels[name]), cmap="gray", interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    ax_curve = fig.add_subplot(grid[1, :])
    ax_curve.plot(np.arange(len(sharpness)), sharpness, marker="o", markersize=3)
    ax_curve.axvline(row["focus_peak_idx"], color="red", linestyle="--", linewidth=0.8)
    ax_curve.axhline(SHARPNESS_CONTAINMENT_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
    ax_curve.set_xlabel("Z slice")
    ax_curve.set_ylabel("normalised sharpness")
    ax_curve.set_ylim(0, 1.05)

    caption = (
        f"round={row['round']}  tile={row['tile']}  deconvolved={row['deconvolved']}\n"
        f"Z={row.get('true_z')}  z_step={row.get('z_step_um'):.3f} um  "
        f"z_span={row.get('z_span_um'):.2f} um  "
        f"xy_pixel={row.get('pixel_size_um'):.4f} um @ RL{row.get('resolution_level')}\n"
        f"focus_peak_idx={row['focus_peak_idx']}  axially_contained={row['axially_contained']}  "
        f"usable_z_range={row.get('usable_z_range')}  n_nuclei={row['n_nuclei']}  "
        f"acquired={row.get('acquisition_datetime', '')}"
    )
    fig.suptitle(caption, fontsize=8, y=0.99, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def render_summary_page(pdf: PdfPages, summary_df: pd.DataFrame, failures: list[dict], thumbnails: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    fig.suptitle("Stage 0 QC: per-folder summary", fontsize=13, y=0.98)

    ax_table = fig.add_axes((0.03, 0.55, 0.94, 0.4))
    ax_table.axis("off")
    cols = ["round", "n_tiles", "true_z", "z_step_um", "focus_range", "verdict"]
    table_data = summary_df[cols].astype(str).values if len(summary_df) else [[""] * len(cols)]
    table = ax_table.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.3)

    n_thumbs = min(8, len(thumbnails))
    if n_thumbs:
        thumb_items = list(thumbnails.items())[:n_thumbs]
        for i, (folder, thumb) in enumerate(thumb_items):
            ax = fig.add_axes((0.03 + (i % 4) * 0.235, 0.05 + (i // 4) * 0.22, 0.21, 0.2))
            ax.imshow(display_u16_to_u8(thumb), cmap="gray")
            ax.set_title(folder[-28:], fontsize=5)
            ax.axis("off")

    if failures:
        text = "Read failures:\n" + "\n".join(f"- {f['round']}/{f['tile']}: {f['error'].splitlines()[0]}" for f in failures[:20])
        fig.text(0.03, 0.02, text, fontsize=6, va="bottom", family="monospace")

    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def summarize_folders(tile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for folder, group in tile_df.groupby("round", sort=False):
        ok = group[group["read_ok"]]
        n_tiles = len(group)
        n_ok = len(ok)
        if n_ok == 0:
            rows.append(
                {
                    "round": folder,
                    "n_tiles": n_tiles,
                    "n_read_ok": 0,
                    "true_z": float("nan"),
                    "z_step_um": float("nan"),
                    "focus_range": "",
                    "frac_axially_contained": float("nan"),
                    "verdict": "UNUSABLE: no tiles could be read",
                }
            )
            continue
        z_vals = ok["true_z"].dropna().unique()
        z_step_vals = ok["z_step_um"].dropna().unique()
        focus_lo, focus_hi = ok["focus_peak_idx"].min(), ok["focus_peak_idx"].max()
        frac_contained = float(ok["axially_contained"].mean())
        verdict = "OK"
        if n_ok < n_tiles:
            verdict = f"PARTIAL: {n_tiles - n_ok}/{n_tiles} tiles failed to read"
        elif frac_contained < 0.5:
            verdict = "REVIEW: many tiles not axially contained"
        rows.append(
            {
                "round": folder,
                "n_tiles": n_tiles,
                "n_read_ok": n_ok,
                "true_z": z_vals[0] if len(z_vals) == 1 else "/".join(str(int(v)) for v in sorted(z_vals)),
                "z_step_um": round(float(z_step_vals[0]), 3) if len(z_step_vals) == 1 else "/".join(f"{v:.3f}" for v in sorted(z_step_vals)),
                "focus_range": f"{int(focus_lo)}-{int(focus_hi)}",
                "frac_axially_contained": frac_contained,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def summarize_asymmetry(tile_df: pd.DataFrame) -> pd.DataFrame:
    """Per-folder, per-channel intensity asymmetry at matched sharpness.

    Replaces the old bleaching metric. Computed only from axially-contained
    tiles (the asymmetry is only meaningful when the stack actually captures
    both sides of focus); folders with too few such tiles report "N/A"
    rather than a noisy number.
    """
    channel_names = list(CHANNEL_LABELS.values())
    rows = []
    for folder, group in tile_df.groupby("round", sort=False):
        contained = group[(group["read_ok"]) & (group["axially_contained"])]
        row = {"round": folder, "asymmetry_n_tiles_used": len(contained)}
        for name in channel_names:
            col = f"intensity_asymmetry_{name}"
            vals = contained[col].dropna() if col in contained.columns else pd.Series(dtype=float)
            if len(vals) < MIN_TILES_FOR_FOLDER_METRIC:
                row[f"intensity_asymmetry_{name}"] = "N/A"
            else:
                row[f"intensity_asymmetry_{name}"] = float(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_vignetting(tile_results: list[TileResult]) -> pd.DataFrame:
    """Per-folder, per-channel background-intensity vignetting check.

    Replaces the old nucleus-density-vs-radius metric, which was confounded
    by seeding density and border exclusion. This pools median non-nuclear
    background intensity, binned by radial distance from tile centre, across
    every successfully-read tile in a folder, then reports the linear slope
    and the edge-vs-centre percent change.
    """
    channel_names = list(CHANNEL_LABELS.values())
    by_folder: dict[str, list[TileResult]] = {}
    for result in tile_results:
        by_folder.setdefault(result.row["round"], []).append(result)

    rows = []
    for folder, results in by_folder.items():
        row = {"round": folder, "vignetting_n_tiles_used": len(results)}
        for name in channel_names:
            profiles = [r.radial_background[name] for r in results if name in r.radial_background]
            if len(profiles) < MIN_TILES_FOR_FOLDER_METRIC:
                row[f"vignetting_slope_{name}"] = "N/A"
                row[f"vignetting_edge_vs_center_pct_{name}"] = "N/A"
                row[f"vignetting_exceeds_10pct_{name}"] = "N/A"
                continue
            stacked = np.vstack(profiles)
            pooled = np.nanmean(stacked, axis=0)
            valid = ~np.isnan(pooled)
            if valid.sum() < 2:
                row[f"vignetting_slope_{name}"] = "N/A"
                row[f"vignetting_edge_vs_center_pct_{name}"] = "N/A"
                row[f"vignetting_exceeds_10pct_{name}"] = "N/A"
                continue
            slope, _ = np.polyfit(RADIAL_BIN_CENTERS_UM[valid], pooled[valid], 1)
            center_val = pooled[0] if not np.isnan(pooled[0]) else pooled[valid][0]
            edge_val = pooled[-1] if not np.isnan(pooled[-1]) else pooled[valid][-1]
            edge_pct = float((edge_val - center_val) / center_val * 100.0) if center_val > 0 else float("nan")
            row[f"vignetting_slope_{name}"] = float(slope)
            row[f"vignetting_edge_vs_center_pct_{name}"] = edge_pct
            row[f"vignetting_exceeds_10pct_{name}"] = bool(abs(edge_pct) > 10.0) if not np.isnan(edge_pct) else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TRF2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tiles (debugging)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles, folder_names, tile_counts = discover_tiles(args.root, output_dir)
    if args.limit:
        tiles = tiles[: args.limit]
    print(f"Discovered {len(tiles)} tiles across {len(folder_names)} folders under {args.root}")

    tile_args_all = [(path, derive_round(folder)) for path, folder in tiles]

    # Reuse the previous run where it is safe to: a tile that failed before
    # any resolution-level-dependent read was attempted (no DataSet/
    # DataSetInfo at all, no sibling to recover from) will fail identically
    # now, so skip re-reading it. Everything else is re-read exactly once,
    # at the new resolution level.
    previous = load_previous_tile_qc(output_dir)
    tile_args: list[tuple[Path, str]] = []
    reused_rows: list[dict] = []
    for path, folder in tile_args_all:
        old = previous.get(str(path))
        if (
            old is not None
            and old.get("read_ok") is False
            and isinstance(old.get("error"), str)
            and old["error"].startswith(LEVEL_INDEPENDENT_FAILURE_PREFIXES)
        ):
            reused_rows.append({**old, "round": folder})
        else:
            tile_args.append((path, folder))
    if reused_rows:
        print(f"Reusing {len(reused_rows)} known level-independent failures from the previous run (not re-read)")

    tile_rows: list[dict] = list(reused_rows)
    failures: list[dict] = [r for r in reused_rows if not r.get("read_ok")]
    thumbnails: dict[str, np.ndarray] = {}
    tile_results: list[TileResult] = []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap(process_tile, tile_args, chunksize=1):
            tile_rows.append(result.row)
            if not result.row["read_ok"]:
                failures.append(result.row)
                continue
            tile_results.append(result)
            if result.row["round"] not in thumbnails and result.thumbnail is not None:
                thumbnails[result.row["round"]] = result.thumbnail

    tile_df = pd.DataFrame(tile_rows)
    tile_csv = output_dir / "stage0_tile_qc.csv"
    if tile_csv.exists():
        tile_csv.unlink()
    tile_df.to_csv(tile_csv, index=False)

    summary_df = summarize_folders(tile_df)
    asymmetry_df = summarize_asymmetry(tile_df)
    vignette_df = summarize_vignetting(tile_results)
    summary_df = summary_df.merge(asymmetry_df, on="round", how="left")
    summary_df = summary_df.merge(vignette_df, on="round", how="left")

    empty_folders = [name for name, count in tile_counts.items() if count == 0]
    if empty_folders:
        empty_rows = [
            {
                "round": derive_round(name),
                "n_tiles": 0,
                "n_read_ok": 0,
                "true_z": float("nan"),
                "z_step_um": float("nan"),
                "focus_range": "",
                "frac_axially_contained": float("nan"),
                "verdict": "EMPTY: folder contains no .ims files",
            }
            for name in empty_folders
        ]
        summary_df = pd.concat([summary_df, pd.DataFrame(empty_rows)], ignore_index=True)

    summary_csv = output_dir / "stage0_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    summary_df.to_csv(summary_csv, index=False)

    pdf_path = output_dir / "stage0_qc.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    with PdfPages(pdf_path) as pdf:
        render_summary_page(pdf, summary_df, failures, thumbnails)
        for result in tile_results:
            render_tile_page(pdf, result.row, result.panels, result.sharpness)

    print(f"Wrote {tile_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {pdf_path}")
    print(f"Tiles: {len(tile_df)}  read_ok: {int(tile_df['read_ok'].sum())}  failed: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
