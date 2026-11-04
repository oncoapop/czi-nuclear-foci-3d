"""Can nucleoli be segmented reliably from DAPI alone, and does TRF2 sit in them?

WHY THIS EXISTS
  TRF2 has a documented nucleolar pool that appears, by immunofluorescence, as
  "discrete dots with sizes similar to those present in the nucleoplasm"
  (Zhang et al. 2004, J Cell Sci 117:3935; replicated by Yuan et al. 2017
  Cell Death Discov 3:17043 and Yuan et al. 2018 Cell Death Dis 9:518).
  Those dots are morphologically indistinguishable from telomeric foci in an
  assay like this one. The nucleolar pool is also cell-cycle dependent —
  prominent at G0/S, dispersing in G2 — and Palbociclib arrests cells in
  G0/G1, which is exactly the state where it should be most prominent. That
  makes nucleolar TRF2 a live alternative explanation both for the
  Palbociclib-associated mean shift and for part of the cell-to-cell
  dispersion being attributed to telomere instability.

  This script does NOT assume the confound is real. It asks two questions in
  order, and the second is only meaningful if the first passes:

    1. Can a nucleolus mask be built from DAPI alone, reliably enough to
       trust? There is no nucleolar marker channel (no fibrillarin, UBF or
       B23), so the mask has to come from DAPI-dim regions inside nuclei and
       be judged against what nucleoli are known to look like.
    2. If so, is TRF2 (561) actually enriched inside it, and is the enrichment
       specific — i.e. not matched by 488, which would instead indicate
       autofluorescent debris or a segmentation artefact rather than a real
       nucleolar TRF2 pool.

HOW THE MASK IS BUILT
  Per nucleus, not per field. Nuclear DAPI intensity varies between nuclei
  (and this dataset has a tilted slide, so it varies with position too), so a
  single field-wide threshold would find "nucleoli" in dim nuclei and none in
  bright ones. Each nucleus is thresholded against its own DAPI distribution.

RELIABILITY CRITERIA (pre-registered here, before looking at the output)
  A mask is only usable if the objects it finds look like nucleoli:
    - diameter roughly 0.5-3.5 um (literature range for human cells)
    - 1-5 per nucleus, most commonly 1-3
    - a few to ~20% of nuclear area
    - genuinely DAPI-dim: clear contrast against surrounding nucleoplasm
    - stable across tiles and across both cell lines
  Failing these means the mask is finding something else (chromatin texture,
  noise, segmentation edges) and the confound cannot be tested this way.

DAPI-saturated rounds (_11, _6) are excluded by default: clipping destroys the
intensity contrast this method depends on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stage0_ims_qc import nucleus_labels  # noqa: E402

from czi_foci.ims_reader import inspect_ims, read_channel_arrays_ims  # noqa: E402

ANALYSIS_RESOLUTION_LEVEL = 0

# Nucleolus candidates are DAPI-dim relative to their OWN nucleus. This
# percentile of the nucleus's DAPI distribution is the cut.
NUCLEOLUS_PERCENTILE = 25.0

# Literature plausibility bounds for a human nucleolus.
MIN_NUCLEOLUS_DIAM_UM = 0.5
MAX_NUCLEOLUS_DIAM_UM = 3.5

# A nucleolus is a solid body, not a thin sliver of chromatin texture.
MIN_SOLIDITY = 0.80


def channel_index_map(channel_map: str) -> dict[str, int]:
    """'405=C1;488=C2;561=C0' -> {'DAPI_405': 1, 'GFP_488': 2, 'RFP_561': 0}."""
    label_for = {"405": "DAPI_405", "488": "GFP_488", "561": "RFP_561"}
    out = {}
    for part in str(channel_map).split(";"):
        wavelength, channel = part.split("=")
        out[label_for[wavelength]] = int(channel.lstrip("C"))
    return out


def find_nucleoli(
    dapi_2d: np.ndarray,
    labels_2d: np.ndarray,
    pixel_size_um: float,
) -> tuple[np.ndarray, list[dict]]:
    """Per-nucleus DAPI-dim segmentation. Returns (mask, per-object records)."""
    mask = np.zeros(labels_2d.shape, dtype=bool)
    records: list[dict] = []
    px_area = pixel_size_um**2

    min_area_px = np.pi * (MIN_NUCLEOLUS_DIAM_UM / 2) ** 2 / px_area
    max_area_px = np.pi * (MAX_NUCLEOLUS_DIAM_UM / 2) ** 2 / px_area

    for region in measure.regionprops(labels_2d):
        sl = region.slice
        nuc = labels_2d[sl] == region.label
        dapi_local = dapi_2d[sl].astype(np.float32)
        inside = dapi_local[nuc]
        if inside.size < 50:
            continue

        cut = np.percentile(inside, NUCLEOLUS_PERCENTILE)
        dim = nuc & (dapi_local < cut)
        dim = ndi.binary_opening(dim, np.ones((3, 3)))
        dim = ndi.binary_fill_holes(dim)
        if not dim.any():
            continue

        sub_labels = measure.label(dim)
        kept = np.zeros_like(dim)
        nucleolus_count = 0
        for obj in measure.regionprops(sub_labels):
            if not (min_area_px <= obj.area <= max_area_px):
                continue
            if obj.solidity < MIN_SOLIDITY:
                continue
            kept |= sub_labels == obj.label
            nucleolus_count += 1
            diam_um = 2 * np.sqrt(obj.area * px_area / np.pi)
            obj_mask = sub_labels == obj.label
            inner = float(dapi_local[obj_mask].mean())
            outer = float(dapi_local[nuc & ~dim].mean()) if (nuc & ~dim).any() else np.nan
            records.append(
                {
                    "nucleus_label": int(region.label),
                    "nucleolus_diam_um": round(diam_um, 3),
                    "nucleolus_area_um2": round(obj.area * px_area, 3),
                    "solidity": round(float(obj.solidity), 3),
                    "dapi_inside": round(inner, 1),
                    "dapi_nucleoplasm": round(outer, 1),
                    "dapi_contrast": round(outer / inner, 3) if inner > 0 else np.nan,
                }
            )
        mask[sl] |= kept

        nuc_area_um2 = region.area * px_area
        kept_area_um2 = kept.sum() * px_area
        for rec in records[-nucleolus_count:] if nucleolus_count else []:
            rec["n_nucleoli_in_nucleus"] = nucleolus_count
            rec["nucleus_area_um2"] = round(nuc_area_um2, 1)
            rec["nucleolar_area_fraction"] = round(kept_area_um2 / nuc_area_um2, 4)

    return mask, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-qc", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--tiles-per-round", type=int, default=2)
    parser.add_argument("--rounds", type=str, default="_12,_10,_4,_1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    qc = pd.read_csv(args.tile_qc)
    design = pd.read_csv(args.design)
    qc = qc.merge(design, on=["round", "tile"], how="inner", suffixes=("", "_d"))
    qc = qc[qc.read_ok & ~qc.dapi_saturated_round.astype(bool)]

    wanted = [r.strip() for r in args.rounds.split(",")]
    obj_rows: list[dict] = []
    tile_rows: list[dict] = []

    for rnd in wanted:
        sub = qc[qc["round"] == rnd].head(args.tiles_per_round)
        if sub.empty:
            print(f"  {rnd}: no usable tiles (saturated or unread), skipped", flush=True)
            continue
        for _, row in sub.iterrows():
            path = Path(row["source_path"])
            inspection = inspect_ims(path)
            arrays = read_channel_arrays_ims(
                path, inspection, mode="3d", resolution_level=ANALYSIS_RESOLUTION_LEVEL
            )
            idx = channel_index_map(row["channel_map"])
            px = float(row["pixel_size_um_level0"])
            focus_idx = int(row["focus_peak_idx"])

            dapi = arrays[idx["DAPI_405"]]
            labels_2d, _ = nucleus_labels(dapi, focus_idx, px)
            labels_2d = segmentation.clear_border(labels_2d)
            labels_2d = measure.label(labels_2d > 0).astype(np.int32)
            if labels_2d.max() == 0:
                continue

            lo = max(focus_idx - 2, 0)
            dapi_2d = dapi[lo : focus_idx + 3].astype(np.float32).mean(0)
            nucleolus_mask, records = find_nucleoli(dapi_2d, labels_2d, px)

            # Signal inside vs outside the nucleolus mask, per channel.
            nuc_mask = labels_2d > 0
            nucleoplasm = nuc_mask & ~nucleolus_mask
            tile_rec = {
                "round": rnd,
                "tile": row["tile"],
                "cell_line": row.get("cell_line"),
                "treatment": row.get("treatment"),
                "has_primary": row.get("has_primary"),
                "n_nuclei": int(labels_2d.max()),
                "n_nucleoli": len(records),
                "nucleoli_per_nucleus": round(len(records) / max(labels_2d.max(), 1), 2),
            }
            for name in ("RFP_561", "GFP_488"):
                plane = arrays[idx[name]][lo : focus_idx + 3].astype(np.float32).mean(0)
                if nucleolus_mask.any() and nucleoplasm.any():
                    inside = float(np.median(plane[nucleolus_mask]))
                    outside = float(np.median(plane[nucleoplasm]))
                    tile_rec[f"{name}_nucleolar"] = round(inside, 1)
                    tile_rec[f"{name}_nucleoplasm"] = round(outside, 1)
                    tile_rec[f"{name}_enrichment"] = round(inside / outside, 3) if outside else np.nan
            tile_rows.append(tile_rec)
            for rec in records:
                rec.update({"round": rnd, "tile": row["tile"], "cell_line": row.get("cell_line")})
            obj_rows.extend(records)
            print(
                f"  {rnd} {row['tile'][-10:]}: {tile_rec['n_nuclei']} nuclei, "
                f"{tile_rec['n_nucleoli']} nucleoli "
                f"({tile_rec['nucleoli_per_nucleus']}/nucleus), "
                f"561 enrich={tile_rec.get('RFP_561_enrichment')}, "
                f"488 enrich={tile_rec.get('GFP_488_enrichment')}",
                flush=True,
            )
            arrays.clear()

    if not obj_rows:
        print("no nucleoli found in any tile")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    objs = pd.DataFrame(obj_rows)
    tiles = pd.DataFrame(tile_rows)
    objs.to_csv(args.out / "nucleolus_objects.csv", index=False)
    tiles.to_csv(args.out / "nucleolus_per_tile.csv", index=False)

    print()
    print("=== RELIABILITY CHECK (criteria fixed before running) ===")
    d = objs.nucleolus_diam_um
    print(f"  diameter um      median={d.median():.2f}  IQR={d.quantile(.25):.2f}-{d.quantile(.75):.2f}"
          f"  (expect ~0.5-3.5)")
    npn = objs.groupby(["round", "tile", "nucleus_label"]).size()
    print(f"  nucleoli/nucleus median={npn.median():.1f}  IQR={npn.quantile(.25):.1f}-{npn.quantile(.75):.1f}"
          f"  (expect 1-5)")
    af = objs.drop_duplicates(["round", "tile", "nucleus_label"]).nucleolar_area_fraction
    print(f"  nucleolar area fraction median={af.median():.3f}  (expect ~0.03-0.20)")
    c = objs.dapi_contrast
    print(f"  DAPI contrast (nucleoplasm/nucleolus) median={c.median():.2f}  (expect >1.2)")
    print()
    print("=== PER-TILE ENRICHMENT ===")
    cols = ["round", "tile", "cell_line", "n_nuclei", "nucleoli_per_nucleus",
            "RFP_561_enrichment", "GFP_488_enrichment"]
    print(tiles[[c for c in cols if c in tiles]].to_string(index=False))
    print()
    print(f"wrote {args.out}/nucleolus_objects.csv and nucleolus_per_tile.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
