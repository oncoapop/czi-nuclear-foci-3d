"""Final summary figure for the TRF2 x gH2AX / BRCA1 imaging analysis.

Four panels, in the order the argument runs:
  A  what the counts look like, and how wide they are
  B  whether the dispersion result survives the detection threshold
  C  whether the between-line count difference is telomeres or nuclear size
  D  whether colocalization can be measured at all

Reads only the per-nucleus CSV, so it is cheap to re-run after any change
upstream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PRIMARY_K = 6.0
LINE_ORDER = ["WT_hTERT", "p53-/-BRCA1-/-"]
TREAT_ORDER = ["Palbociclib", "untreated"]
COLORS = {"WT_hTERT": "#2C5F2D", "p53-/-BRCA1-/-": "#B85042"}
HATCH = {"Palbociclib": "", "untreated": "///"}


def _groups(df):
    for line in LINE_ORDER:
        for treat in TREAT_ORDER:
            g = df[(df["cell_line"] == line) & (df["treatment"] == treat)]
            if len(g):
                yield line, treat, g


def panel_a(ax, nuclei):
    at_k = nuclei[(nuclei["detect_k"] == PRIMARY_K) & (nuclei["has_primary"] == True)]  # noqa: E712
    data, labels, colors, hatches, stats = [], [], [], [], []
    for line, treat, g in _groups(at_k):
        counts = g["n_foci_RFP_561"].to_numpy()
        data.append(counts)
        labels.append(f"{'WT' if line.startswith('WT') else 'BRCA1-/-'}\n{treat}")
        colors.append(COLORS[line])
        hatches.append(HATCH[treat])
        stats.append((counts.mean(), counts.std() / counts.mean(), counts.var() / counts.mean()))
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, color, hatch in zip(parts["bodies"], colors, hatches):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_hatch(hatch)
    for i, counts in enumerate(data, start=1):
        q1, med, q3 = np.percentile(counts, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", lw=3)
        ax.plot(i, med, "o", color="white", ms=4, zorder=3)
    top = max(np.percentile(d, 99) for d in data)
    for i, (mean, cv, fano) in enumerate(stats, start=1):
        ax.text(i, top * 1.04, f"CV {cv:.2f}\nFano {fano:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("TRF2 foci per nucleus (1.5 µm slab)", fontsize=8)
    ax.set_title(f"A  Telomere-count distributions, k={PRIMARY_K:.0f}", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)


def panel_b(ax, nuclei):
    stained = nuclei[nuclei["has_primary"] == True]  # noqa: E712
    ks = sorted(stained["detect_k"].unique())
    for line in LINE_ORDER:
        for treat in TREAT_ORDER:
            fanos = []
            for k in ks:
                g = stained[
                    (stained["detect_k"] == k)
                    & (stained["cell_line"] == line)
                    & (stained["treatment"] == treat)
                ]["n_foci_RFP_561"]
                fanos.append(g.var() / g.mean() if len(g) and g.mean() > 0 else np.nan)
            ax.plot(ks, fanos, marker="o" if treat == "Palbociclib" else "s",
                    ls="-" if treat == "Palbociclib" else "--",
                    color=COLORS[line], ms=4, lw=1.4,
                    label=f"{'WT' if line.startswith('WT') else 'BRCA1-/-'} · {treat}")
    ax.set_xlabel("detection threshold k (σ above noise)", fontsize=8)
    ax.set_ylabel("Fano factor (variance / mean)", fontsize=8)
    ax.set_title("B  Overdispersion is threshold-robust", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax.tick_params(labelsize=7)


def panel_c(ax, nuclei):
    at_k = nuclei[(nuclei["detect_k"] == PRIMARY_K) & (nuclei["has_primary"] == True)]  # noqa: E712
    lines, per_nuc, per_vol, areas = [], [], [], []
    for line in LINE_ORDER:
        g = at_k[at_k["cell_line"] == line]
        lines.append("WT" if line.startswith("WT") else "BRCA1-/-")
        per_nuc.append(g["n_foci_RFP_561"].mean())
        per_vol.append(g["foci_RFP_per_um3"].mean())
        areas.append(g["nucleus_area_um2"].mean())
    x = np.arange(len(lines))
    ax.bar(x - 0.2, per_nuc, width=0.38, color=[COLORS[l] for l in LINE_ORDER], alpha=0.85,
           label="per nucleus")
    ax.set_ylabel("TRF2 foci per nucleus", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{a:.0f} µm² nuclei" for l, a in zip(lines, areas)], fontsize=7)
    twin = ax.twinx()
    twin.bar(x + 0.2, per_vol, width=0.38, color=[COLORS[l] for l in LINE_ORDER], alpha=0.4,
             hatch="///", edgecolor="black", linewidth=0.5)
    twin.set_ylabel("TRF2 foci per µm³ (size-corrected)", fontsize=8)
    twin.tick_params(labelsize=7)
    ratio_nuc = per_nuc[1] / per_nuc[0]
    ratio_vol = per_vol[1] / per_vol[0]
    ax.text(0.5, 0.94, f"per nucleus {ratio_nuc:.1f}×  →  per volume {ratio_vol:.2f}×",
            transform=ax.transAxes, ha="center", fontsize=7.5)
    ax.set_title("C  The count gap is nuclear size, not telomere number", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)


def panel_d(ax, per_sample):
    ks = sorted(k for k in per_sample["detect_k"].unique() if k >= 4)
    stained, control = [], []
    for k in ks:
        at_k = per_sample[per_sample["detect_k"] == k]
        s = at_k[at_k["has_primary"] == True]  # noqa: E712
        c = at_k[at_k["has_primary"] == False]  # noqa: E712
        stained.append(
            s["coloc_observed_total"].sum() / s["coloc_expected_total"].sum()
            if s["coloc_expected_total"].sum() > 0 else np.nan
        )
        control.append(
            c["coloc_observed_total"].sum() / c["coloc_expected_total"].sum()
            if c["coloc_expected_total"].sum() > 0 else np.nan
        )
    ax.plot(ks, control, marker="s", color="#B85042", lw=1.6, ms=5,
            label="no-antibody control wells (4, 8)")
    ax.plot(ks, stained, marker="o", color="#2C5F2D", lw=1.6, ms=5, label="stained wells")
    ax.axhline(1.0, color="grey", lw=0.9, ls=":")
    ax.text(ks[0], 1.06, "no enrichment", fontsize=6.5, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("detection threshold k (σ above noise)", fontsize=8)
    ax.set_ylabel("colocalization enrichment (obs / null)", fontsize=8)
    ax.set_title("D  Colocalization fails its own negative control", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax.tick_params(labelsize=7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("/Volumes/Backup/Andor imaging/TRF2/_analysis"))
    args = parser.parse_args()

    nuclei = pd.read_csv(args.analysis_dir / "stage45_prelim_per_nucleus.csv")
    per_sample = pd.read_csv(args.analysis_dir / "stage45_prelim_per_sample.csv")
    out = args.analysis_dir / "final_summary.pdf"

    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        fig.suptitle(
            "TRF2 × γH2AX on WT vs p53-/-BRCA1-/- hTERT — what this slide can and cannot support",
            fontsize=12, y=0.975,
        )
        grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34,
                                left=0.075, right=0.925, top=0.88, bottom=0.17)
        panel_a(fig.add_subplot(grid[0, 0]), nuclei)
        panel_b(fig.add_subplot(grid[0, 1]), nuclei)
        panel_c(fig.add_subplot(grid[1, 0]), nuclei)
        panel_d(fig.add_subplot(grid[1, 1]), per_sample)

        fig.text(
            0.075, 0.015,
            "Imaged 2026-08-07, Andor spinning disk, 1.5 µm in-focus slab per tile, 141 tiles, "
            "8,569 nuclei. TRF2 (rabbit) 1:1000 → RedX → 561/594; γH2AX (mouse) 1:100 → FITC → 488.\n"
            "Wells 4 and 8 carry no primary antibody in either channel and are the measured "
            "false-positive floor, one per cell line. Detection threshold k is set from those "
            "controls, not from an assumed telomere count:\nthese cells are aneuploid and no "
            "published figure covers this system. A  Palbociclib (G1 arrest, fixed chromosome "
            "number) narrows the distribution in 7 of 8 threshold comparisons, and BRCA1-/- is "
            "wider than WT in 7 of 8 —\nboth predicted directions. B  The 3-4× excess dispersion "
            "in BRCA1-/- holds at every threshold, and runs opposite to counting noise, which "
            "would give the higher-mean line a LOWER CV. C  The 2.3× per-nucleus\ncount gap "
            "collapses to 0.83× per unit volume, so it is nuclear size (consistent with higher "
            "ploidy in a p53-null background), not telomere number per genome. D  Colocalization "
            "is NOT measurable: wells with no\nantibody return 3-10× enrichment, exceeding every "
            "stained well, because broadband autofluorescent debris is detected in both channels "
            "at the same place — a physical coincidence no positional null corrects.",
            fontsize=6.4, va="bottom", linespacing=1.45,
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
