> PRESERVED COPY. This note is the only surviving primary record of the
> TSIV stage 1-3 pipeline, whose source code is lost (see PROVENANCE.md).
> Copied verbatim 2026-10-02 from ~/Documents/Codex/projects/.

# Project: TSIV 2 hr Andor image analysis (nuclear segmentation → foci → colocalization)

Last updated: 2026-08-13

## Goal

Analyze `/Volumes/Backup/Andor imaging/TSIV 2 hr` (widefield, Andor Fusion spinning-disk, 60x/NA1.2,
3 channels: 405/DAPI, 488/GFP, 561/RFP) with a fresh, Andor-appropriate pipeline (explicitly not a port
of the older Zeiss/Fiji "Foci Analyzer" pipeline used on `/Volumes/Backup/TS Runs/`): nuclear segmentation
on DAPI, then foci detection/counting in the other two channels, then colocalization — each stage gated on
a contact-sheet + score review before moving to the next.

## Current Status

- Stage 0 (raw contact sheet), Stage 1 (nuclear segmentation), Stage 2 (foci detection), Stage 3
  (colocalization) all built and passed an initial QC round on representative tiles.
- Nuclear segmentation has since been revisited multiple times specifically on request — see Decisions.
- Stages 2/3 have NOT been re-validated against the current (as of this note) segmentation algorithm
  revision; they were last run against an earlier version of `segment_nuclei`.

## Key Files

- Production pipeline code: `/private/tmp/.../scratchpad/tsiv_pipeline/` this session (`ims_io.py`,
  `segmentation.py`, `foci.py`, `colocalization.py`, `stage0-3_*.py`) — **scratchpad, not durable**; if this
  work continues in a future session, the code will need to be re-created or relocated to a durable path.
- Algorithm-comparison harness: `algos.py`, `score_algos.py` (7 candidate nuclear segmentation algorithms
  scored on hand-verified real tiles + a synthetic size/touching-pair stress test).
- All analysis outputs (contact sheets, ROC/score plots, CSVs): `/Volumes/Backup/Andor imaging/TSIV 2 hr/_analysis/`.

## People

- User runs drug-treatment experiments on cultured cells; nuclear size is expected to increase under
  treatment (this is the central design constraint for the segmentation algorithm — see Decisions).

## Decisions

- **Segmentation method stays as `regularity_watershed`** (distance-transform watershed, splitting gated by
  blob size relative to a reference max-nucleus-diameter, then merging back any post-split fragment that's
  too concave/irregular to be a real separate nucleus). This is what's live in `segmentation.py`.
- Explored and rejected: `saddle_gated_watershed` (gates splitting by the depth of the "saddle" between two
  candidate peaks relative to their own height, rather than by absolute size or whole-blob solidity). It
  scored better on a controlled synthetic test (88% vs ~76%) and specifically fixed a case regularity_watershed
  missed (a genuinely-touching pair with higher solidity than an enlarged single nucleus — i.e., neither size
  nor shape alone reliably separates "one big nucleus" from "two touching nuclei"). But on a real ambiguous
  dense nucleus cluster in round_2, it split further than `regularity_watershed` already did, and the user's
  own visual judgment on that specific case matched `regularity_watershed`'s (more conservative) call. Decided
  2026-08-13 to keep `regularity_watershed` in production on that basis.
- **Why nuclear size matters here**: confirmed empirically that this dataset's own nuclei already reach
  ~39 µm equivalent diameter (p95 ~24 µm) even pre-treatment, and that solidity/size cannot cleanly separate
  "one enlarged nucleus" from "two touching nuclei" (measured: touching pair solidity 0.92 > single nucleus
  enlarged 2-3x solidity 0.81-0.89). Any future retuning of size thresholds for treated conditions should
  keep this confound in mind rather than relying on a single fixed cutoff.
- Whole-slide stitching (TeraStitcher, invoked via the `.bat`/`.xml` files Andor Fusion writes per round) can't
  be reproduced on this Mac (Windows-only binaries) — analysis works per-tile (2048x2048), using the
  deconvolved tile set where available.

## Open Questions

- No treatment/well legend exists for the 12 rounds in `TSIV 2 hr` (unnumbered + `_1`..`_10`) — currently
  treated as unlabeled replicates of a single 2 hr condition. If a legend exists, results should be re-stratified.
- Rounds `_1` (2/25 tiles) and `_6` (5/25 tiles) look like partially-aborted acquisitions.
- Stages 2 (foci) and 3 (colocalization) should be re-run against the current segmentation before trusting
  their numbers, since they were last validated against an earlier segmentation revision.

## Next Actions

- If this work continues: re-run foci/colocalization stages against `regularity_watershed` output to confirm
  they still hold, then decide whether to scale the pipeline from "one representative tile per round" to all
  25 tiles per round for a full quantification.

## Related Memory

- `~/.codex/memories/MEMORY.md` — Codex CLI's own auto-generated task log has a "Tasks workflow CZI
  microscopy analysis" entry covering the **older** Zeiss/`TS Runs` pipeline (gH2AX/53BP1,
  publication freeze, staged focus QC) that this project explicitly does not reuse — background
  only, not this pipeline's history. See `memory/operating-profile.md`'s "External Memory Stores"
  note for why that log isn't merged into this system.
- `~/Documents/Tasks workflow/czi-analysis/` — the actual code for that older pipeline, in case a
  method or threshold choice from it is worth comparing against.
