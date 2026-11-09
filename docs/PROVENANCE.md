# Provenance

What in this repository is reproducible, what is not, and why. Mirrors
`docs/PROVENANCE.md` in the 2-D repository (`oncoapop/czi-nuclear-foci-2d`).

## Environment — do not create a fresh venv

Use `/Users/dyap/miniconda3/bin/python3` with `PYTHONPATH=src`.

| package | pinned version |
|---|---|
| python | 3.11.4 |
| numpy | 1.26.2 |
| scipy | 1.13.1 |
| scikit-image | **0.24.0** |
| pandas | 2.1.4 |
| matplotlib | 3.8.2 |
| h5py | 3.16.0 |

Newer scikit-image changes `peak_local_max`, `watershed` and `threshold_otsu`
behaviour, and segmentation output changes with it. A fresh `pip install -e .`
will not reproduce published results.

## The TSIV stage 1-3 pipeline is LOST, and its outputs are unreproducible

`/Volumes/Backup/Andor imaging/TSIV 2 hr/_analysis/` contains outputs from a
four-stage pipeline (stage 0-3 plus a micronuclei pass and an algorithm
comparison harness), written 2026-08-12 and 2026-08-13. **No code for stages
1-3 exists anywhere.**

### Timeline

| date | event |
|---|---|
| 2026-08-12/13 | Pipeline written to a session scratchpad at `/private/tmp/.../scratchpad/tsiv_pipeline/` and run. 16 files, ~2,043 lines. Never committed. |
| 2026-08-14 | Project note records the risk explicitly: *"scratchpad, not durable; if this work continues in a future session, the code will need to be re-created or relocated to a durable path."* |
| 2026-09-15 | Audit locates the full source in the originating session transcript (`~/.claude/projects/-Users-dyap-Documents-Codex/2fadbf06-7527-4933-9ea0-638d73739702.jsonl`) and confirms it replays cleanly. Recovery is possible at this point but is not performed. |
| 2026-10-02 | **That transcript no longer exists.** The scratchpad is long gone, no Time Machine destination is configured, and no other transcript, branch, stash, reflog, dangling object, shell history or backup contains the code. |

The recovery window closed between 15 September and 2 October. This is the
third instance of the same failure mode in this project — the 2-D pipeline's
published results also came from gitignored, untracked scripts.

### What survives

- The **outputs** themselves (CSVs and PNGs) — see `DO_NOT_USE.md` beside them.
- The **column headers**, which fingerprint the schema.
- The **project note**: `~/Documents/Codex/projects/tsiv-2hr-andor-nuclear-segmentation.md`
  (2026-08-14). Primary source. Records the algorithm decision, the parameters
  rationale, and the known defects.
- A **secondary description** of the 16 files and their output mapping, in this
  repository's audit record. Not the source.

### Why the outputs must not be built on, independent of the code loss

Even with the code, these numbers would not be usable:

1. **One tile per round.** `stage1_per_tile.csv` has 11 rows against 25 tiles
   per round — roughly 4% of the data, mostly tile `F12`, with no recorded
   selection rule. The project note confirms the intent: *"one representative
   tile per round"*.
2. **Raw and deconvolved mixed in one table.** 9 of those 11 tiles are
   deconvolved and 2 are raw, side by side with no covariate. Rounds `_1` and
   `_6` are raw only because their deconvolved folders are empty — the
   raw/deconvolved split was decided by which files happened to exist. Measured
   on paired identical fields, deconvolution changes foci counts by ×1.17
   (561) to ×1.51 (488) in 3-D, so these rows are not comparable.
3. **Stages 2 and 3 are stale.** Their outputs are timestamped 2026-08-12
   15:19-15:22; `segmentation.py` was last edited 2026-08-13 23:39. The project
   note states plainly: *"Stages 2/3 have NOT been re-validated against the
   current segmentation algorithm revision."*
4. **No interpreter was pinned.** Scripts were invoked as bare `python3`.
5. **No condition legend.** The project note: *"No treatment/well legend exists
   for the 12 rounds ... currently treated as unlabeled replicates of a single
   2 hr condition."*

### Consequence for this repository

Stage 1-3 for the `.ims` data must be **written fresh**, not ported. Nothing in
`TSIV 2 hr/_analysis/` is a reference implementation or a regression baseline.
The one planned verification that depended on recovery — re-running the legacy
code on its original 11 tiles and diffing against the orphaned CSVs — is no
longer possible and has been dropped.

## Rules this repository now enforces

- Analysis code lives in `scripts/` or `src/`, committed, never in a scratchpad.
- Every output row records `source_path`, `resolution_level` and the resolved
  channel names, so a CSV can always be traced back to the pixels it came from.

## Known-fixed defects

Four defects, all found by diffing outputs against a previous run or an
independent survey. **None of them raised an error or changed an exit code** —
each produced a complete set of plausible-looking output files. That is the
failure mode to expect from this code: not crashes, but confident wrong numbers.

| defect | how it showed | status |
|---|---|---|
| `pixel_size_um_ims` ignored `resolution_level`, reporting level-0 spacing for level-N pixels (4x too small at level 2) | only by comparing the recorded spacing against the level actually read | **fixed**; per-level dimensions are read from the file, with a fallback to the halving rule. Tests in `tests/test_ims_reader.py`. |
| Channel identity assumed positionally | would have segmented nuclei out of the TRF2 channel under the `.czi`-ordered config | **fixed**; `resolve_channel_indices()` resolves by recorded name and raises on an unexpected channel set. DAPI is channel **1** here, not 0 or 2. |
| `derive_round` derived labels from `os.path.commonprefix` over every folder in the root | creating `_analysis_v2` meant the old `_analysis` directory entered the folder list, collapsing the shared prefix and relabelling all 14 rounds to their full folder names; a join against the previous run matched **zero** rows | **fixed**; the label is parsed from a single folder's own name. `discover_tiles` also skips directories starting with `.` or `_` and any containing no `.ims`. Tests in `tests/test_stage0_round_labels.py`. |
| `QC_RESOLUTION_LEVEL = 2` requested unconditionally | the single-plane TSIV files have `ResolutionLevel 0` only, so **500 of 501 tiles failed** while the script still exited 0 and wrote all three outputs | **fixed**; `choose_resolution_level()` falls back to the deepest level present, records both the level used and the level requested, and rescales the nucleus-size constants (lengths by the level ratio, areas by its square) so a fallback does not silently change nucleus counts. |

### Regenerated outputs

`_analysis_v2/` in both datasets now holds stage 0 run with all four fixes:

| dataset | tiles | read_ok | level used | pixel_size_um |
|---|---|---|---|---|
| TRF2 | 214 | 189 | 2 | 0.402 (level-0: 0.1005) |
| TSIV 2 hr | 501 | 458 | 0 (457 tiles), 2 (1 stitched mosaic) | 0.1005 / 0.4021 |

The TSIV read count matches an independent HDF5 metadata survey exactly.

`TRF2/_analysis/stage0_tile_qc.csv` (2026-09-10) predates all of this. Its
`pixel_size_um` column reads 0.1005 for data read at level 2, where the true
spacing is 0.402. **Superseded by `_analysis_v2/`; do not use.**
