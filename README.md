# CZI Nuclear Foci Analysis

Config-driven local analysis for 3-channel ZEISS CZI images with:

- nuclear segmentation from one channel;
- independent segmentation of two nuclear DNA-damage focus channels;
- co-localisation calls between the two focus channels inside nuclei;
- mask TIFFs and channel-specific QC PNGs for manual review;
- per-image, per-nucleus, per-focus and co-localised-pair CSV outputs;
- sensitivity, specificity and selectivity tables for simple per-nucleus count cut-offs.

The package is designed for local research workflows. It does not modify raw CZI
files and does not upload image data.

## Install

> **Do not `pip install -e .` without constraints if you intend to run an
> analysis.** `pyproject.toml` declares its runtime dependencies without upper
> bounds, so a plain install resolves to whatever is current that day. Newer
> scikit-image changes `peak_local_max`, `watershed` and `threshold_otsu`
> behaviour, and segmentation output changes with it — silently. Nothing errors;
> the numbers just stop matching every previously published figure.

From this directory:

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[analysis,test]' -c constraints-analysis.txt
```

**Both halves of that command are needed.** A constraints file only bounds packages
that something else has already selected; `-c` on a package nobody requested is
silently inert. The earlier documented command omitted the `analysis` extra and so
installed **no matplotlib at all** — neither the base dependencies nor the `test`
extra requested it, and the constraint on it did nothing. The `analysis` extra does
the selecting; the constraints file bounds the transitive closure too.

The versions, pinned by the `analysis` extra and recorded in
[`constraints-analysis.txt`](constraints-analysis.txt):

| package | version | where |
|---|---|---|
| python | 3.11.4 | interpreter — see the caveat below |
| numpy | 1.26.2 | `analysis` |
| scipy | 1.13.1 | `analysis` |
| scikit-image | **0.24.0** | `analysis` |
| pandas | 2.1.4 | `analysis` |
| matplotlib | 3.8.1 | `analysis` |
| Pillow | 11.0.0 | `analysis` |
| reportlab | 4.0.8 | `analysis` |
| h5py | 3.16.0 | `analysis` — `.ims` (Imaris) reading |
| aicspylibczi | 4.0.1 | `test` |

Two things this still does not pin:

- **The interpreter.** `python3 -m venv` builds against whatever `python3` invoked
  it; it cannot give you 3.11.4 on request. Check `python -V` after activating, or
  build the environment with conda, which can pin the interpreter.
- **The full transitive closure.** `constraints-analysis.txt` lists direct
  dependencies, not every package in the environment. It is a pin set, not a lock
  file.

`tests/test_environment_pins.py` fails if `pyproject.toml` and
`constraints-analysis.txt` drift apart, or if a declared runtime dependency is left
out of the `analysis` extra — the shape of the matplotlib bug above.

The `test` extra pulls in `aicspylibczi`, which the real-file half of
`tests/test_czi_reader_offsets.py` needs to cross-check this package's CZI
decoding against an independent implementation of the format. That comparison is
what caught the `data_offset` bug; without the extra installed it cannot run,
and the suite can go green having never checked a real pixel.

The package tests use the standard library `unittest`, so no separate test
runner is required:

```zsh
env CZI_TEST_MANIFEST=/path/to/sample_sheet.csv PYTHONPATH=src \
    python -m unittest discover -s tests -t tests
```

Setting `CZI_TEST_MANIFEST` is an explicit request for the real-file comparison,
so it **fails** rather than skips if `aicspylibczi` is missing or the manifest
resolves to no readable file. It accepts a directory of `.czi`, a single `.czi`,
or a CSV with a `source_path` column. Setting it also **disables the
`sample_sheet.csv` fallback** — a manifest naming files that cannot be resolved
must fail rather than quietly compare some other dataset. Leave it unset and the
repo's own `sample_sheet.csv` is used; with neither, the test skips, so a checkout
without the acquisition volume still runs green.

Reading `.ims` (Imaris) data needs `h5py`, now pinned in the `analysis` extra. It
was previously undeclared, so the documented install left the 3-D scripts under
`scripts/` expecting a package nothing supplied.

## Run A New 3-Channel Experiment

1. Copy and edit the example config:

```zsh
cp configs/three_channel_dna_damage.example.json configs/my_experiment.json
cp configs/example_condition_map.csv configs/my_condition_map.csv
```

2. Set the channel definitions in the config:

- `nuclei.channel_index`: nuclear marker, usually DAPI.
- `focus_a.channel_index`: first focus channel.
- `focus_b.channel_index`: second focus channel.
- `focus_a.name` and `focus_b.name`: short output-safe names, for example
  `AF488`, `gH2AX`, `53BP1` or `RAD51`.
- `colocalization_dilation_px`: pixel tolerance for overlap. A value of `1`
  means a focus is co-localised if it intersects the other focus after 1-pixel
  dilation.
- `control_conditions`: conditions treated as negative controls for threshold
  sensitivity/specificity calculations.
- `condition_map_csv`: optional CSV mapping filename codes to biological
  condition names. It must contain `condition_code` and `condition` columns.
  Use a path relative to the config file.

3. Run a one-file QC pass:

```zsh
czi-foci analyse \
  --root "/Volumes/Backup/TS runs/Time Series V (24+48hr)" \
  --config configs/three_channel_dna_damage.example.json \
  --output-dir output/generic_qc_test \
  --limit 1 \
  --qc-title "Generic CZI Foci QC"
```

4. Inspect the output masks and QC overlays. If segmentation is acceptable,
run the full folder:

```zsh
czi-foci analyse \
  --root "/Volumes/Backup/TS runs/Time Series V (24+48hr)" \
  --config configs/three_channel_dna_damage.example.json \
  --output-dir output/generic_time_series_V \
  --qc-title "Time Series V Generic Nuclear Foci QC"
```

Use `--overwrite` only when you intentionally want to replace an existing output
directory.

## Output Files

Each run writes:

- `resolved_config.json`: exact config used for the run.
- `image_summary.csv`: field-level counts and CZI channel metadata.
- `nucleus_measurements.csv`: nucleus area, intensities and per-nucleus focus counts.
- `<focus_a.name>_measurements.csv`: per-focus measurements for focus A.
- `<focus_b.name>_measurements.csv`: per-focus measurements for focus B.
- `colocalized_focus_pairs.csv`: focus-pair calls.
- `condition_summary.csv`: condition/timepoint summary.
- `threshold_performance_<metric>.csv`: sensitivity, specificity and selectivity
  across candidate thresholds.
- `best_thresholds.csv`: best threshold by Youden J, then specificity, then
  sensitivity.
- `masks/*.tif`: nuclei, focus A, focus B and co-localised focus masks.
- `qc_overlays/*.png`: composite QC overlays.
- `qc_channels/*.png`: single-channel QC views.
- `qc_contact_sheet.pdf`: review PDF of composite overlays.

## Batch-Aware Thresholds

After one or more segmentation runs, use `batch-summary` to avoid blindly
applying one pooled cut-off across experiments.

```zsh
czi-foci batch-summary \
  --nucleus-csv output/generic_time_series_III/nucleus_measurements.csv \
  --nucleus-csv output/generic_time_series_IV/nucleus_measurements.csv \
  --nucleus-csv output/generic_time_series_V/nucleus_measurements.csv \
  --metric focus_a_count \
  --metric focus_b_count \
  --metric colocalized_focus_b_count \
  --control-condition Aqueous \
  --control-condition DMSO \
  --control-condition NAH2PO4 \
  --control-condition Media \
  --output-dir output/batch_aware_summary
```

By default, batches are `experiment + timepoint_hr` when those columns are
present. You can override this with repeated `--batch-column` arguments.

This command writes:

- `control_stats.csv`: control median and robust standard deviation within each
  batch.
- `normalised_nuclei.csv`: per-nucleus metrics with control-normalised z-scores.
  Formula: `(nucleus value - batch control median) / batch control robust SD`.
- `raw_best_thresholds_by_batch.csv`: best raw count cut-off per batch.
- `normalised_best_thresholds_by_batch.csv`: best control-normalised cut-off per
  batch.
- `raw_threshold_stability.csv` and `normalised_threshold_stability.csv`: whether
  the same best cut-off is stable across batches.
- `effect_sizes.csv`: experiment-normalised condition effects. Use this for
  biological interpretation when cut-offs are not stable across batches.

For the existing TS III/IV/V dataset, use the mapped combined tables and
experiment-only batches:

```zsh
czi-foci current-dataset-report \
  --output-dir output/segmentation/batch_aware_current_dataset \
  --overwrite
```

This writes experiment-specific raw and control-normalised thresholds, nucleus
classifications, field and condition summaries, and control-only QC PDFs.

## Current Limitations

- The bundled parser supports uncompressed Gray8 and Gray16 CZI planes. If a CZI
  uses JPEG XR, mosaics, pyramids or other compressed variants, use a validated
  CZI reader before analysis.
- Threshold performance treats non-control conditions as positive examples. This
  is suitable for screening-level separation from background, not for proving
  biological mechanism.
- Cross-experiment comparisons remain vulnerable to staining, acquisition and
  batch effects unless normalised to a confirmed control strategy.

## Verify

```zsh
python3 -m py_compile src/czi_foci/*.py
PYTHONPATH=src python3 -m unittest discover -s tests
```
