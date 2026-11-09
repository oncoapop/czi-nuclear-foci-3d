# Phase 0: is Andor Fusion's deconvolution worth keeping?

Run 2026-10-02 by `scripts/phase0_deconvolution_value.py`.
Outputs: `<dataset>/_analysis_v2/phase0_detection_adjudication.csv` and `phase0_fwhm.csv`.

**Recommendation: analyse raw. Stop archiving the TSIV deconvolved copies. Keep
the single TRF2 deconvolved folder as a sensitivity arm, not as primary data.**

## The question, correctly framed

Deconvolution ran in real time inside Fusion during acquisition — the raw and
deconvolved `TimeBegin` stamps differ by 0.55 s for a single plane and 3.75 s
for a 10-plane stack, about 0.4 s/plane on GPU, written as a parallel output
stream. **Time cost is effectively zero.** The cost is storage: a second full
copy, 18.6 GB of the 111 GB archive (12.6% of TRF2, 52.3% of TSIV).

So the question is whether it adds information worth 2x disk and a second file
lineage, not whether it is worth waiting for.

## Scope and what limits it

| arm | operation | pairs measured |
|---|---|---|
| TRF2 `_1` (Z=10, dz 0.38 um) | genuine 3-D deconvolution | 25 tiles, **one core only** |
| TSIV x9 (Z=1) | single-plane; no adjacent planes to deblur | 72 tiles, 9 cores |

The evidence base is inverted relative to what we would want: the arm where
deconvolution could genuinely help has n=1 core.

Two optical facts bound every conclusion below:

- **This is widefield**, not spinning disk (`MicroscopeMode = Widefield`,
  `LSMPinhole = 0`). Maximum out-of-focus light to remove, but also a missing cone.
- **Z is sampled below Nyquist in every acquisition.** At 60x/NA 1.2 water
  (n=1.33) axial Nyquist is 292 nm (405), 344 nm (488), 393 nm (561); actual dz
  is 380 nm here and 500-740 nm elsewhere in TRF2. Deconvolution cannot restore
  axial frequencies that were never sampled.

**No bead slide was acquired**, so resolution is measured relative to the raw
image, not against a known PSF. A bead slide would make this absolute.

## Result 1 — deconvolution changes detection counts substantially, unevenly

Same detector, same shared nuclear mask from the raw DAPI, only the image varies.

| arm | channel | foci deconv/raw (median, IQR) |
|---|---|---|
| TRF2 3-D | 488 | **1.64** (1.58-1.92) |
| TRF2 3-D | 561 | **1.17** (1.08-1.25) |
| TSIV 2-D | 488 | 0.96 (0.87-1.19) |
| TSIV 2-D | 561 | 1.07 (0.88-1.34) |

In 3-D it inflates the dim 488 channel by 64%. In single-plane data it does
essentially nothing to counts — both IQRs straddle 1.0.

## Result 2 — it is not fabricating foci in empty background

Every detection present in only one image was adjudicated against the other
image's pixel values, relative to an empirical null (the DoG response at random
positions inside the same nuclear mask).

| arm | channel | added | sitting on real raw signal | not in raw |
|---|---|---|---|---|
| TRF2 3-D | 488 | 16,807 | 16,802 (100.0%) | 5 |
| TRF2 3-D | 561 | 13,783 | 13,075 (94.9%) | 708 (5.1%) |
| TSIV 2-D | 488 | 1,905 | 1,905 (100%) | 0 |
| TSIV 2-D | 561 | 2,966 | 2,791 (94.1%) | 175 (5.9%) |

**This rules out gross fabrication, and nothing more.** "Elevated raw signal
just below threshold" describes a genuine dim focus and a noise fluctuation
equally well. Deconvolution amplifies peaks, so the detections it adds are by
construction the ones that were nearly at threshold already. Without a
replicate acquisition or a bead slide this test cannot separate the two.

## Result 3 — the resolution measurement, which is decisive

FWHM of isolated puncta, measured in a bounded +/-1.5 um window; nan when the
profile never reaches half maximum there.

| arm | channel | lateral raw -> deconv | axial raw -> deconv | axially measurable |
|---|---|---|---|---|
| TRF2 3-D | 561 | 0.492 -> **0.382 um** (-22%) | 1.192 -> 1.055 um (-11%) | 18-21% |
| TRF2 3-D | 488 | 0.619 -> 0.569 um (-8%) | 0.929 -> **1.195 um (+29%)** | 27-37% |
| TSIV 2-D | 561 | 0.952 -> **0.673 um** (-29%) | n/a | n/a |
| TSIV 2-D | 488 | 0.847 -> **0.608 um** (-28%) | n/a | n/a |

Theory for this objective: lateral 252 nm (561) / 221 nm (488); axial 971 / 852 nm.

Three things follow.

1. **Lateral sharpening is real and consistent** — 8-29% narrowing everywhere,
   largest in the single-plane data.
2. **Axial restoration, the entire justification for 3-D deconvolution, does not
   happen.** 561 improves 11%; **488 gets 29% worse**. This is what below-Nyquist
   Z sampling predicts: at 0.38 um across a ~1.2 um axial PSF there are about
   three samples to work with.
3. **Only 18-37% of puncta reach half maximum in Z at all**, which is itself a
   direct symptom of axial undersampling.

The combination in 488 is the argument: deconvolution **adds 64% more detections
while making the axial PSF 29% worse**. A process that degrades resolution
cannot be resolving more foci; it is lowering the effective detection threshold.

## A test that did not work, recorded so it is not repeated

The ringing test — are added detections closer to existing foci than chance? —
returned ratios of 0.26-0.58 across both arms, which looks like a strong halo
signature. **It is not interpretable as one.** The null is uniform-in-mask, but
real foci are spatially clustered, so genuine dim foci would also sit closer to
bright ones than a uniform draw. The test cannot distinguish ringing from
clustering and should be rebuilt against a clustered null, or dropped.

## Recommendation

**TSIV (2-D arm): do not keep the deconvolved copies.** Foci counts are
unchanged (0.96x, 1.07x, IQRs spanning 1.0), no axial information exists to
recover, and the lateral sharpening changes no conclusion. That is 6.1 GB,
52% of the dataset, for no analytical gain. Delete after confirming the raw set
is intact, or stop generating them.

**TRF2 (3-D arm): analyse raw.** Deconvolution inflates 488 counts 64% while
degrading the 488 axial PSF, on Z-sampling that cannot support axial
restoration. Keep `_1_deconvolved` (12.5 GB) as the sensitivity arm — it is the
only true 3-D pair and will be needed to report robustness — but it is not the
primary data.

**Never pool or compare raw with deconvolved.** The effect is large (1.17x to
1.64x), channel-specific, and opposite in direction between the 3-D and
single-plane arms.

**If deconvolution is wanted in future, fix the acquisition first:** dz <= 0.3 um,
and a bead slide to measure the PSF and the 488/561 chromatic registration. At
the current Z sampling the algorithm is being asked to restore information the
microscope never recorded.

## Caveats

- The 3-D conclusion rests on **25 tiles from one core**. Tiles are independent
  optical fields, which is adequate for an optical question, but they are one
  biological sample and one staining round.
- No bead slide: resolution is relative, not absolute.
- The detection adjudication rules out fabrication, not threshold-boundary noise
  promotion.
