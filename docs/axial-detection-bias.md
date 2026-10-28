# Axial detection bias in the TRF2 stacks

Measured 2026-10-05 on 15 fields across rounds `_4`, `_6`, `_7`, `_11`, `_12`
(raw, ResolutionLevel 0, DoG + 5x MAD inside the DAPI mask).

Two separate problems, both affecting any per-nucleus foci count or
colocalisation rate from this dataset. Neither is fixed by better analysis.

## 1. Detections are not axially centred on the nuclei

`z_rel` = mean plane index of detections, 0 at the first plane, 1 at the last.

| round | Z | 561 | 488 | DAPI mask centre |
|---|---|---|---|---|
| `_4` | 14 | 0.55-0.66 | 0.53-0.60 | 0.50-0.52 |
| `_6` | 10 | 0.72-0.92 | 0.68-0.81 | 0.51-0.52 |
| `_7` | 12 | **0.11-0.49** | **0.04-0.22** | 0.48-0.49 |
| `_11` | 17 | 0.53-0.56 | 0.49-0.53 | 0.50-0.51 |
| `_12` | 18 | 0.72-0.83 | 0.55-0.66 | 0.47-0.51 |

**The nuclei are centred in every stack (mask centre 0.47-0.52) but the
detections are not.** They concentrate near the top of the stack in `_6` and
`_12`, near the bottom in `_7`, and mid-stack in `_4` and `_11`.

Foci are being found preferentially wherever that acquisition happened to be
sharpest, and the sharp plane differs between acquisitions. Combined with the
poor axial containment that stage 0 reports (0% of tiles axially contained in
`_1` and `_8`), a per-nucleus foci count from these stacks is substantially a
statement about where the focal plane sat, not about the nucleus.

## 2. 488 detections sit systematically shallower than 561

The 488-minus-561 difference in `z_rel` is **negative in all 15 fields**, with
no exceptions: range -0.018 to -0.277, median about -0.065, which at these
stack depths is roughly -0.45 um.

Independently, matching peaks in XY only and reading off the axial displacement
gives a pooled mean dz of **-0.350 um** (95% CI -0.436 to -0.263, n = 1350
pairs, median 0.000, sd 1.617) — significantly non-zero and the same sign.

### Two candidate causes, confounded in this data

- **Axial chromatic shift** — optical, constant across acquisitions.
- **Depth-dependent attenuation or spherical aberration** — 488 is scattered
  and absorbed more strongly than 561 in tissue, so 488 is recovered less well
  with depth. Sample-dependent, varies with how deep the core was imaged.

The per-round pattern argues against a purely optical cause: the offset is
-0.02 to -0.06 um in `_7` but -0.75 to -0.89 um in `_12`, measured on the same
objective with the same filters on the same afternoon. Chromatic aberration
would be identical in both.

### Why this cannot be resolved from the existing data

A bead is the only object with no biology to attenuate, so any 488/561 axial
difference measured on a bead is instrumental by construction. A **bead-only
slide imaged as a depth series** (coverslip plane and 5-15 um into the mount,
dz 0.1 um, all three channels) separates the two causes. Bead-coated coverslips
alone do not — they characterise the one depth at which there are no cells.

Note also that the existing z-step (0.38-0.74 um) is too coarse for the stacks
to measure their own registration error: any offset below one z-step is
invisible, and 100-400 nm is exactly the range that matters against a 0.5 um
colocalisation criterion.

## Consequences for the analysis

1. **Colocalisation is depth-biased within every stack.** If 488 recovery falls
   off with depth faster than 561, apparent colocalisation is higher near the
   coverslip and lower deeper in. The randomisation control does not correct
   this: the null shifts peaks randomly while the bias is systematic.
2. **Report foci counts per unit *imaged* nuclear volume**, not per nucleus,
   or restrict to nuclei fully contained in the stack. Given stage 0's
   containment fractions the second option discards most of the data.
3. **Do not compare foci counts across rounds** without accounting for the
   focal-plane position. `_7` and `_6` differ more in where their detections
   sit than any plausible biological effect.
4. **Use `volume_3d`, not `mip_2d_propagate`.** The extruded-cylinder mask
   spans the whole stack, so it cannot exclude anything by depth — including
   coverslip beads sitting beneath a nucleus, which would be counted as
   colocalised foci in both channels.

## What to record at the bench next time

The depth of each stack within the section. If the effect is attenuation rather
than chromatic, no instrument characterisation fixes it retrospectively — the
analysis has to model depth, and that is only possible if depth was written down.
