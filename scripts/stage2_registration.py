"""Stage 2: registration self-calibration via the triggered-average endpoint.

The first 3D colocalization rate on this data came out *below* chance (1.2%
vs a 1.8% z-shuffled null). Below-chance colocalization is a registration
signature, not biology. Channel-focus profiles could not resolve the axial
offset (mean -0.26 um but SD 0.75 um across 21 tiles - 488 is too dim to
localize axially), so the offset is estimated here from the data itself.

METHOD (ANALYSIS_PLAN.md "Primary endpoint")
  For each RFP_561 focus centroid, take the mean background-flattened GFP_488
  intensity in a PSF-sized lateral disc (r = 0.3 um) as a function of axial
  offset dz. Average over foci, and compare against positions permuted within
  the same nucleus. Fit

      pedestal(dz) + A * exp(-(dz - mu)^2 / (2 sigma^2))

  with sigma FIXED to the PSF width convolved with voxel integration,
  sigma = sqrt(0.38^2 + (z_step/sqrt(12))^2), leaving only A and mu free.
    mu -> the axial registration offset (what this stage delivers)
    A  -> colocalization strength (Stage 6's business, reported here as a
          precondition: mu is only interpretable where A > 0 significantly)

  The broad pedestal is real but is NOT colocalization - telomeres sit in
  chromatin, so their neighbourhoods carry more antibody, more
  autofluorescence and (decisively, in widefield) more out-of-focus 488 light
  spread over the axial PSF hourglass. Only the narrow component is the TIF
  signal, so the pedestal is modelled explicitly and only the narrow
  amplitude is reported.

WHY dz IS SAMPLED ON THE NATIVE SLICE GRID
  Focus z is kept at the integer peak slice and dz steps by whole slices. The
  resulting +-z_step/2 quantization of each focus's true axial position
  broadens the pooled profile symmetrically with variance z_step^2/12 - which
  is exactly the voxel-integration term already in the fixed sigma above - and
  being symmetric it cannot shift mu. Interpolating onto a finer dz grid would
  add its own smoothing for no gain in the one quantity this stage exists to
  measure.

WHY TWO dz WINDOWS
  The plan's +-2.5 um window needs 2.5 um of acquired stack on both sides of a
  focus. Stage 0 found the slide is tilted 4.5-7.0 um/mm, so usable slabs are
  only 1.5-5.0 um and sit at different stack depths per field. At +-2.5 um,
  well positions 5 and 6 contribute ZERO tiles - and position 6 is a
  designated control. So two windows are fitted:
    wide   +-2.5 um, quadratic pedestal - the plan's canonical procedure,
           available for positions 1,2,3,4,7,8
    narrow +-1.5 um, linear pedestal    - available for all 8 positions
  Over +-1.5 um (= +-3.7 sigma) a linear pedestal is a sound local
  approximation of a broad hourglass, and pedestal *curvature* is symmetric so
  it perturbs A, not mu; the linear term, which could shift mu, is fitted
  explicitly. Where both windows are available they must agree - that
  agreement is what licenses the narrow-window value for positions 5 and 6.

PROVISIONAL INPUTS, DELIBERATELY
  Stage 3 has not yet frozen detection parameters and Stage 4 has not yet
  produced final nuclei, but Stage 2 precedes both by design. Registration
  does not need complete detection, it needs reliable centroids, so foci are
  detected brightest-first and mu is re-fitted at several detection
  thresholds; a mu that moves with threshold would invalidate the estimate.
  Nuclei come from Stage 0's own segmentation rule (physical units), used
  only to bound foci and to define the within-nucleus permutation.

Writes stage2_registration.pdf and stops. Nothing downstream runs until the
PDF is reviewed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_ims_qc import CHANNEL_LABELS, nucleus_labels  # noqa: E402

from czi_foci.ims_reader import inspect_ims, read_channel_arrays_ims  # noqa: E402

# --- Physical constants. Every one of these is in um and is converted to
# --- voxels at runtime from the pixel size of the level actually read.
ANALYSIS_RESOLUTION_LEVEL = 0
DISC_RADIUS_UM = 0.3  # PSF-sized lateral sampling disc
PSF_SIGMA_Z_UM = 0.38  # axial PSF sigma, 60x/1.2NA widefield
PSF_ASPECT = 3.6  # axial:lateral PSF aspect, sets the DoG z kernel
DOG_SIGMA_XY_UM = 0.15
DOG_SIGMA_RATIO = 1.6  # standard difference-of-Gaussians ratio
BACKGROUND_SIGMA_UM = 2.5  # large-scale haze removed before disc sampling
MIN_FOCUS_SEPARATION_XY_UM = 0.4
MIN_FOCUS_SEPARATION_Z_UM = 1.0
DZ_MAX_WIDE_UM = 2.5
DZ_MAX_NARROW_UM = 1.5
PEDESTAL_WING_UM = 1.5  # |dz| beyond which the profile is pedestal-only
MU_BOUND_UM = 1.5  # keeps the fit off the window edge
DETECT_MAD_K = (6.0, 8.0, 12.0)  # fitted at each; mu must not move with it
PRIMARY_MAD_K = 8.0
N_PERMUTATIONS = 3  # within-nucleus null draws per focus
MIN_FOCI_PER_TILE = 30
MIN_POOLED_FOCI = 500  # per sample/window/direction, for the primary-k choice
N_BOOTSTRAP = 400
MAX_FOCI_PER_TILE = 4000  # caps memory on the densest tiles; brightest kept
EDGE_MARGIN_UM = 1.0  # keep sampling discs clear of the tile border


def fixed_sigma_um(z_step_um: float) -> float:
    """Axial PSF sigma convolved with voxel integration over one z-step.

    Fixed, never fitted: with only A and mu free the fit is far more stable,
    and this sigma is the physically correct width for on-grid sampling of an
    integer-slice focus position.
    """
    return float(np.sqrt(PSF_SIGMA_Z_UM**2 + (z_step_um / np.sqrt(12.0)) ** 2))


def disc_offsets(radius_um: float, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel offsets of a lateral disc of the given physical radius."""
    r_px = max(1, int(round(radius_um / pixel_size_um)))
    dy, dx = np.mgrid[-r_px : r_px + 1, -r_px : r_px + 1]
    inside = (dy**2 + dx**2) <= r_px**2
    return dy[inside].astype(np.int32), dx[inside].astype(np.int32)


def flatten_background(volume: np.ndarray, sigma_um: float, pixel_size_um: float) -> np.ndarray:
    """Remove slowly varying haze, plane by plane.

    The disc measurement must read a *local excess*, not the absolute level,
    or it would be dominated by the haze gradient and by any residual
    vignetting Stage 1b's illumination field failed to flatten (it fails the
    smooth-bowl gate on RFP_561 and GFP_488, so a residual is expected). The
    subtraction is applied identically to the observed and permuted samples,
    so it cannot manufacture a difference between them.
    """
    sigma_px = sigma_um / pixel_size_um
    out = np.empty(volume.shape, dtype=np.float32)
    for z in range(volume.shape[0]):
        plane = volume[z].astype(np.float32)
        out[z] = plane - ndi.gaussian_filter(plane, sigma_px)
    return out


def difference_of_gaussians(volume: np.ndarray, pixel_size_um: float, z_step_um: float) -> np.ndarray:
    """PSF-matched 3D DoG.

    sigma_z is matched to the PSF aspect ratio, not made isotropic in voxels:
    at 0.5 um z-steps a physically isotropic kernel would give sigma_z ~ 0.3
    voxels, which is aliased and a poor matched filter.
    """
    sigma_xy_px = DOG_SIGMA_XY_UM / pixel_size_um
    sigma_z_px = (DOG_SIGMA_XY_UM * PSF_ASPECT) / z_step_um
    small = ndi.gaussian_filter(volume.astype(np.float32), (sigma_z_px, sigma_xy_px, sigma_xy_px))
    large = ndi.gaussian_filter(
        volume.astype(np.float32),
        (sigma_z_px * DOG_SIGMA_RATIO, sigma_xy_px * DOG_SIGMA_RATIO, sigma_xy_px * DOG_SIGMA_RATIO),
    )
    return small - large


@dataclass
class DetectionStats:
    """Robust statistics of the in-nucleus DoG, so any k can be applied later.

    Detection runs once at the loosest k; the exact threshold for a stricter k
    is median + k * 1.4826 * MAD of this same distribution, so the stricter
    focus set is recovered exactly by filtering on the recorded amplitudes -
    no re-detection, and no quantile proxy standing in for a real threshold.
    """

    median: float
    mad_scaled: float

    def threshold(self, mad_k: float) -> float:
        return self.median + mad_k * self.mad_scaled


def detect_foci_3d(
    dog: np.ndarray,
    nuclear_mask_2d: np.ndarray,
    pixel_size_um: float,
    z_step_um: float,
    mad_k: float,
) -> tuple[np.ndarray, np.ndarray, DetectionStats]:
    """Local maxima of a DoG volume inside nuclei, above a robust threshold.

    Returns (coords, amplitudes, stats) with coords (N, 3) integer
    (z, y, x). Non-maximum suppression uses a separable rectangular
    neighbourhood sized from the physical PSF separation: an ellipsoidal
    footprint is the better shape but makes ``maximum_filter`` non-separable
    and hundreds of times slower on a 75-megavoxel voxel grid, for a
    suppression radius that is approximate either way.
    """
    inside = dog[:, nuclear_mask_2d]
    if inside.size == 0:
        return np.zeros((0, 3), dtype=int), np.zeros(0), DetectionStats(0.0, 0.0)
    median = float(np.median(inside))
    mad_scaled = float(np.median(np.abs(inside - median))) * 1.4826
    del inside
    threshold = median + mad_k * mad_scaled

    rz = max(1, int(round(MIN_FOCUS_SEPARATION_Z_UM / z_step_um)))
    rxy = max(1, int(round(MIN_FOCUS_SEPARATION_XY_UM / pixel_size_um)))
    local_max = ndi.maximum_filter(dog, size=(2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), mode="nearest")

    peaks = (dog >= local_max) & (dog > threshold)
    peaks &= nuclear_mask_2d[None, :, :]
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=int), np.zeros(0), DetectionStats(median, mad_scaled)
    amplitudes = dog[coords[:, 0], coords[:, 1], coords[:, 2]]
    return coords, amplitudes, DetectionStats(median, mad_scaled)


def refine_centroids_xy(dog: np.ndarray, coords: np.ndarray, half: int = 2) -> np.ndarray:
    """Sub-pixel lateral centroid refinement, z left on the integer slice.

    Applied only to foci that survived filtering, so the Python-level loop
    runs over hundreds of peaks rather than tens of thousands. z deliberately
    stays on its slice: the resulting quantization is the voxel-integration
    term already carried in the fixed sigma, and being symmetric it cannot
    shift mu.
    """
    centers = coords.astype(np.float64)
    if len(coords) == 0:
        return centers
    _, ny, nx = dog.shape
    for i, (z, y, x) in enumerate(coords):
        y0, y1 = max(0, y - half), min(ny, y + half + 1)
        x0, x1 = max(0, x - half), min(nx, x + half + 1)
        patch = dog[z, y0:y1, x0:x1].astype(np.float64)
        patch = np.clip(patch - patch.min(), 0, None)
        total = patch.sum()
        if total > 0:
            gy, gx = np.mgrid[y0:y1, x0:x1]
            centers[i, 1] = (patch * gy).sum() / total
            centers[i, 2] = (patch * gx).sum() / total
    return centers


def nucleus_pixel_index(labels_2d: np.ndarray) -> dict[int, np.ndarray]:
    """Map each nucleus label to its pixel coordinates in one sorting pass.

    The obvious ``np.argwhere(labels == L)`` per label rescans the whole
    4-megapixel label image once per nucleus - with ~200 nuclei and several
    permutation draws per tile that alone would dominate the stage's runtime.
    """
    flat = labels_2d.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    n_labels = int(flat.max())
    if n_labels == 0:
        return {}
    starts = np.searchsorted(sorted_labels, np.arange(1, n_labels + 2), side="left")
    width = labels_2d.shape[1]
    index = {}
    for label in range(1, n_labels + 1):
        flat_indices = order[starts[label - 1] : starts[label]]
        if flat_indices.size:
            index[label] = np.column_stack(np.divmod(flat_indices, width))
    return index


def qualifying_mask(
    centers: np.ndarray,
    dz_slices: np.ndarray,
    shape: tuple[int, int, int],
    edge_margin_px: int,
) -> np.ndarray:
    """Foci whose full dz range and whole sampling disc stay inside the tile.

    Requiring the *full* range per focus keeps the contributing focus set
    identical at every dz. Letting foci drop in and out across dz would mix
    populations and distort the profile shape - and shape is the entire
    signal this stage reads.
    """
    nz, ny, nx = shape
    z = centers[:, 0].astype(int)
    ok = (z + dz_slices.min() >= 0) & (z + dz_slices.max() <= nz - 1)
    ok &= (centers[:, 1] >= edge_margin_px) & (centers[:, 1] <= ny - 1 - edge_margin_px)
    ok &= (centers[:, 2] >= edge_margin_px) & (centers[:, 2] <= nx - 1 - edge_margin_px)
    return ok


def extract_triggered_profiles(
    volume: np.ndarray,
    centers: np.ndarray,
    dz_slices: np.ndarray,
    disc_dy: np.ndarray,
    disc_dx: np.ndarray,
) -> np.ndarray:
    """Mean intensity in a lateral disc, per focus, per axial offset.

    Returns (n_foci, n_dz).

    Offsets that fall outside the acquired stack come back as NaN rather than
    being silently clamped, so the caller can require complete coverage per
    window. Foci drop in and out of range at different dz otherwise, and a
    focus set that changes across dz mixes populations and distorts the
    profile shape - which is the whole signal.
    """
    if len(centers) == 0:
        return np.zeros((0, len(dz_slices)), dtype=np.float32)
    nz = volume.shape[0]
    z = centers[:, 0].astype(np.int32)[:, None] + dz_slices[None, :]  # (n, ndz)
    in_bounds = (z >= 0) & (z <= nz - 1)
    z = np.clip(z, 0, nz - 1)
    y = np.rint(centers[:, 1]).astype(np.int32)[:, None] + disc_dy[None, :]  # (n, ndisc)
    x = np.rint(centers[:, 2]).astype(np.int32)[:, None] + disc_dx[None, :]
    samples = volume[z[:, :, None], y[:, None, :], x[:, None, :]]  # (n, ndz, ndisc)
    profiles = samples.mean(axis=2).astype(np.float32)
    profiles[~in_bounds] = np.nan
    return profiles


def permuted_centers(
    centers: np.ndarray,
    focus_labels: np.ndarray,
    pixels_by_label: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Random lateral positions inside the same nucleus, at the SAME z.

    Preserving z is what makes this null safe: 488 brightness varies strongly
    with axial position (that is the whole point of a focus-limited slab), so
    a null that re-drew z as well would differ from the observed set in
    exactly the dimension being measured, and the difference would show up as
    profile structure that is pure sampling artifact.
    """
    out = centers.copy()
    if len(centers) == 0:
        return out
    for i, label in enumerate(focus_labels):
        pixels = pixels_by_label.get(int(label))
        if pixels is None or len(pixels) == 0:
            continue
        y, x = pixels[rng.integers(len(pixels))]
        out[i, 1] = y
        out[i, 2] = x
    return out


def _gaussian_with_pedestal(dz, c0, c1, c2, amplitude, mu, sigma):
    return c0 + c1 * dz + c2 * dz**2 + amplitude * np.exp(-((dz - mu) ** 2) / (2.0 * sigma**2))


def fit_triggered_profile(
    dz_um: np.ndarray,
    profile: np.ndarray,
    sigma_um: float,
    pedestal_order: int,
    wing_only_pedestal: bool = False,
) -> dict:
    """Fit pedestal(dz) + A exp(-(dz-mu)^2 / 2 sigma^2) with sigma fixed.

    ``pedestal_order`` is 2 for the wide window and 1 for the narrow one.
    ``wing_only_pedestal`` reproduces the plan's alternative recipe: fit the
    pedestal on |dz| > PEDESTAL_WING_UM alone, subtract, then fit the narrow
    component to the residual. Reported as a robustness column - a mu that
    depends on which pedestal recipe was used would not be trustworthy.
    """
    valid = np.isfinite(profile)
    if valid.sum() < pedestal_order + 3:
        return {"ok": False, "mu_um": np.nan, "amplitude": np.nan, "pedestal": np.nan}
    dz = dz_um[valid]
    y = profile[valid]

    if wing_only_pedestal:
        wings = np.abs(dz) > PEDESTAL_WING_UM
        if wings.sum() < pedestal_order + 1:
            return {"ok": False, "mu_um": np.nan, "amplitude": np.nan, "pedestal": np.nan}
        coeffs = np.polyfit(dz[wings], y[wings], pedestal_order)
        residual = y - np.polyval(coeffs, dz)

        def narrow(d, amplitude, mu):
            return amplitude * np.exp(-((d - mu) ** 2) / (2.0 * sigma_um**2))

        try:
            popt, pcov = curve_fit(
                narrow,
                dz,
                residual,
                p0=[max(residual.max(), 1e-6), 0.0],
                bounds=([-np.inf, -MU_BOUND_UM], [np.inf, MU_BOUND_UM]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            return {"ok": False, "mu_um": np.nan, "amplitude": np.nan, "pedestal": np.nan}
        return {
            "ok": True,
            "mu_um": float(popt[1]),
            "amplitude": float(popt[0]),
            "pedestal": float(np.polyval(coeffs, 0.0)),
            "mu_se_um": float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else np.nan,
        }

    def model(d, c0, c1, c2, amplitude, mu):
        if pedestal_order < 2:
            c2 = 0.0
        return _gaussian_with_pedestal(d, c0, c1, c2, amplitude, mu, sigma_um)

    c0_0 = float(np.median(y))
    a_0 = float(max(y.max() - c0_0, 1e-6))
    lower = [-np.inf, -np.inf, -np.inf if pedestal_order >= 2 else -1e-12, -np.inf, -MU_BOUND_UM]
    upper = [np.inf, np.inf, np.inf if pedestal_order >= 2 else 1e-12, np.inf, MU_BOUND_UM]
    try:
        popt, pcov = curve_fit(
            model, dz, y, p0=[c0_0, 0.0, 0.0, a_0, 0.0], bounds=(lower, upper), maxfev=20000
        )
    except (RuntimeError, ValueError):
        return {"ok": False, "mu_um": np.nan, "amplitude": np.nan, "pedestal": np.nan}
    return {
        "ok": True,
        "mu_um": float(popt[4]),
        "amplitude": float(popt[3]),
        "pedestal": float(popt[0]),
        "mu_se_um": float(np.sqrt(pcov[4, 4])) if np.all(np.isfinite(pcov)) else np.nan,
    }


def xy_offset_by_crosscorr(
    dog_a: np.ndarray,
    dog_b: np.ndarray,
    z_shift_slices: int,
    pixel_size_um: float,
    max_shift_um: float = 1.5,
) -> tuple[float, float, float]:
    """Lateral offset between two DoG volumes, at a given axial shift.

    Returns (dy_um, dx_um, peak_correlation). b is shifted by
    ``z_shift_slices`` relative to a before the z-projection, so this measures
    the lateral offset *at the fitted axial registration*, which is the only
    place the two channels' structures should correspond.
    """
    nz = dog_a.shape[0]
    lo_a = max(0, -z_shift_slices)
    hi_a = min(nz, nz - z_shift_slices)
    if hi_a - lo_a < 1:
        return np.nan, np.nan, np.nan
    a = dog_a[lo_a:hi_a].sum(axis=0)
    b = dog_b[lo_a + z_shift_slices : hi_a + z_shift_slices].sum(axis=0)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom <= 0:
        return np.nan, np.nan, np.nan
    correlation = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / denom
    correlation = np.fft.fftshift(correlation)
    centre = np.array(correlation.shape) // 2
    radius = max(1, int(round(max_shift_um / pixel_size_um)))
    window = correlation[
        centre[0] - radius : centre[0] + radius + 1, centre[1] - radius : centre[1] + radius + 1
    ]
    peak = np.unravel_index(np.argmax(window), window.shape)
    dy = (peak[0] - radius) * pixel_size_um
    dx = (peak[1] - radius) * pixel_size_um
    return float(dy), float(dx), float(window[peak])


@dataclass
class TileProfiles:
    """Per-focus profiles from one tile, small enough to ship between processes."""

    round_label: str
    tile: str
    well_position_id: float
    z_step_um: float
    pixel_size_um: float
    dz_um: np.ndarray
    forward_obs: np.ndarray  # (n_foci, n_dz) RFP_561-triggered GFP_488
    forward_null: np.ndarray
    forward_amplitude: np.ndarray
    forward_stats: "DetectionStats"
    reverse_obs: np.ndarray  # (n_foci, n_dz) GFP_488-triggered RFP_561
    reverse_null: np.ndarray
    reverse_amplitude: np.ndarray
    reverse_stats: "DetectionStats"
    n_nuclei: int
    xy_by_shift: dict
    error: str | None = None


_FLATFIELD_CACHE: dict[str, np.ndarray] = {}


def load_flatfield(analysis_dir: Path, relative_path: str) -> np.ndarray | None:
    if not relative_path or not isinstance(relative_path, str):
        return None
    if relative_path not in _FLATFIELD_CACHE:
        path = analysis_dir / relative_path
        if not path.exists():
            return None
        _FLATFIELD_CACHE[relative_path] = tifffile.imread(path).astype(np.float32)
    return _FLATFIELD_CACHE[relative_path]


def _process_tile(args: tuple) -> TileProfiles:
    row, analysis_dir, mad_k_min, seed = args
    try:
        return _process_tile_inner(row, analysis_dir, mad_k_min, seed)
    except Exception as exc:  # noqa: BLE001
        return TileProfiles(
            round_label=row["round"],
            tile=row["tile"],
            well_position_id=row.get("well_position_id", np.nan),
            z_step_um=np.nan,
            pixel_size_um=np.nan,
            dz_um=np.zeros(0),
            forward_obs=np.zeros((0, 0)),
            forward_null=np.zeros((0, 0)),
            forward_amplitude=np.zeros(0),
            forward_stats=DetectionStats(0.0, 0.0),
            reverse_obs=np.zeros((0, 0)),
            reverse_null=np.zeros((0, 0)),
            reverse_amplitude=np.zeros(0),
            reverse_stats=DetectionStats(0.0, 0.0),
            n_nuclei=0,
            xy_by_shift={},
            error=f"{type(exc).__name__}: {exc}",
        )


def _channel_index_map(channel_map: str) -> dict[str, int]:
    out = {}
    for part in str(channel_map).split(";"):
        wavelength, index = part.split("=C")
        out[CHANNEL_LABELS[wavelength]] = int(index)
    return out


def _detect_and_profile(
    trigger_volume_raw: np.ndarray,
    target_volume_flat: np.ndarray,
    nuclear_mask: np.ndarray,
    labels_2d: np.ndarray,
    pixels_by_label: dict[int, np.ndarray],
    pixel_size_um: float,
    z_step_um: float,
    dz_slices: np.ndarray,
    mad_k: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DetectionStats, np.ndarray]:
    dog = difference_of_gaussians(trigger_volume_raw, pixel_size_um, z_step_um)
    coords, amplitudes, stats = detect_foci_3d(dog, nuclear_mask, pixel_size_um, z_step_um, mad_k)

    edge_margin_px = int(round((EDGE_MARGIN_UM + DISC_RADIUS_UM) / pixel_size_um))
    # Screen on the NARROW range only - the minimum any window needs. Foci
    # that also reach the wide range are identified later by their profile
    # having no NaN there. Screening on the wide range here would silently
    # discard every focus in the shallow-slab fields, which is exactly the
    # two positions (5 and 6, one of them a control) the narrow window exists
    # to rescue.
    narrow_slices = dz_slices[np.abs(dz_slices) <= np.floor(DZ_MAX_NARROW_UM / z_step_um)]
    keep = qualifying_mask(coords.astype(float), narrow_slices, target_volume_flat.shape, edge_margin_px)
    coords, amplitudes = coords[keep], amplitudes[keep]
    if len(coords) > MAX_FOCI_PER_TILE:
        brightest = np.argsort(amplitudes)[::-1][:MAX_FOCI_PER_TILE]
        coords, amplitudes = coords[brightest], amplitudes[brightest]

    centers = refine_centroids_xy(dog, coords)
    focus_labels = labels_2d[coords[:, 1], coords[:, 2]] if len(coords) else np.zeros(0, dtype=int)

    disc_dy, disc_dx = disc_offsets(DISC_RADIUS_UM, pixel_size_um)
    observed = extract_triggered_profiles(target_volume_flat, centers, dz_slices, disc_dy, disc_dx)

    null_accumulator = np.zeros_like(observed)
    for _ in range(N_PERMUTATIONS):
        permuted = permuted_centers(centers, focus_labels, pixels_by_label, rng)
        keep_perm = qualifying_mask(permuted, dz_slices, target_volume_flat.shape, edge_margin_px)
        permuted[~keep_perm] = centers[~keep_perm]  # a draw that left the tile keeps the real position
        null_accumulator += extract_triggered_profiles(
            target_volume_flat, permuted, dz_slices, disc_dy, disc_dx
        )
    null = null_accumulator / N_PERMUTATIONS if N_PERMUTATIONS else null_accumulator
    return centers, observed, null, stats, amplitudes, dog


def _process_tile_inner(row, analysis_dir: Path, mad_k_min: float, seed: int) -> TileProfiles:
    path = Path(row["source_path"])
    inspection = inspect_ims(path)
    arrays = read_channel_arrays_ims(path, inspection, mode="3d", resolution_level=ANALYSIS_RESOLUTION_LEVEL)
    index_by_name = _channel_index_map(row["channel_map"])

    pixel_size_um = float(row["pixel_size_um_level0"])
    z_step_um = float(row["z_step_um"])
    focus_idx = int(row["focus_peak_idx"])

    dapi = arrays[index_by_name["DAPI_405"]]
    labels_2d, nuclear_mask = nucleus_labels(dapi, focus_idx, pixel_size_um)
    del dapi
    n_nuclei = int(labels_2d.max())
    if n_nuclei == 0:
        raise ValueError("no nuclei segmented")

    rfp = arrays[index_by_name["RFP_561"]].astype(np.float32)
    gfp = arrays[index_by_name["GFP_488"]].astype(np.float32)
    arrays.clear()

    # Stage 1b illumination correction, applied as a division before anything
    # measures intensity. Recorded per tile in stage0_tile_qc.csv.
    for name, volume in (("RFP_561", rfp), ("GFP_488", gfp)):
        field = load_flatfield(analysis_dir, row.get(f"flatfield_path_{name}"))
        if field is not None and field.shape == volume.shape[1:]:
            volume /= field[None, :, :]

    m_wide = int(np.floor(DZ_MAX_WIDE_UM / z_step_um))
    dz_slices = np.arange(-m_wide, m_wide + 1, dtype=np.int32)
    dz_um = dz_slices * z_step_um

    rfp_flat = flatten_background(rfp, BACKGROUND_SIGMA_UM, pixel_size_um)
    gfp_flat = flatten_background(gfp, BACKGROUND_SIGMA_UM, pixel_size_um)

    pixels_by_label = nucleus_pixel_index(labels_2d)
    rng = np.random.default_rng(seed)
    _, f_obs, f_null, f_stats, f_amp, dog_rfp = _detect_and_profile(
        rfp, gfp_flat, nuclear_mask, labels_2d, pixels_by_label,
        pixel_size_um, z_step_um, dz_slices, mad_k_min, rng,
    )
    _, r_obs, r_null, r_stats, r_amp, dog_gfp = _detect_and_profile(
        gfp, rfp_flat, nuclear_mask, labels_2d, pixels_by_label,
        pixel_size_um, z_step_um, dz_slices, mad_k_min, rng,
    )
    del rfp, gfp, rfp_flat, gfp_flat

    xy_by_shift = {}
    for shift in range(-2, 3):
        dy, dx, peak = xy_offset_by_crosscorr(dog_rfp, dog_gfp, shift, pixel_size_um)
        xy_by_shift[shift] = (dy, dx, peak)

    return TileProfiles(
        round_label=row["round"],
        tile=row["tile"],
        well_position_id=row.get("well_position_id", np.nan),
        z_step_um=z_step_um,
        pixel_size_um=pixel_size_um,
        dz_um=dz_um,
        forward_obs=f_obs,
        forward_null=f_null,
        forward_amplitude=f_amp,
        forward_stats=f_stats,
        reverse_obs=r_obs,
        reverse_null=r_null,
        reverse_amplitude=r_amp,
        reverse_stats=r_stats,
        n_nuclei=n_nuclei,
        xy_by_shift=xy_by_shift,
    )


def pooled_profile(tiles: list[TileProfiles], direction: str, dz_max_um: float, mad_k: float):
    """Pool per-focus profiles across tiles sharing one z-step grid."""
    rows_obs, rows_null, tile_index, dz_um = [], [], [], None
    for i, tile in enumerate(tiles):
        obs = tile.forward_obs if direction == "forward" else tile.reverse_obs
        null = tile.forward_null if direction == "forward" else tile.reverse_null
        amplitudes = tile.forward_amplitude if direction == "forward" else tile.reverse_amplitude
        stats = tile.forward_stats if direction == "forward" else tile.reverse_stats
        if obs.size == 0:
            continue
        keep_dz = np.abs(tile.dz_um) <= dz_max_um + 1e-9
        keep_foci = amplitudes >= stats.threshold(mad_k)
        # Complete coverage across this window, so the contributing focus set
        # is identical at every dz in it.
        keep_foci &= np.isfinite(obs[:, keep_dz]).all(axis=1)
        keep_foci &= np.isfinite(null[:, keep_dz]).all(axis=1)
        if keep_foci.sum() < MIN_FOCI_PER_TILE:
            continue
        rows_obs.append(obs[np.ix_(keep_foci, keep_dz)])
        rows_null.append(null[np.ix_(keep_foci, keep_dz)])
        tile_index.append(np.full(int(keep_foci.sum()), i))
        if dz_um is None:
            dz_um = tile.dz_um[keep_dz]
    if not rows_obs:
        return None
    return (
        dz_um,
        np.vstack(rows_obs),
        np.vstack(rows_null),
        np.concatenate(tile_index),
    )


def choose_primary_k(cache: dict) -> float:
    """Strictest detection threshold that still yields enough foci in BOTH directions.

    Registration wants the brightest, most reliably centroided foci, so the
    strictest usable k is preferred. But the threshold is median + k*MAD of
    the *in-nucleus* DoG, and in the densest fields the real foci themselves
    inflate that MAD - field _8 (well position 4, ~174 nuclei per tile) falls
    from thousands of foci to a median of 4 per tile between k=6 and k=8, and
    drops out of the analysis entirely. That is a sensitivity artifact, not
    biology, and fixing it properly is Stage 3's job.

    Backing off to a looser k for such a sample is safe HERE specifically
    because mu is demonstrably threshold-insensitive (the per-k columns in
    stage2_registration_per_sample.csv show < 0.05 um of movement across
    k = 6, 8, 12). This is a calibration quantity, not the blinded endpoint:
    no detection threshold chosen here propagates to Stage 6.
    """
    both = [
        k
        for k in sorted(DETECT_MAD_K, reverse=True)
        if ("forward", k) in cache
        and ("reverse", k) in cache
        and len(cache[("forward", k)][1]) >= MIN_POOLED_FOCI
        and len(cache[("reverse", k)][1]) >= MIN_POOLED_FOCI
    ]
    if both:
        return both[0]
    available = [k for k in sorted(DETECT_MAD_K, reverse=True) if ("forward", k) in cache]
    return available[0] if available else min(DETECT_MAD_K)


def bootstrap_delta(
    forward,
    reverse,
    sigma_um: float,
    pedestal_order: int,
    wing_only: bool,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Cluster bootstrap CI for the antisymmetric offset delta.

    Tiles are resampled ONCE per replicate and both directions are refitted
    on that same tile draw, because forward and reverse are measured on the
    same tiles and their errors are therefore correlated; resampling them
    independently would overstate the precision of their difference.
    """
    dz_f, obs_f, _, tiles_f = forward
    dz_r, obs_r, _, tiles_r = reverse
    shared = np.intersect1d(np.unique(tiles_f), np.unique(tiles_r))
    if len(shared) < 2:
        return np.nan, np.nan
    rows_f = {t: np.flatnonzero(tiles_f == t) for t in shared}
    rows_r = {t: np.flatnonzero(tiles_r == t) for t in shared}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(shared, size=len(shared), replace=True)
        pick_f = np.concatenate([rng.choice(rows_f[t], size=len(rows_f[t]), replace=True) for t in drawn])
        pick_r = np.concatenate([rng.choice(rows_r[t], size=len(rows_r[t]), replace=True) for t in drawn])
        fit_f = fit_triggered_profile(dz_f, obs_f[pick_f].mean(axis=0), sigma_um, pedestal_order, wing_only)
        fit_r = fit_triggered_profile(dz_r, obs_r[pick_r].mean(axis=0), sigma_um, pedestal_order, wing_only)
        if fit_f["ok"] and fit_r["ok"]:
            estimates.append((fit_f["mu_um"] - fit_r["mu_um"]) / 2.0)
    if len(estimates) < 10:
        return np.nan, np.nan
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def bootstrap_mu(
    dz_um: np.ndarray,
    obs: np.ndarray,
    tile_index: np.ndarray,
    sigma_um: float,
    pedestal_order: int,
    wing_only: bool,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Cluster bootstrap over TILES, then foci within the resampled tiles.

    Resampling foci alone would treat thousands of foci from a handful of
    tiles as independent and report a CI far too tight: nucleus in tile in
    field is a hierarchy, and tiles are the level at which focus drift,
    staining and illumination actually vary.
    """
    rng = np.random.default_rng(seed)
    unique_tiles = np.unique(tile_index)
    by_tile = {t: np.flatnonzero(tile_index == t) for t in unique_tiles}
    estimates = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique_tiles, size=len(unique_tiles), replace=True)
        picks = []
        for t in drawn:
            rows = by_tile[t]
            picks.append(rng.choice(rows, size=len(rows), replace=True))
        sample = obs[np.concatenate(picks)]
        fit = fit_triggered_profile(dz_um, sample.mean(axis=0), sigma_um, pedestal_order, wing_only)
        if fit["ok"]:
            estimates.append(fit["mu_um"])
    if len(estimates) < 10:
        return np.nan, np.nan
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TRF2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Where stage0_tile_qc.csv and flatfield/ live. Defaults to the output dir; "
        "set it separately to dry-run into a scratch directory without overwriting results.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rounds", type=str, default=None, help="Comma-separated round labels to restrict to")
    parser.add_argument(
        "--include-deconvolved",
        action="store_true",
        help="Include *_deconvolved folders. Off by default: the plan forbids mixing one "
        "deconvolved field with raw ones - all or none.",
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "_analysis")
    input_dir = args.input_dir or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_df = pd.read_csv(input_dir / "stage0_tile_qc.csv")
    usable = tile_df[(tile_df["read_ok"] == True) & tile_df["usable_z_range"].notna()].copy()  # noqa: E712
    if not args.include_deconvolved:
        usable = usable[~usable["round"].str.contains("deconvolved")]
    if args.rounds:
        wanted = {r.strip() for r in args.rounds.split(",")}
        usable = usable[usable["round"].isin(wanted)]
    if args.limit:
        usable = usable.head(args.limit)
    print(f"Stage 2: {len(usable)} tiles to process ({args.workers} workers)")

    mad_k_min = min(DETECT_MAD_K)
    tasks = [(row, input_dir, mad_k_min, 1000 + i) for i, (_, row) in enumerate(usable.iterrows())]
    results: list[TileProfiles] = []
    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_tile, tasks, chunksize=1):
            results.append(result)
            if result.error:
                print(f"  WARNING {result.round_label}/{result.tile}: {result.error}")
    good = [r for r in results if r.error is None and r.forward_obs.size]
    print(f"Profiled {len(good)}/{len(results)} tiles; "
          f"{sum(len(r.forward_amplitude) for r in good)} RFP_561 foci, "
          f"{sum(len(r.reverse_amplitude) for r in good)} GFP_488 foci")
    if not good:
        print("No tiles produced profiles - nothing to fit.")
        return 1

    # Each window gets the pedestal treatment that is actually identifiable
    # for it. Over the wide window a free quadratic pedestal is degenerate
    # with the narrow component - it can mimic the bump, swallow it, and
    # leave the Gaussian fitting a negative dip at the wrong dz (verified on
    # synthetic data: it returned mu=-0.55 um with amplitude -79 against a
    # planted +1.0 um). So the wide window uses the plan's recipe: fit the
    # pedestal on the wings alone, where the narrow component is < 0.2% of
    # its peak, then fit A and mu to the residual. The narrow window has no
    # wings, but a strictly LINEAR pedestal has no curvature and so cannot
    # mimic a symmetric bump - that fit is well conditioned jointly.
    windows = {
        "wide": (DZ_MAX_WIDE_UM, 2, True),
        "narrow": (DZ_MAX_NARROW_UM, 1, False),
    }
    sample_rows, profile_rows, tile_rows, decomposition_rows = [], [], [], []

    for position in sorted({r.well_position_id for r in good if np.isfinite(r.well_position_id)}):
        at_position = [r for r in good if r.well_position_id == position]
        z_steps = pd.Series([r.z_step_um for r in at_position])
        modal_z_step = float(z_steps.mode().iloc[0])
        matched = [r for r in at_position if abs(r.z_step_um - modal_z_step) < 1e-6]
        sigma_um = fixed_sigma_um(modal_z_step)

        for window_name, (dz_max, pedestal_order, wing_only) in windows.items():
            cache = {}
            for direction in ("forward", "reverse"):
                for mad_k in DETECT_MAD_K:
                    pooled = pooled_profile(matched, direction, dz_max, mad_k)
                    if pooled is not None:
                        cache[(direction, mad_k)] = pooled
            if not cache:
                continue
            primary_k = choose_primary_k(cache)

            for (direction, mad_k), (dz_um, obs, null, tile_index) in sorted(cache.items()):
                is_primary = mad_k == primary_k
                mean_obs = obs.mean(axis=0)
                mean_null = null.mean(axis=0)
                fit = fit_triggered_profile(dz_um, mean_obs, sigma_um, pedestal_order, wing_only)
                fit_alt = fit_triggered_profile(dz_um, mean_obs, sigma_um, pedestal_order, not wing_only)
                fit_minus_null = fit_triggered_profile(
                    dz_um, mean_obs - mean_null, sigma_um, pedestal_order, wing_only
                )
                ci_lo, ci_hi = (np.nan, np.nan)
                if is_primary and direction == "forward" and fit["ok"]:
                    ci_lo, ci_hi = bootstrap_mu(
                        dz_um, obs, tile_index, sigma_um, pedestal_order, wing_only,
                        args.bootstrap, int(position) * 17 + 3,
                    )
                sample_rows.append(
                    {
                        "well_position_id": position,
                        "window": window_name,
                        "direction": direction,
                        "detect_mad_k": mad_k,
                        "is_primary_k": is_primary,
                        "n_tiles": len(np.unique(tile_index)),
                        "n_foci": len(obs),
                        "z_step_um": modal_z_step,
                        "sigma_fixed_um": sigma_um,
                        "mu_um": fit["mu_um"],
                        "mu_ci_lo_um": ci_lo,
                        "mu_ci_hi_um": ci_hi,
                        "narrow_amplitude": fit["amplitude"],
                        "pedestal_amplitude": fit["pedestal"],
                        "amplitude_positive": bool(fit["ok"] and fit["amplitude"] > 0),
                        "mu_um_alt_pedestal": fit_alt["mu_um"],
                        "mu_um_obs_minus_null": fit_minus_null["mu_um"],
                        "rounds": ",".join(sorted({r.round_label for r in matched})),
                    }
                )
                if is_primary:
                    for i, dz in enumerate(dz_um):
                        profile_rows.append(
                            {
                                "well_position_id": position,
                                "window": window_name,
                                "direction": direction,
                                "detect_mad_k": mad_k,
                                "dz_um": float(dz),
                                "observed": float(mean_obs[i]),
                                "permuted_null": float(mean_null[i]),
                                "observed_sem": float(obs[:, i].std(ddof=1) / np.sqrt(len(obs))),
                            }
                        )

            # --- The actual deliverable: split forward/reverse into the
            # --- antisymmetric (registration) and common-mode parts.
            forward = cache.get(("forward", primary_k))
            reverse = cache.get(("reverse", primary_k))
            if forward is None or reverse is None:
                continue
            mu_f = fit_triggered_profile(
                forward[0], forward[1].mean(axis=0), sigma_um, pedestal_order, wing_only
            )
            mu_r = fit_triggered_profile(
                reverse[0], reverse[1].mean(axis=0), sigma_um, pedestal_order, wing_only
            )
            if not (mu_f["ok"] and mu_r["ok"]):
                continue
            delta = (mu_f["mu_um"] - mu_r["mu_um"]) / 2.0
            common = (mu_f["mu_um"] + mu_r["mu_um"]) / 2.0
            d_lo, d_hi = bootstrap_delta(
                forward, reverse, sigma_um, pedestal_order, wing_only,
                args.bootstrap, int(position) * 31 + 7,
            )
            decomposition_rows.append(
                {
                    "well_position_id": position,
                    "window": window_name,
                    "detect_mad_k": primary_k,
                    "mu_forward_um": mu_f["mu_um"],
                    "mu_reverse_um": mu_r["mu_um"],
                    "delta_registration_um": delta,
                    "delta_ci_lo_um": d_lo,
                    "delta_ci_hi_um": d_hi,
                    "common_mode_um": common,
                    "n_foci_forward": len(forward[1]),
                    "n_foci_reverse": len(reverse[1]),
                    "rounds": ",".join(sorted({r.round_label for r in matched})),
                }
            )

    for tile in good:
        sigma_um = fixed_sigma_um(tile.z_step_um)
        keep_dz = np.abs(tile.dz_um) <= DZ_MAX_NARROW_UM + 1e-9
        keep = tile.forward_amplitude >= tile.forward_stats.threshold(PRIMARY_MAD_K)
        if tile.forward_obs.size:
            keep &= np.isfinite(tile.forward_obs[:, keep_dz]).all(axis=1)
        row = {
            "round": tile.round_label,
            "tile": tile.tile,
            "well_position_id": tile.well_position_id,
            "n_nuclei": tile.n_nuclei,
            "n_foci_RFP_561": int(keep.sum()),
            "n_foci_GFP_488": len(tile.reverse_amplitude),
            "z_step_um": tile.z_step_um,
            "mu_um_narrow": np.nan,
        }
        if keep.sum() >= MIN_FOCI_PER_TILE:
            fit = fit_triggered_profile(
                tile.dz_um[keep_dz], tile.forward_obs[np.ix_(keep, keep_dz)].mean(axis=0),
                sigma_um, pedestal_order=1, wing_only_pedestal=False,
            )
            row["mu_um_narrow"] = fit["mu_um"]
        tile_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)
    profile_df = pd.DataFrame(profile_rows)
    tile_df_out = pd.DataFrame(tile_rows)
    decomposition_df = pd.DataFrame(decomposition_rows)

    # Lateral offset, evaluated at the z-shift nearest each sample's fitted mu.
    lateral_rows = []
    primary = sample_df[
        (sample_df["direction"] == "forward")
        & (sample_df["is_primary_k"])
        & (sample_df["window"] == "narrow")
    ]
    for position, group in primary.groupby("well_position_id"):
        mu = float(group["mu_um"].iloc[0])
        for tile in [t for t in good if t.well_position_id == position]:
            shift = int(np.rint(mu / tile.z_step_um)) if np.isfinite(mu) else 0
            shift = int(np.clip(shift, -2, 2))
            dy, dx, peak = tile.xy_by_shift.get(shift, (np.nan, np.nan, np.nan))
            lateral_rows.append(
                {"well_position_id": position, "round": tile.round_label, "tile": tile.tile,
                 "z_shift_slices": shift, "dy_um": dy, "dx_um": dx, "corr_peak": peak}
            )
    lateral_df = pd.DataFrame(lateral_rows)

    sample_df.to_csv(output_dir / "stage2_registration_per_sample.csv", index=False)
    profile_df.to_csv(output_dir / "stage2_registration_profiles.csv", index=False)
    tile_df_out.to_csv(output_dir / "stage2_registration_per_tile.csv", index=False)
    lateral_df.to_csv(output_dir / "stage2_registration_lateral.csv", index=False)
    decomposition_df.to_csv(output_dir / "stage2_registration_decomposition.csv", index=False)

    _print_summary(sample_df, lateral_df, decomposition_df)
    _write_pdf(
        output_dir / "stage2_registration.pdf",
        sample_df, profile_df, tile_df_out, lateral_df, decomposition_df,
    )
    print(f"\nWrote {output_dir / 'stage2_registration.pdf'}")
    return 0


def _print_summary(sample_df: pd.DataFrame, lateral_df: pd.DataFrame, decomposition_df: pd.DataFrame) -> None:
    if len(decomposition_df):
        print("\n=== REGISTRATION OFFSET (the deliverable) ===")
        print("A real channel offset d gives forward=+d and reverse=-d, so only the")
        print("ANTISYMMETRIC half is registration. The common-mode half is an axial")
        print("gradient of the stack itself (it keeps its sign when the trigger channel")
        print("is swapped, which a registration offset cannot do) and must NOT be applied.")
        print(f"\n{'pos':>4} {'window':>7} {'k':>3} {'mu_fwd':>8} {'mu_rev':>8} {'delta':>8} "
              f"{'delta 95% CI':>20} {'common':>8}")
        for _, row in decomposition_df.sort_values(["window", "well_position_id"]).iterrows():
            ci = (f"[{row['delta_ci_lo_um']:+.3f}, {row['delta_ci_hi_um']:+.3f}]"
                  if np.isfinite(row["delta_ci_lo_um"]) else "n/a")
            print(f"{row['well_position_id']:>4.0f} {row['window']:>7} {row['detect_mad_k']:>3.0f} "
                  f"{row['mu_forward_um']:>+8.3f} {row['mu_reverse_um']:>+8.3f} "
                  f"{row['delta_registration_um']:>+8.3f} {ci:>20} {row['common_mode_um']:>+8.3f}")
        narrow = decomposition_df[decomposition_df["window"] == "narrow"]
        if len(narrow):
            worst = narrow["delta_registration_um"].abs().max()
            print(f"\nLargest |delta| across samples (narrow window): {worst:.3f} um")
            print("Plan's acceptance criterion is 0 +- 0.1 um.",
                  "MET - no material axial offset." if worst <= 0.1 else "NOT met.")

    primary = sample_df[sample_df["is_primary_k"]]
    print("\nPer-sample axial offset mu (um), forward = RFP_561-triggered GFP_488:")
    print(f"{'pos':>4} {'window':>7} {'n_tiles':>8} {'n_foci':>8} {'mu':>8} {'95% CI':>18} {'A':>10} {'pedestal':>10}")
    for _, row in primary[primary["direction"] == "forward"].sort_values(["well_position_id", "window"]).iterrows():
        ci = f"[{row['mu_ci_lo_um']:+.3f}, {row['mu_ci_hi_um']:+.3f}]" if np.isfinite(row["mu_ci_lo_um"]) else "n/a"
        print(f"{row['well_position_id']:>4.0f} {row['window']:>7} {row['n_tiles']:>8.0f} {row['n_foci']:>8.0f} "
              f"{row['mu_um']:>+8.3f} {ci:>18} {row['narrow_amplitude']:>10.2f} {row['pedestal_amplitude']:>10.2f}")

    print("\nRaw sign-flip check (literal form: forward and reverse must have opposite signs).")
    print("A FAIL here is expected when a common-mode term is present and larger than delta;")
    print("read the decomposition above, not this line, for the registration answer:")
    for window in ("narrow", "wide"):
        sub = primary[primary["window"] == window]
        for position in sorted(sub["well_position_id"].unique()):
            fwd = sub[(sub["well_position_id"] == position) & (sub["direction"] == "forward")]["mu_um"]
            rev = sub[(sub["well_position_id"] == position) & (sub["direction"] == "reverse")]["mu_um"]
            if len(fwd) and len(rev):
                f, r = float(fwd.iloc[0]), float(rev.iloc[0])
                verdict = "PASS" if np.isfinite(f) and np.isfinite(r) and f * r < 0 else "FAIL"
                print(f"  pos {position:.0f} [{window:>6}] forward {f:+.3f}  reverse {r:+.3f}  sum {f + r:+.3f}  {verdict}")

    print("\nmu vs detection threshold (forward, narrow window) - mu must not move with k:")
    narrow = sample_df[(sample_df["window"] == "narrow") & (sample_df["direction"] == "forward")]
    for position in sorted(narrow["well_position_id"].unique()):
        values = narrow[narrow["well_position_id"] == position].sort_values("detect_mad_k")
        text = "  ".join(f"k={r['detect_mad_k']:.0f}: {r['mu_um']:+.3f}" for _, r in values.iterrows())
        print(f"  pos {position:.0f}  {text}")

    if len(lateral_df):
        print("\nLateral offset at the fitted axial shift (median over tiles):")
        for position, group in lateral_df.groupby("well_position_id"):
            print(f"  pos {position:.0f}  dy {group['dy_um'].median():+.3f} um  dx {group['dx_um'].median():+.3f} um")


def _write_pdf(path: Path, sample_df, profile_df, tile_df_out, lateral_df, decomposition_df) -> None:
    primary = sample_df[sample_df["is_primary_k"]]
    with PdfPages(path) as pdf:
        if len(decomposition_df):
            fig = plt.figure(figsize=(11, 8.5), dpi=300)
            fig.suptitle("Stage 2: is there an axial registration offset between the channels?", fontsize=13)
            ax = fig.add_axes((0.06, 0.52, 0.88, 0.34))
            narrow = decomposition_df[decomposition_df["window"] == "narrow"].sort_values("well_position_id")
            positions = narrow["well_position_id"].to_numpy()
            ax.axhspan(-0.1, 0.1, color="tab:green", alpha=0.12,
                       label="plan's acceptance band (0 +- 0.1 um)")
            lower = narrow["delta_registration_um"] - narrow["delta_ci_lo_um"]
            upper = narrow["delta_ci_hi_um"] - narrow["delta_registration_um"]
            ax.errorbar(positions, narrow["delta_registration_um"],
                        yerr=[lower.abs(), upper.abs()], fmt="o", capsize=3, color="tab:blue",
                        label="delta = (fwd - rev)/2  [registration]")
            ax.plot(positions, narrow["common_mode_um"], "s--", color="tab:red", alpha=0.8,
                    label="common mode = (fwd + rev)/2  [axial gradient, NOT registration]")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xlabel("well position id")
            ax.set_ylabel("axial offset (um)")
            ax.legend(fontsize=7, loc="best")
            ax.set_title("narrow window, primary detection threshold per sample", fontsize=9)

            ax2 = fig.add_axes((0.06, 0.08, 0.88, 0.33))
            ax2.axis("off")
            show = decomposition_df[
                ["well_position_id", "window", "detect_mad_k", "mu_forward_um", "mu_reverse_um",
                 "delta_registration_um", "delta_ci_lo_um", "delta_ci_hi_um", "common_mode_um"]
            ].copy()
            for column in show.columns:
                if column.endswith("_um"):
                    show[column] = show[column].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "n/a")
            table = ax2.table(cellText=show.values, colLabels=list(show.columns), loc="upper center",
                              cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1, 1.25)
            worst = decomposition_df[decomposition_df["window"] == "narrow"]["delta_registration_um"].abs().max()
            fig.text(0.06, 0.015,
                     "A genuine channel offset d must appear as forward=+d and reverse=-d, so it is the "
                     "ANTISYMMETRIC half of the two measurements.\nThe common-mode half keeps its sign when the "
                     "trigger channel is swapped - a registration offset cannot do that - so it is a property of "
                     "the stack's own axial\nintensity gradient (Stage 0: tilted slide, stacks started too high) "
                     f"and must NOT be applied as a shift. Largest |delta| = {worst:.3f} um.",
                     fontsize=7, va="bottom")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        fig.suptitle("Stage 2: registration self-calibration - per-sample axial offset", fontsize=13)
        ax = fig.add_axes((0.03, 0.08, 0.94, 0.82))
        ax.axis("off")
        show = primary[primary["direction"] == "forward"][
            ["well_position_id", "window", "rounds", "n_tiles", "n_foci", "mu_um",
             "mu_ci_lo_um", "mu_ci_hi_um", "narrow_amplitude", "mu_um_alt_pedestal"]
        ].copy()
        for column in ("mu_um", "mu_ci_lo_um", "mu_ci_hi_um", "mu_um_alt_pedestal"):
            show[column] = show[column].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "n/a")
        show["narrow_amplitude"] = show["narrow_amplitude"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
        table = ax.table(cellText=show.values, colLabels=list(show.columns), loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.3)
        fig.text(0.03, 0.045,
                 "mu = fitted centre of the narrow (PSF-width) component of the RFP_561-triggered GFP_488 profile, "
                 "i.e. the axial offset to apply before any colocalization.\n"
                 "'wide' = +-2.5 um window with a quadratic pedestal (the plan's canonical fit); 'narrow' = +-1.5 um "
                 "with a linear pedestal, the only window the tilted-slide\nusable z slabs permit at positions 5 and 6. "
                 "CI = 95% cluster bootstrap over tiles, then foci within tiles. mu is only interpretable where the "
                 "narrow amplitude is positive.", fontsize=7, va="bottom")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        for position in sorted(profile_df["well_position_id"].unique()):
            fig = plt.figure(figsize=(11, 8.5), dpi=300)
            grid = fig.add_gridspec(2, 2)
            for panel, (window, direction) in enumerate(
                [("narrow", "forward"), ("wide", "forward"), ("narrow", "reverse"), ("wide", "reverse")]
            ):
                ax = fig.add_subplot(grid[panel // 2, panel % 2])
                sub = profile_df[
                    (profile_df["well_position_id"] == position)
                    & (profile_df["window"] == window)
                    & (profile_df["direction"] == direction)
                ].sort_values("dz_um")
                if not len(sub):
                    ax.text(0.5, 0.5, f"{window} / {direction}\nno data", ha="center", va="center", fontsize=8)
                    ax.axis("off")
                    continue
                ax.errorbar(sub["dz_um"], sub["observed"], yerr=sub["observed_sem"], marker="o", lw=1, label="observed")
                ax.plot(sub["dz_um"], sub["permuted_null"], marker="s", lw=1, label="within-nucleus permuted")
                row = primary[
                    (primary["well_position_id"] == position)
                    & (primary["window"] == window)
                    & (primary["direction"] == direction)
                ]
                if len(row) and np.isfinite(row["mu_um"].iloc[0]):
                    ax.axvline(float(row["mu_um"].iloc[0]), color="crimson", ls="--", lw=1)
                    ax.set_title(f"{window} / {direction}   mu={float(row['mu_um'].iloc[0]):+.3f} um  "
                                 f"A={float(row['narrow_amplitude'].iloc[0]):.2f}  n={int(row['n_foci'].iloc[0])}",
                                 fontsize=8)
                ax.axvline(0.0, color="grey", lw=0.6)
                ax.set_xlabel("axial offset dz (um)", fontsize=8)
                ax.set_ylabel("mean background-flattened intensity in r=0.3 um disc", fontsize=7)
                ax.legend(fontsize=6)
                ax.tick_params(labelsize=7)
            rounds = primary[primary["well_position_id"] == position]["rounds"]
            lateral = lateral_df[lateral_df["well_position_id"] == position]
            fig.suptitle(
                f"Stage 2 - well position {position:.0f} (rounds: {rounds.iloc[0] if len(rounds) else 'n/a'})\n"
                f"triggered-average profiles; red dashed = fitted mu, grey = dz 0. "
                f"Lateral offset at fitted shift: dy {lateral['dy_um'].median():+.3f} um, "
                f"dx {lateral['dx_um'].median():+.3f} um"
                if len(lateral) else f"Stage 2 - well position {position:.0f}",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5), dpi=300)
        ax = fig.add_subplot(2, 1, 1)
        narrow = sample_df[(sample_df["window"] == "narrow") & (sample_df["direction"] == "forward")]
        for k in DETECT_MAD_K:
            sub = narrow[narrow["detect_mad_k"] == k].sort_values("well_position_id")
            ax.plot(sub["well_position_id"], sub["mu_um"], marker="o", label=f"detection k={k:.0f}")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("well position id")
        ax.set_ylabel("mu (um)")
        ax.set_title("mu vs detection threshold - a mu that moves with k would not be trustworthy", fontsize=9)
        ax.legend(fontsize=7)

        ax2 = fig.add_subplot(2, 1, 2)
        valid = tile_df_out[tile_df_out["mu_um_narrow"].notna()]
        if len(valid):
            ax2.scatter(valid["well_position_id"], valid["mu_um_narrow"], s=12, alpha=0.6)
            ax2.axhline(0, color="grey", lw=0.6)
        ax2.set_xlabel("well position id")
        ax2.set_ylabel("per-tile mu (um)")
        ax2.set_title("per-tile mu scatter (diagnostic; tiles are the independent unit, not foci)", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
