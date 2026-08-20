import numpy as np
import json
from dataclasses import dataclass

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - scipy is expected in production requirements.
    least_squares = None

DEBIAS_THRESHOLD = 1.57
SIGMA_VALIDITY_THRESHOLD = 3.0
EPSILON = 1e-6
STOKES_LABELS = ("I", "Q", "U", "V")
PHASE_SLICE_QUANTITY_KEYS = ("P/I", "L/I", "|V/I|", "V/I", "PA", "EA")
POLARISATION_STACK_KEYS = ("PA", "EA", "P/I", "L/I", "|V/I|", "V/I")


def _angle_residual_deg(observed, model, period=180.0):
    return ((np.asarray(observed, dtype=float) - np.asarray(model, dtype=float) + period / 2) % period) - period / 2


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_list(values):
    values = np.asarray(values, dtype=float)
    return [_finite_or_none(value) for value in values]


def _finite_matrix(values):
    values = np.asarray(values, dtype=float)
    return [_finite_list(row) for row in values]


def _phase_slice_quantity_specs(params):
    return {
        "P/I": (params.p_frac, "P/I"),
        "L/I": (params.l_frac, "L/I"),
        "|V/I|": (params.abs_vfrac, "|V/I|"),
        "V/I": (params.v_frac, "V/I"),
        "PA": (params.PA_deg, "PA [deg]"),
        "EA": (params.EA_deg, "EA [deg]"),
    }


def _polarisation_stack_specs(params):
    return {
        "PA": (params.PA_deg, "PA [deg]"),
        "EA": (params.EA_deg, "EA [deg]"),
        "P/I": (params.p_frac, "P/I"),
        "L/I": (params.l_frac, "L/I"),
        "|V/I|": (params.abs_vfrac, "|V/I|"),
        "V/I": (params.v_frac, "V/I"),
    }


def _phase_slice_histogram_specs(params, sigma_threshold=3.0):
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses
    on_pulse_mask = (phase_axis >= float(params.on_pulse[0])) & (phase_axis <= float(params.on_pulse[1]))
    off_pulse_mask = ~on_pulse_mask
    warnings = []

    if np.sum(off_pulse_mask) < 5:
        warnings.append("Not enough off-pulse bins for baseline subtraction and noise masks.")
        baseline_I = np.zeros((num_pulses, 1), dtype=float)
        baseline_Q = np.zeros((num_pulses, 1), dtype=float)
        baseline_U = np.zeros((num_pulses, 1), dtype=float)
        baseline_V = np.zeros((num_pulses, 1), dtype=float)
    else:
        baseline_I = np.nanmean(params.I[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_Q = np.nanmean(params.Q[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_U = np.nanmean(params.U[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_V = np.nanmean(params.V[:, off_pulse_mask], axis=1, keepdims=True)

    I0 = params.I - baseline_I
    Q0 = params.Q - baseline_Q
    U0 = params.U - baseline_U
    V0 = params.V - baseline_V

    def _offpulse_sigma(values):
        if np.sum(off_pulse_mask) < 5:
            return EPSILON
        finite = np.asarray(values[:, off_pulse_mask], dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return EPSILON
        return _safe_scalar(np.nanstd(finite))

    sigma_I = _offpulse_sigma(I0)
    sigma_Q = _offpulse_sigma(Q0)
    sigma_U = _offpulse_sigma(U0)
    sigma_V = _offpulse_sigma(V0)
    sigma_L = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2))
    sigma_P = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2 + sigma_V ** 2))
    sigma_threshold = float(sigma_threshold) if np.isfinite(float(sigma_threshold)) else 3.0

    L_raw = np.sqrt(Q0 ** 2 + U0 ** 2)
    P_raw = np.sqrt(Q0 ** 2 + U0 ** 2 + V0 ** 2)
    L_debiased = _debias_polarisation(L_raw, sigma_L)
    P_debiased = _debias_polarisation(P_raw, sigma_P)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_frac = np.divide(P_debiased, I0, out=np.full_like(P_debiased, np.nan, dtype=float), where=np.abs(I0) > EPSILON)
        l_frac = np.divide(L_debiased, I0, out=np.full_like(L_debiased, np.nan, dtype=float), where=np.abs(I0) > EPSILON)
        v_frac = np.divide(V0, I0, out=np.full_like(V0, np.nan, dtype=float), where=np.abs(I0) > EPSILON)

    PA_deg = np.degrees(0.5 * np.arctan2(U0, Q0))
    EA_deg = np.degrees(0.5 * np.arctan2(V0, L_debiased))
    intensity_valid = np.abs(I0) > sigma_threshold * sigma_I
    linear_valid = L_raw > sigma_threshold * sigma_L
    total_pol_valid = P_raw > sigma_threshold * sigma_P

    specs = {
        "P/I": (p_frac, "P/I", intensity_valid, [0.0, 1.5], "abs(I0) > sigma_threshold * sigma_I"),
        "L/I": (l_frac, "L/I", intensity_valid, [0.0, 1.5], "abs(I0) > sigma_threshold * sigma_I"),
        "|V/I|": (np.abs(v_frac), "|V/I|", intensity_valid, [0.0, 1.5], "abs(I0) > sigma_threshold * sigma_I"),
        "V/I": (v_frac, "V/I", intensity_valid, [-1.5, 1.5], "abs(I0) > sigma_threshold * sigma_I"),
        "PA": (PA_deg, "PA [deg]", linear_valid, [-90.0, 90.0], "L_raw > sigma_threshold * sqrt(sigma_Q^2 + sigma_U^2)"),
        "EA": (EA_deg, "EA [deg]", total_pol_valid, [-45.0, 45.0], "P_raw > sigma_threshold * sqrt(sigma_Q^2 + sigma_U^2 + sigma_V^2)"),
    }
    metadata = {
        "sigma_threshold": float(sigma_threshold),
        "sigma_I": _finite_or_none(sigma_I),
        "sigma_Q": _finite_or_none(sigma_Q),
        "sigma_U": _finite_or_none(sigma_U),
        "sigma_V": _finite_or_none(sigma_V),
        "sigma_L": _finite_or_none(sigma_L),
        "sigma_P": _finite_or_none(sigma_P),
        "uses_rician_debiased_LP": True,
    }
    return specs, warnings, metadata


def _adaptive_display_range(values, default_range):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    vmin, vmax = float(default_range[0]), float(default_range[1])
    if not finite.size:
        return vmin, vmax

    data_min = float(np.nanmin(finite))
    data_max = float(np.nanmax(finite))
    vmin = min(vmin, data_min)
    vmax = max(vmax, data_max)
    if vmin == vmax:
        pad = max(abs(vmin) * 0.1, 0.5)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _iqr(values):
    if len(values) == 0:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return q75 - q25


@dataclass(slots=True)
class StokesPrecompute:
    data: np.ndarray
    num_pulses: int
    num_bins: int
    phase_axis: np.ndarray
    pulse_number: np.ndarray
    I: np.ndarray
    Q: np.ndarray
    U: np.ndarray
    V: np.ndarray
    I_mean: np.ndarray
    Q_mean: np.ndarray
    U_mean: np.ndarray
    V_mean: np.ndarray
    mean_profiles: np.ndarray
    I0: float
    I_over_I0: np.ndarray
    I_mean_over_I0: np.ndarray


@dataclass(slots=True)
class PolarimetryPrecompute(StokesPrecompute):
    on_pulse: tuple[float, float]
    on_pulse_mask: np.ndarray
    off_pulse_mask: np.ndarray
    off_pulse_std: float
    threshold: float
    L: np.ndarray
    L_sigma: np.ndarray
    L_mask: np.ndarray
    L_true: np.ndarray
    P: np.ndarray
    P_sigma: np.ndarray
    P_mask: np.ndarray
    P_true: np.ndarray
    p_frac: np.ndarray
    l_frac: np.ndarray
    v_frac: np.ndarray
    abs_vfrac: np.ndarray
    PA_rad: np.ndarray
    PA_deg: np.ndarray
    EA_rad: np.ndarray
    EA_deg: np.ndarray
    dPA_dphi: np.ndarray
    pulse_off_pulse_std: np.ndarray
    pulse_threshold: np.ndarray
    pulse_L_true: np.ndarray
    pulse_P_true: np.ndarray
    pulse_p_frac: np.ndarray
    pulse_l_frac: np.ndarray
    pulse_v_frac: np.ndarray
    pulse_abs_vfrac: np.ndarray
    pulse_EA_rad: np.ndarray
    pulse_EA_deg: np.ndarray
    pulse_dPA_dphi: np.ndarray
    pulse_x: np.ndarray
    pulse_y: np.ndarray
    pulse_z: np.ndarray
    pulse_radius_of_curvature: np.ndarray
    mean_L: np.ndarray
    mean_L_sigma: np.ndarray
    mean_L_mask: np.ndarray
    mean_L_true: np.ndarray
    mean_P: np.ndarray
    mean_P_sigma: np.ndarray
    mean_P_mask: np.ndarray
    mean_P_true: np.ndarray
    mean_p_frac: np.ndarray
    mean_l_frac: np.ndarray
    mean_v_frac: np.ndarray
    mean_abs_vfrac: np.ndarray
    mean_PA_rad: np.ndarray
    mean_PA_deg: np.ndarray
    mean_EA_rad: np.ndarray
    mean_EA_deg: np.ndarray
    mean_dPA_dphi: np.ndarray
    mean_lon: np.ndarray
    mean_lat: np.ndarray
    mean_x: np.ndarray
    mean_y: np.ndarray
    mean_z: np.ndarray
    mean_radius_of_curvature: np.ndarray
    roc_phase: np.ndarray


def _validate_data(data):
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[1] < 4:
        raise ValueError("Expected data shape (num_pulses, 4, num_phase_bins)")
    return data


def get_pulse_energies(data_or_precomputed):
    params = precompute_stokes(data_or_precomputed) if isinstance(data_or_precomputed, np.ndarray) else data_or_precomputed
    return np.sum(params.I, axis=1)


def get_top_pulse_indices(data_or_precomputed, top_n):
    pulse_energies = get_pulse_energies(data_or_precomputed)
    top_n = min(max(int(top_n), 0), len(pulse_energies))
    if top_n == 0:
        return np.array([], dtype=int)
    return np.argsort(pulse_energies)[-top_n:][::-1]


def get_top_pulse_power_summary(data_or_precomputed, top_n=10):
    pulse_energies = get_pulse_energies(data_or_precomputed)
    top_indices = get_top_pulse_indices(data_or_precomputed, top_n)
    return [
        {
            "pulse_index": int(index),
            "pulse_number": int(index),
            "pulse_power": float(pulse_energies[index]),
        }
        for index in top_indices
    ]


def _safe_scalar(value):
    value = float(value)
    return value if np.isfinite(value) and abs(value) > EPSILON else EPSILON


def _phase_bounds(phase_axis, start_phase, end_phase, end_side="left"):
    start_idx = int(np.searchsorted(phase_axis, start_phase, side="left"))
    end_idx = int(np.searchsorted(phase_axis, end_phase, side=end_side))
    start_idx = max(0, min(start_idx, len(phase_axis)))
    end_idx = max(0, min(end_idx, len(phase_axis)))
    return start_idx, end_idx


def _debias_polarisation(amplitude, sigma_off, threshold=DEBIAS_THRESHOLD):
    """Debias polarisation amplitudes with broadcasting support."""
    sigma_off = np.maximum(np.asarray(sigma_off, dtype=float), EPSILON)
    sigma_ratio = amplitude / sigma_off
    return np.where(
        sigma_ratio >= threshold,
        sigma_off * np.sqrt(np.maximum(sigma_ratio ** 2 - 1, 0.0)),
        0.0,
    )


def _fraction(numerator, denominator, threshold):
    out = np.zeros_like(numerator, dtype=float)
    mask = (denominator >= threshold) & (denominator != 0)
    return np.divide(numerator, denominator, out=out, where=mask)


def _normalised_gradient(values, phase_axis, axis=-1, per_profile=False):
    if len(phase_axis) < 2:
        return np.zeros_like(values, dtype=float)

    grad = np.gradient(values, phase_axis, axis=axis)
    if per_profile:
        max_abs = np.nanmax(np.abs(grad), axis=axis, keepdims=True)
        return np.divide(grad, max_abs, out=np.zeros_like(grad), where=max_abs > 0)

    max_abs = np.nanmax(np.abs(grad))
    return grad / max_abs if max_abs > 0 else np.zeros_like(grad)


def _radius_of_curvature_from_xyz(x, y, z):
    points = np.stack((x, y, z), axis=-1)
    radius = np.full(points.shape[:-1], np.nan, dtype=float)
    if points.shape[-2] < 3:
        return radius

    norms = np.linalg.norm(points, axis=-1, keepdims=True)
    unit_points = np.divide(points, norms, out=np.zeros_like(points), where=norms > 0)
    p1 = unit_points[..., :-2, :]
    p2 = unit_points[..., 1:-1, :]
    p3 = unit_points[..., 2:, :]

    normal = np.cross(p2 - p1, p3 - p1)
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = normal_norm[..., 0] > 0
    unit_normal = np.divide(normal, normal_norm, out=np.zeros_like(normal), where=normal_norm > 0)
    d = np.abs(np.sum(p1 * unit_normal, axis=-1))
    d = np.clip(d, 0.0, 1.0)
    radius[..., 1:-1] = np.where(valid, np.sqrt(1.0 - d ** 2), np.nan)
    return radius


def precompute_stokes(data):
    """Precompute shape, phase, raw Stokes views, and mean profiles once per dataset."""
    data = _validate_data(data)
    num_pulses, _, num_bins = data.shape
    phase_axis = np.linspace(0, 1, num_bins)
    pulse_number = np.arange(num_pulses)
    I = data[:, 0, :]
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]
    mean_profiles = data[:, :4, :].mean(axis=0)
    I_mean, Q_mean, U_mean, V_mean = mean_profiles
    I0 = float(np.nanmax(np.abs(I_mean))) if I_mean.size else 0.0
    I0_safe = I0 if I0 > EPSILON else EPSILON

    return StokesPrecompute(
        data=data,
        num_pulses=num_pulses,
        num_bins=num_bins,
        phase_axis=phase_axis,
        pulse_number=pulse_number,
        I=I,
        Q=Q,
        U=U,
        V=V,
        I_mean=I_mean,
        Q_mean=Q_mean,
        U_mean=U_mean,
        V_mean=V_mean,
        mean_profiles=mean_profiles,
        I0=I0_safe,
        I_over_I0=I / I0_safe,
        I_mean_over_I0=I_mean / I0_safe,
    )


def _as_stokes_precompute(data_or_precomputed):
    if isinstance(data_or_precomputed, StokesPrecompute):
        return data_or_precomputed
    return precompute_stokes(data_or_precomputed)


def precompute_polarimetry(data_or_precomputed, on_pulse):
    """Precompute on-pulse-dependent derived arrays once and reuse slices later."""
    base = _as_stokes_precompute(data_or_precomputed)
    default_start, default_end = (float(on_pulse[0]), float(on_pulse[1]))
    on_pulse_mask = (base.phase_axis >= default_start) & (base.phase_axis <= default_end)
    off_pulse_mask = ~on_pulse_mask

    if np.any(off_pulse_mask):
        baseline_I = np.nanmean(base.I[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_Q = np.nanmean(base.Q[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_U = np.nanmean(base.U[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_V = np.nanmean(base.V[:, off_pulse_mask], axis=1, keepdims=True)
    else:
        baseline_I = np.zeros((base.num_pulses, 1), dtype=float)
        baseline_Q = np.zeros((base.num_pulses, 1), dtype=float)
        baseline_U = np.zeros((base.num_pulses, 1), dtype=float)
        baseline_V = np.zeros((base.num_pulses, 1), dtype=float)

    I = base.I - baseline_I
    Q = base.Q - baseline_Q
    U = base.U - baseline_U
    V = base.V - baseline_V
    mean_profiles = np.stack((I, Q, U, V), axis=1).mean(axis=0)
    I_mean, Q_mean, U_mean, V_mean = mean_profiles
    I0_scale = float(np.nanmax(np.abs(I_mean))) if I_mean.size else EPSILON
    I0_scale = I0_scale if I0_scale > EPSILON else EPSILON

    def _offpulse_sigma(values):
        if not np.any(off_pulse_mask):
            return EPSILON
        finite = np.asarray(values[:, off_pulse_mask], dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return EPSILON
        return _safe_scalar(np.nanstd(finite))

    sigma_I = _offpulse_sigma(I)
    sigma_Q = _offpulse_sigma(Q)
    sigma_U = _offpulse_sigma(U)
    sigma_V = _offpulse_sigma(V)
    sigma_L = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2))
    sigma_P = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2 + sigma_V ** 2))
    mean_sigma_I = sigma_I / np.sqrt(max(base.num_pulses, 1))
    mean_sigma_L = sigma_L / np.sqrt(max(base.num_pulses, 1))
    mean_sigma_P = sigma_P / np.sqrt(max(base.num_pulses, 1))
    off_pulse_std = sigma_I
    pulse_off_pulse_std = np.full((base.num_pulses, 1), sigma_I)

    L = np.sqrt(Q ** 2 + U ** 2)
    P = np.sqrt(Q ** 2 + U ** 2 + V ** 2)
    L_sigma = L / sigma_L
    P_sigma = P / sigma_P
    L_mask = L > SIGMA_VALIDITY_THRESHOLD * sigma_L
    P_mask = P > SIGMA_VALIDITY_THRESHOLD * sigma_P
    intensity_mask = np.abs(I) > SIGMA_VALIDITY_THRESHOLD * sigma_I
    L_true = _debias_polarisation(L, sigma_L)
    P_true = _debias_polarisation(P, sigma_P)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_frac_raw = np.divide(P_true, I, out=np.full_like(P_true, np.nan, dtype=float), where=np.abs(I) > EPSILON)
        l_frac_raw = np.divide(L_true, I, out=np.full_like(L_true, np.nan, dtype=float), where=np.abs(I) > EPSILON)
        v_frac_raw = np.divide(V, I, out=np.full_like(V, np.nan, dtype=float), where=np.abs(I) > EPSILON)
    p_frac = np.where(intensity_mask, p_frac_raw, np.nan)
    l_frac = np.where(intensity_mask, l_frac_raw, np.nan)
    v_frac = np.where(intensity_mask, v_frac_raw, np.nan)
    abs_vfrac = np.abs(v_frac)
    PA_rad_raw = 0.5 * np.arctan2(U, Q)
    EA_rad_raw = 0.5 * np.arctan2(V, L_true)
    PA_rad = np.where(L_mask, PA_rad_raw, np.nan)
    EA_rad = np.where(P_mask, EA_rad_raw, np.nan)
    PA_deg = np.degrees(PA_rad)
    EA_deg = np.degrees(EA_rad)
    dPA_dphi = _normalised_gradient(PA_deg, base.phase_axis, axis=-1)

    pulse_threshold = np.full((base.num_pulses, 1), SIGMA_VALIDITY_THRESHOLD * sigma_I)
    pulse_L_true = L_true
    pulse_P_true = P_true
    pulse_p_frac = p_frac
    pulse_l_frac = l_frac
    pulse_v_frac = v_frac
    pulse_abs_vfrac = abs_vfrac
    pulse_EA_rad = EA_rad
    pulse_EA_deg = EA_deg
    pulse_dPA_dphi = _normalised_gradient(PA_deg, base.phase_axis, axis=-1, per_profile=True)
    pulse_lon = 2 * PA_rad
    pulse_lat = 2 * EA_rad
    pulse_cos_lat = np.cos(pulse_lat)
    pulse_x = pulse_cos_lat * np.cos(pulse_lon)
    pulse_y = pulse_cos_lat * np.sin(pulse_lon)
    pulse_z = np.sin(pulse_lat)
    pulse_radius_of_curvature = _radius_of_curvature_from_xyz(pulse_x, pulse_y, pulse_z)

    mean_L = np.sqrt(Q_mean ** 2 + U_mean ** 2)
    mean_P = np.sqrt(Q_mean ** 2 + U_mean ** 2 + V_mean ** 2)
    mean_L_sigma = mean_L / mean_sigma_L
    mean_P_sigma = mean_P / mean_sigma_P
    mean_L_mask = mean_L > SIGMA_VALIDITY_THRESHOLD * mean_sigma_L
    mean_P_mask = mean_P > SIGMA_VALIDITY_THRESHOLD * mean_sigma_P
    mean_intensity_mask = np.abs(I_mean) > SIGMA_VALIDITY_THRESHOLD * mean_sigma_I
    mean_L_true = _debias_polarisation(mean_L, mean_sigma_L)
    mean_P_true = _debias_polarisation(mean_P, mean_sigma_P)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_p_frac_raw = np.divide(mean_P_true, I_mean, out=np.full_like(mean_P_true, np.nan, dtype=float), where=np.abs(I_mean) > EPSILON)
        mean_l_frac_raw = np.divide(mean_L_true, I_mean, out=np.full_like(mean_L_true, np.nan, dtype=float), where=np.abs(I_mean) > EPSILON)
        mean_v_frac_raw = np.divide(V_mean, I_mean, out=np.full_like(V_mean, np.nan, dtype=float), where=np.abs(I_mean) > EPSILON)
    mean_p_frac = np.where(mean_intensity_mask, mean_p_frac_raw, np.nan)
    mean_l_frac = np.where(mean_intensity_mask, mean_l_frac_raw, np.nan)
    mean_v_frac = np.where(mean_intensity_mask, mean_v_frac_raw, np.nan)
    mean_abs_vfrac = np.abs(mean_v_frac)
    mean_PA_rad_raw = 0.5 * np.arctan2(U_mean, Q_mean)
    mean_EA_rad_raw = 0.5 * np.arctan2(V_mean, mean_L_true)
    mean_PA_rad = np.where(mean_L_mask, mean_PA_rad_raw, np.nan)
    mean_EA_rad = np.where(mean_P_mask, mean_EA_rad_raw, np.nan)
    mean_PA_deg = np.degrees(mean_PA_rad)
    mean_EA_deg = np.degrees(mean_EA_rad)
    mean_dPA_dphi = _normalised_gradient(mean_PA_deg, base.phase_axis)
    mean_lon = 2 * mean_PA_rad
    mean_lat = 2 * mean_EA_rad
    mean_cos_lat = np.cos(mean_lat)
    mean_x = mean_cos_lat * np.cos(mean_lon)
    mean_y = mean_cos_lat * np.sin(mean_lon)
    mean_z = np.sin(mean_lat)
    mean_radius_of_curvature = _radius_of_curvature_from_xyz(mean_x, mean_y, mean_z)

    return PolarimetryPrecompute(
        data=base.data,
        num_pulses=base.num_pulses,
        num_bins=base.num_bins,
        phase_axis=base.phase_axis,
        pulse_number=base.pulse_number,
        I=I,
        Q=Q,
        U=U,
        V=V,
        I_mean=I_mean,
        Q_mean=Q_mean,
        U_mean=U_mean,
        V_mean=V_mean,
        mean_profiles=mean_profiles,
        I0=I0_scale,
        I_over_I0=I / I0_scale,
        I_mean_over_I0=I_mean / I0_scale,
        on_pulse=(default_start, default_end),
        on_pulse_mask=on_pulse_mask,
        off_pulse_mask=off_pulse_mask,
        off_pulse_std=off_pulse_std,
        threshold=SIGMA_VALIDITY_THRESHOLD * sigma_I,
        L=L,
        L_sigma=L_sigma,
        L_mask=L_mask,
        L_true=L_true,
        P=P,
        P_sigma=P_sigma,
        P_mask=P_mask,
        P_true=P_true,
        p_frac=p_frac,
        l_frac=l_frac,
        v_frac=v_frac,
        abs_vfrac=abs_vfrac,
        PA_rad=PA_rad,
        PA_deg=PA_deg,
        EA_rad=EA_rad,
        EA_deg=EA_deg,
        dPA_dphi=dPA_dphi,
        pulse_off_pulse_std=pulse_off_pulse_std,
        pulse_threshold=pulse_threshold,
        pulse_L_true=pulse_L_true,
        pulse_P_true=pulse_P_true,
        pulse_p_frac=pulse_p_frac,
        pulse_l_frac=pulse_l_frac,
        pulse_v_frac=pulse_v_frac,
        pulse_abs_vfrac=pulse_abs_vfrac,
        pulse_EA_rad=pulse_EA_rad,
        pulse_EA_deg=pulse_EA_deg,
        pulse_dPA_dphi=pulse_dPA_dphi,
        pulse_x=pulse_x,
        pulse_y=pulse_y,
        pulse_z=pulse_z,
        pulse_radius_of_curvature=pulse_radius_of_curvature,
        mean_L=mean_L,
        mean_L_sigma=mean_L_sigma,
        mean_L_mask=mean_L_mask,
        mean_L_true=mean_L_true,
        mean_P=mean_P,
        mean_P_sigma=mean_P_sigma,
        mean_P_mask=mean_P_mask,
        mean_P_true=mean_P_true,
        mean_p_frac=mean_p_frac,
        mean_l_frac=mean_l_frac,
        mean_v_frac=mean_v_frac,
        mean_abs_vfrac=mean_abs_vfrac,
        mean_PA_rad=mean_PA_rad,
        mean_PA_deg=mean_PA_deg,
        mean_EA_rad=mean_EA_rad,
        mean_EA_deg=mean_EA_deg,
        mean_dPA_dphi=mean_dPA_dphi,
        mean_lon=mean_lon,
        mean_lat=mean_lat,
        mean_x=mean_x,
        mean_y=mean_y,
        mean_z=mean_z,
        mean_radius_of_curvature=mean_radius_of_curvature,
        roc_phase=base.phase_axis,
    )


def _as_polarimetry_precompute(data_or_precomputed, on_pulse):
    if isinstance(data_or_precomputed, PolarimetryPrecompute):
        return data_or_precomputed
    return precompute_polarimetry(data_or_precomputed, on_pulse)


def compute_common_stokes_params(data, phase_axis=None, on_pulse=None):
    """Return cached/common arrays in the legacy dictionary shape."""
    if isinstance(data, PolarimetryPrecompute):
        params = data
    else:
        if on_pulse is None:
            raise ValueError("on_pulse is required when raw data is provided")
        params = precompute_polarimetry(data, on_pulse)

    return {
        "I": params.I,
        "Q": params.Q,
        "U": params.U,
        "V": params.V,
        "I_mean": params.I_mean,
        "Q_mean": params.Q_mean,
        "U_mean": params.U_mean,
        "V_mean": params.V_mean,
        "I_over_I0": params.I_over_I0,
        "I_mean_over_I0": params.I_mean_over_I0,
        "L": params.L,
        "L_sigma": params.L_sigma,
        "L_mask": params.L_mask,
        "L_true": params.L_true,
        "P": params.P,
        "P_sigma": params.P_sigma,
        "P_mask": params.P_mask,
        "P_true": params.P_true,
        "p_frac": params.p_frac,
        "l_frac": params.l_frac,
        "v_frac": params.v_frac,
        "abs_vfrac": params.abs_vfrac,
        "PA": params.PA_deg,
        "EA": params.EA_deg,
        "dPA": params.dPA_dphi,
        "off_pulse_std": params.off_pulse_std,
        "threshold": params.threshold,
        "on_pulse_mask": params.on_pulse_mask,
        "off_pulse_mask": params.off_pulse_mask,
    }

def return_xyz_interactive_poincare_sphere(data, start_phase, end_phase, on_pulse, obs_id):
    params = _as_polarimetry_precompute(data, on_pulse)
    start_idx, end_idx = _phase_bounds(params.phase_axis, start_phase, end_phase)
    return (
        params.mean_x[start_idx:end_idx],
        params.mean_y[start_idx:end_idx],
        params.mean_z[start_idx:end_idx],
    )

def get_all_profiles(data, start_phase, end_phase):
    """Get all 4 Stokes profiles efficiently in one call"""
    params = _as_stokes_precompute(data)
    return {
        label: {"x": params.phase_axis, "y": params.mean_profiles[idx]}
        for idx, label in enumerate(STOKES_LABELS)
    }

def get_I_profile(data, start_phase, end_phase):
    params = _as_stokes_precompute(data)
    return params.phase_axis, params.I_mean

def get_Q_profile(data, start_phase, end_phase):
    params = _as_stokes_precompute(data)
    return params.phase_axis, params.Q_mean

def get_U_profile(data, start_phase, end_phase):
    params = _as_stokes_precompute(data)
    return params.phase_axis, params.U_mean

def get_V_profile(data, start_phase, end_phase):
    params = _as_stokes_precompute(data)
    return params.phase_axis, params.V_mean

# Unified function to compute all heatmaps in one pass
def plot_all_heatmaps(data, start_phase, end_phase, obs_id):
    """Compute all four Stokes heatmaps (I, Q, U, V) efficiently in a single pass."""
    params = _as_stokes_precompute(data)
    start_idx, end_idx = _phase_bounds(params.phase_axis, start_phase, end_phase, end_side="right")
    phase_slice = params.phase_axis[start_idx:end_idx]

    heatmaps = {}
    for stokes_idx, label in enumerate(STOKES_LABELS):
        heatmap_data = params.data[:, stokes_idx, start_idx:end_idx]
        vmin = heatmap_data.min()
        vmax = heatmap_data.max()
        
        heatmaps[label] = {
            'pulse_phase': phase_slice,
            'pulse_number': params.pulse_number,
            'heatmap_data': heatmap_data,
            'vmin': float(vmin),
            'vmax': float(vmax),
            'label': label,
            'obs_id': obs_id
        }
    
    return heatmaps


def _select_even_indices(length, max_count):
    if length <= max_count:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_count).astype(int))


def _preprocess_intensity_stack(I, offpulse_mask=None, remove_mean_profile=True, subtract_offpulse=True):
    """Return an intensity stack prepared for pulse-number FFT diagnostics."""
    D = np.asarray(I, dtype=float).copy()
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    if subtract_offpulse and offpulse_mask is not None and np.any(offpulse_mask):
        baseline = np.mean(D[:, offpulse_mask], axis=1)
        D = D - baseline[:, None]

    if remove_mean_profile:
        mean_profile = np.mean(D, axis=0)
        D = D - mean_profile[None, :]

    return D


def _compute_modulation_index(I, offpulse_mask):
    D = np.asarray(I, dtype=float).copy()
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    if offpulse_mask is not None and np.any(offpulse_mask):
        baseline = np.mean(D[:, offpulse_mask], axis=1)
        D = D - baseline[:, None]

    mean_profile = np.mean(D, axis=0)
    var_profile = np.var(D, axis=0, ddof=1) if D.shape[0] > 1 else np.zeros(D.shape[1])
    if offpulse_mask is not None and np.any(offpulse_mask) and D.shape[0] > 1:
        noise_var = float(np.mean(np.var(D[:, offpulse_mask], axis=0, ddof=1)))
        intrinsic_var = np.maximum(var_profile - noise_var, 0.0)
    else:
        intrinsic_var = var_profile

    denominator = np.where(np.abs(mean_profile) > EPSILON, mean_profile, np.nan)
    modulation = np.sqrt(intrinsic_var) / denominator
    modulation = np.where((mean_profile > EPSILON) & np.isfinite(modulation), modulation, np.nan)
    return modulation, mean_profile, intrinsic_var


def _compute_lrfs(I, offpulse_mask=None, remove_mean_profile=True):
    D = _preprocess_intensity_stack(I, offpulse_mask, remove_mean_profile=remove_mean_profile)
    n_pulses = D.shape[0]
    if n_pulses > 1:
        D = D * np.hanning(n_pulses)[:, None]
    spectrum = np.fft.rfft(D, axis=0)
    power = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(n_pulses, d=1.0)
    return freq, power


def _compute_lrfs_complex(I, offpulse_mask=None, remove_mean_profile=True):
    D = _preprocess_intensity_stack(I, offpulse_mask, remove_mean_profile=remove_mean_profile)
    n_pulses = D.shape[0]
    if n_pulses > 1:
        D = D * np.hanning(n_pulses)[:, None]
    spectrum = np.fft.rfft(D, axis=0)
    power = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(n_pulses, d=1.0)
    return freq, power, spectrum


def _compute_2dfs(I, offpulse_mask=None):
    D = _preprocess_intensity_stack(I, offpulse_mask, remove_mean_profile=True)
    n_pulses, n_phase = D.shape
    if n_pulses > 1:
        D = D * np.hanning(n_pulses)[:, None]
    if n_phase > 1:
        D = D * np.hanning(n_phase)[None, :]
    spectrum = np.fft.fft2(D)
    power = np.abs(np.fft.fftshift(spectrum, axes=(0, 1))) ** 2
    f3 = np.fft.fftshift(np.fft.fftfreq(n_pulses, d=1.0))
    f2 = np.fft.fftshift(np.fft.fftfreq(n_phase, d=1.0)) * n_phase
    return f3, f2, power


def _compute_sliding_lrfs(I, offpulse_mask=None, window_size=None, step=None):
    n_pulses, _ = I.shape
    if n_pulses < 4:
        return np.array([]), np.array([]), np.zeros((0, 0))

    if window_size is None:
        window_size = int(np.clip(2 ** int(np.floor(np.log2(max(4, n_pulses // 4)))), 16, min(256, n_pulses)))
    window_size = int(np.clip(window_size, 4, n_pulses))
    if step is None:
        step = max(1, window_size // 16)

    centers = []
    spectra = []
    freq = None
    for start in range(0, n_pulses - window_size + 1, step):
        block = I[start:start + window_size, :]
        local_freq, local_power = _compute_lrfs(block, offpulse_mask, remove_mean_profile=True)
        centers.append(start + window_size / 2)
        spectra.append(np.sum(local_power, axis=1))
        freq = local_freq

    if not spectra:
        return np.array([]), np.array([]), np.zeros((0, 0))
    return np.asarray(centers), np.asarray(freq), np.asarray(spectra)


def _contiguous_true_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    regions = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append((start, index))
            start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions


def _matched_offpulse_energy_reference(I_baseline_subtracted, offpulse_mask, width):
    if offpulse_mask is None or not np.any(offpulse_mask) or width <= 0:
        return np.array([]), np.array([])

    references = []
    peak_references = []
    for start, end in _contiguous_true_regions(offpulse_mask):
        segment_width = end - start
        if segment_width < width:
            continue
        for window_start in range(start, end - width + 1, width):
            window = I_baseline_subtracted[:, window_start:window_start + width]
            references.append(np.sum(window, axis=1))
            peak_references.append(np.max(window, axis=1))

    if not references:
        off_indices = np.where(offpulse_mask)[0]
        if off_indices.size >= width:
            window = I_baseline_subtracted[:, off_indices[:width]]
            references.append(np.sum(window, axis=1))
            peak_references.append(np.max(window, axis=1))

    if not references:
        return np.array([]), np.array([])
    return np.concatenate(references), np.concatenate(peak_references)


def _compute_sliding_2dfs(I, offpulse_mask=None, window_size=None, step=None):
    n_pulses, _ = I.shape
    if n_pulses < 4:
        return {
            "centers": np.array([]),
            "f3": np.array([]),
            "f2": np.array([]),
            "P3": np.array([]),
            "P2_bins": np.array([]),
            "drift_direction": np.array([]),
            "peak_power": np.array([]),
        }

    if window_size is None:
        window_size = int(np.clip(2 ** int(np.floor(np.log2(max(4, n_pulses // 4)))), 16, min(256, n_pulses)))
    window_size = int(np.clip(window_size, 4, n_pulses))
    if step is None:
        step = max(1, window_size // 8)

    centers = []
    f3_values = []
    f2_values = []
    p3_values = []
    p2_values = []
    drift_values = []
    peak_values = []

    for start in range(0, n_pulses - window_size + 1, step):
        block = I[start:start + window_size, :]
        f3, f2, power = _compute_2dfs(block, offpulse_mask)
        estimate = _estimate_p2_p3(f3, f2, power)
        f3_peak = estimate["f3"]
        f2_peak = estimate["f2"]
        centers.append(start + window_size / 2)
        f3_values.append(f3_peak if f3_peak is not None else np.nan)
        f2_values.append(f2_peak if f2_peak is not None else np.nan)
        p3_values.append(estimate["P3"] if estimate["P3"] is not None else np.nan)
        p2_values.append(estimate["P2_bins"] if estimate["P2_bins"] is not None else np.nan)
        drift_values.append(estimate["drift_direction"])

        if f3_peak is None or f2_peak is None:
            peak_values.append(np.nan)
        else:
            f3_index = int(np.argmin(np.abs(f3 - f3_peak)))
            f2_index = int(np.argmin(np.abs(f2 - f2_peak)))
            peak_values.append(float(power[f3_index, f2_index]))

    return {
        "centers": np.asarray(centers),
        "f3": np.asarray(f3_values),
        "f2": np.asarray(f2_values),
        "P3": np.asarray(p3_values),
        "P2_bins": np.asarray(p2_values),
        "drift_direction": np.asarray(drift_values),
        "peak_power": np.asarray(peak_values),
    }


def _estimate_p2_p3(f3, f2, power):
    if power.size == 0:
        return {"f3": None, "f2": None, "P3": None, "P2_bins": None, "drift_direction": 0}

    f3_grid = f3[:, None]
    f2_grid = f2[None, :]
    mask = (np.abs(f3_grid) >= 0.01) & (np.abs(f3_grid) <= 0.5) & (np.abs(f2_grid) > EPSILON)
    masked = np.where(mask, power, np.nan)
    if not np.isfinite(masked).any():
        return {"f3": None, "f2": None, "P3": None, "P2_bins": None, "drift_direction": 0}

    peak_i, peak_j = np.unravel_index(np.nanargmax(masked), masked.shape)
    f3_peak = float(f3[peak_i])
    f2_peak = float(f2[peak_j])
    return {
        "f3": f3_peak,
        "f2": f2_peak,
        "P3": float(1.0 / abs(f3_peak)) if abs(f3_peak) > EPSILON else None,
        "P2_bins": float(1.0 / abs(f2_peak)) if abs(f2_peak) > EPSILON else None,
        "drift_direction": int(np.sign(f2_peak)),
    }


def _log_power(power):
    power = np.asarray(power, dtype=float)
    log_power = np.full_like(power, np.nan, dtype=float)
    mask = np.isfinite(power) & (power > 0)
    log_power[mask] = np.log10(power[mask])
    return log_power


def _lrfs_local_peak_candidates(freq, integrated_power, max_candidates=8, context_bins=5):
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(integrated_power, dtype=float)
    valid = np.isfinite(freq) & np.isfinite(power) & (freq > EPSILON) & (power > 0)
    if freq.size < 3 or not np.any(valid):
        return []

    candidates = []
    for index in range(1, len(freq) - 1):
        if not valid[index]:
            continue
        if not (power[index] >= power[index - 1] and power[index] >= power[index + 1]):
            continue
        left = max(0, index - context_bins)
        right = min(len(power), index + context_bins + 1)
        context = np.concatenate((power[left:index], power[index + 1:right]))
        context = context[np.isfinite(context) & (context > 0)]
        local_level = float(np.nanmedian(context)) if context.size else 0.0
        prominence = float(power[index] - local_level)
        contrast = float(power[index] / local_level) if local_level > EPSILON else None
        candidates.append({
            "frequency": float(freq[index]),
            "P3": float(1.0 / freq[index]),
            "power": float(power[index]),
            "prominence": prominence,
            "local_contrast": contrast,
            "index": int(index),
        })

    if not candidates:
        top_indices = np.flatnonzero(valid)
        top_indices = sorted(top_indices, key=lambda idx: power[idx], reverse=True)[:max_candidates]
        candidates = [
            {
                "frequency": float(freq[index]),
                "P3": float(1.0 / freq[index]),
                "power": float(power[index]),
                "prominence": None,
                "local_contrast": None,
                "index": int(index),
            }
            for index in top_indices
        ]
    else:
        candidates.sort(
            key=lambda item: (
                -np.inf if item["prominence"] is None else item["prominence"],
                item["power"],
            ),
            reverse=True,
        )

    result = []
    for rank, candidate in enumerate(candidates[:max_candidates], start=1):
        result.append({
            "rank": rank,
            "frequency": _finite_or_none(candidate["frequency"]),
            "P3": _finite_or_none(candidate["P3"]),
            "power": _finite_or_none(candidate["power"]),
            "prominence": _finite_or_none(candidate["prominence"]) if candidate["prominence"] is not None else None,
            "local_contrast": _finite_or_none(candidate["local_contrast"]) if candidate["local_contrast"] is not None else None,
            "index": int(candidate["index"]),
        })
    return result


def _log_counts(counts):
    counts = np.asarray(counts, dtype=float)
    return np.log1p(np.maximum(counts, 0.0))


def _freedman_diaconis_bin_count(values, min_bins=20, max_bins=160):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return min_bins

    value_range = float(np.max(values) - np.min(values))
    if value_range <= EPSILON:
        return min_bins

    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr <= EPSILON:
        fallback = int(np.sqrt(values.size))
        return int(np.clip(fallback, min_bins, max_bins))

    bin_width = 2 * iqr / np.cbrt(values.size)
    if bin_width <= EPSILON:
        fallback = int(np.sqrt(values.size))
        return int(np.clip(fallback, min_bins, max_bins))

    return int(np.clip(np.ceil(value_range / bin_width), min_bins, max_bins))


def _distribution_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"size": int(values.size), "finite": 0, "min": None, "max": None, "mean": None}
    return {
        "size": int(values.size),
        "finite": int(finite.size),
        "min": _finite_or_none(np.min(finite)),
        "max": _finite_or_none(np.max(finite)),
        "mean": _finite_or_none(np.mean(finite)),
    }


def _profile_stabilisation_diagnostics(I_baseline_subtracted):
    I_baseline_subtracted = np.asarray(I_baseline_subtracted, dtype=float)
    n_pulses, n_phase = I_baseline_subtracted.shape
    if n_pulses < 2 or n_phase < 2:
        return {"pulse_count": np.array([]), "correlation": np.array([]), "one_minus_correlation": np.array([]), "reference": np.array([])}

    final_profile = np.nanmean(I_baseline_subtracted, axis=0)
    final_centered = final_profile - np.nanmean(final_profile)
    final_norm = np.sqrt(np.nansum(final_centered ** 2))
    if not np.isfinite(final_norm) or final_norm <= EPSILON:
        return {"pulse_count": np.array([]), "correlation": np.array([]), "one_minus_correlation": np.array([]), "reference": np.array([])}

    pulse_counts = np.unique(np.clip(np.round(np.geomspace(1, n_pulses, min(80, n_pulses))).astype(int), 1, n_pulses))
    correlations = []
    for count in pulse_counts:
        profile = np.nanmean(I_baseline_subtracted[:count, :], axis=0)
        centered = profile - np.nanmean(profile)
        norm = np.sqrt(np.nansum(centered ** 2))
        correlations.append(float(np.nansum(centered * final_centered) / (norm * final_norm)) if np.isfinite(norm) and norm > EPSILON else np.nan)
    correlations = np.asarray(correlations, dtype=float)
    one_minus = 1.0 - correlations
    valid = np.flatnonzero(np.isfinite(one_minus) & (one_minus > 0))
    scale = float(one_minus[valid[0]] * np.sqrt(pulse_counts[valid[0]])) if valid.size else np.nan
    reference = scale / np.sqrt(pulse_counts) if np.isfinite(scale) else np.full_like(pulse_counts, np.nan, dtype=float)
    return {"pulse_count": pulse_counts, "correlation": correlations, "one_minus_correlation": one_minus, "reference": reference}


def _acf_psd_diagnostics(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return {"lag": np.array([]), "acf": np.array([]), "frequency": np.array([]), "psd": np.array([])}
    series = values.copy()
    series[~np.isfinite(series)] = float(np.nanmean(finite))
    series = series - np.mean(series)
    n = series.size
    variance = float(np.dot(series, series))
    if variance <= EPSILON:
        return {"lag": np.array([]), "acf": np.array([]), "frequency": np.array([]), "psd": np.array([])}
    corr = np.correlate(series, series, mode="full")[n - 1:]
    norm = variance * np.maximum(n - np.arange(n), 1) / n
    acf = corr / np.maximum(norm, EPSILON)
    spectrum = np.fft.rfft(series * np.hanning(n))
    psd = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(n, d=1.0)
    return {"lag": np.arange(n), "acf": acf, "frequency": freq, "psd": psd}


def _trial_null_fraction_diagnostics(on_energy, off_energy):
    on = np.asarray(on_energy, dtype=float)
    off = np.asarray(off_energy, dtype=float)
    on = on[np.isfinite(on)]
    off = off[np.isfinite(off)]
    if on.size == 0 or off.size == 0:
        return {"threshold_sigma": np.array([]), "null_fraction": np.array([]), "default_threshold_sigma": 3.0, "default_null_fraction": np.nan, "off_rms": np.nan}
    off_rms = float(np.sqrt(np.nanmean(off ** 2)))
    if not np.isfinite(off_rms) or off_rms <= EPSILON:
        off_rms = float(np.nanstd(off)) if off.size > 1 else 1.0
    if not np.isfinite(off_rms) or off_rms <= EPSILON:
        off_rms = 1.0
    thresholds = np.linspace(-3.0, 6.0, 91)
    fractions = np.asarray([np.mean(on <= threshold * off_rms) for threshold in thresholds], dtype=float)
    default_threshold = 3.0
    return {
        "threshold_sigma": thresholds,
        "null_fraction": fractions,
        "default_threshold_sigma": default_threshold,
        "default_null_fraction": float(np.mean(on <= default_threshold * off_rms)),
        "off_rms": off_rms,
    }


def _adp_diagnostics(I_baseline_subtracted, max_lag=80):
    I_baseline_subtracted = np.asarray(I_baseline_subtracted, dtype=float)
    n_pulses, n_phase = I_baseline_subtracted.shape
    if n_pulses < 3 or n_phase < 2:
        return {"phase_lag_bins": np.array([]), "correlation": np.array([])}
    data = I_baseline_subtracted.copy()
    data = data - np.nanmean(data, axis=0, keepdims=True)
    data[~np.isfinite(data)] = 0.0
    max_lag = int(min(max_lag, n_phase - 1))
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        shifted = np.roll(data[1:, :], shift=lag, axis=1)
        base = data[:-1, :]
        denom = np.sqrt(np.sum(base ** 2) * np.sum(shifted ** 2))
        correlations.append(float(np.sum(base * shifted) / denom) if denom > EPSILON else np.nan)
    return {"phase_lag_bins": lags, "correlation": np.asarray(correlations, dtype=float)}


def total_intensity_evolution(data, start_phase, end_phase, on_pulse, obs_id, default_bins=160, normalization="energy_mean_on"):
    """Build JSON-ready Stokes-I single-pulse diagnostics for the selected phase window."""
    params = _as_stokes_precompute(data)
    phase_axis = params.phase_axis
    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase, end_side="right")
    selected_bins_were_empty = end_idx <= start_idx
    if end_idx <= start_idx:
        end_idx = min(len(phase_axis), start_idx + 1)

    selected_phase = phase_axis[start_idx:end_idx]
    I = params.I[:, start_idx:end_idx]
    full_on_mask = (phase_axis >= float(on_pulse[0])) & (phase_axis <= float(on_pulse[1]))
    offpulse_mask = ~full_on_mask
    selected_offpulse_mask = offpulse_mask[start_idx:end_idx] if len(offpulse_mask) else None
    if selected_offpulse_mask is not None and not np.any(selected_offpulse_mask):
        selected_offpulse_mask = None

    full_I = np.asarray(params.I, dtype=float)
    distribution_warnings = []
    intensity_warnings = []
    if selected_bins_were_empty:
        intensity_warnings.append("No phase bins selected; check start_phase/end_phase.")
    if np.sum(offpulse_mask) < 5:
        distribution_warnings.append("Not enough off-pulse bins for baseline subtraction.")
        intensity_warnings.append("Not enough off-pulse bins for baseline subtraction.")
        pulse_baseline = np.zeros(full_I.shape[0], dtype=float)
    else:
        pulse_baseline = np.nanmean(full_I[:, offpulse_mask], axis=1)
    full_I_baseline_subtracted = full_I - pulse_baseline[:, None]
    I_baseline_subtracted = full_I_baseline_subtracted[:, start_idx:end_idx]

    mean_profile = np.nanmean(I_baseline_subtracted, axis=0)
    mean_profile_full = np.nanmean(full_I_baseline_subtracted, axis=0)
    active_profile = mean_profile_full[full_on_mask] if np.any(full_on_mask) else mean_profile_full
    finite_mean_profile = active_profile[np.isfinite(active_profile)]
    mean_profile_peak = float(np.nanmax(finite_mean_profile)) if finite_mean_profile.size else 1.0
    if not np.isfinite(mean_profile_peak) or mean_profile_peak <= EPSILON:
        distribution_warnings.append("Invalid mean profile peak for profile-peak normalization.")
        mean_profile_peak = 1.0
    pulse_energies = np.sum(I_baseline_subtracted, axis=1)
    pulse_peak_intensity = np.max(I_baseline_subtracted, axis=1)
    off_energy_reference, off_peak_intensity_reference = _matched_offpulse_energy_reference(
        full_I_baseline_subtracted,
        offpulse_mask,
        I_baseline_subtracted.shape[1],
    )

    normalization = str(normalization or "energy_mean_on").strip().lower()
    legacy_normalization = {
        "mean_on": "energy_mean_on",
        "offpulse_rms": "energy_off_rms",
        "peak_intensity": "peak_i_over_mean_profile_peak",
    }
    normalization = legacy_normalization.get(normalization, normalization)
    valid_normalizations = {
        "energy_mean_on",
        "energy_off_rms",
        "peak_i_over_mean_profile_peak",
    }
    if normalization not in valid_normalizations:
        normalization = "energy_mean_on"

    if normalization == "peak_i_over_mean_profile_peak":
        on_distribution_values = pulse_peak_intensity
        off_distribution_values = off_peak_intensity_reference
    else:
        on_distribution_values = pulse_energies
        off_distribution_values = off_energy_reference

    finite_on_values = on_distribution_values[np.isfinite(on_distribution_values)]
    finite_off_values = off_distribution_values[np.isfinite(off_distribution_values)]
    if normalization.endswith("mean_profile_peak"):
        normalization_factor = mean_profile_peak
    elif normalization.endswith("mean_on"):
        normalization_factor = float(np.mean(finite_on_values)) if finite_on_values.size else 1.0
    elif normalization.endswith("off_rms"):
        normalization_factor = float(np.sqrt(np.mean(finite_off_values ** 2))) if finite_off_values.size else 1.0
    if not np.isfinite(normalization_factor) or normalization_factor <= EPSILON:
        distribution_warnings.append("Invalid normalization scale for this mode.")
        normalization_factor = 1.0

    plot_pulse_energies = on_distribution_values / normalization_factor
    plot_off_energy_reference = off_distribution_values / normalization_factor
    finite_plot_pulse_energies = plot_pulse_energies[np.isfinite(plot_pulse_energies)]
    finite_plot_off_energy_reference = plot_off_energy_reference[np.isfinite(plot_off_energy_reference)]
    if finite_plot_pulse_energies.size == 0:
        distribution_warnings.append("No finite on-pulse values for this mode.")
    if finite_plot_off_energy_reference.size == 0:
        distribution_warnings.append("No finite off-pulse reference values for this mode.")

    energy_values = finite_plot_pulse_energies
    if energy_values.size:
        energy_bin_count = _freedman_diaconis_bin_count(energy_values)
        energy_counts, energy_edges = np.histogram(energy_values, bins=energy_bin_count, density=True)
    else:
        energy_bin_count = 20
        energy_counts, energy_edges = np.histogram([0.0], bins=20, density=True)
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    intensity_values = I_baseline_subtracted[np.isfinite(I_baseline_subtracted)]
    if intensity_values.size:
        vmin = float(np.nanmin(intensity_values))
        vmax = float(np.nanmax(intensity_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            intensity_warnings.append("No finite Stokes I values available for histogram.")
            vmin, vmax = 0.0, 1.0
        outside_display_range = 0.0
    else:
        intensity_warnings.append("No finite Stokes I values available for histogram.")
        vmin, vmax = 0.0, 1.0
        outside_display_range = 0.0
    quantity_bins = int(np.clip(default_bins, 50, 300))
    hist2d = np.zeros((quantity_bins, I.shape[1]))
    bin_edges = np.linspace(vmin, vmax, quantity_bins + 1)
    for phase_i in range(I.shape[1]):
        values = I_baseline_subtracted[:, phase_i]
        values = values[np.isfinite(values)]
        hist2d[:, phase_i], _ = np.histogram(values, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    log10_hist2d = np.full_like(hist2d, np.nan, dtype=float)
    positive_count_mask = hist2d > 0
    log10_hist2d[positive_count_mask] = np.log10(hist2d[positive_count_mask])

    fluc_warnings = []
    if selected_bins_were_empty:
        fluc_warnings.append("No selected phase bins.")
    if params.num_pulses < 4:
        fluc_warnings.append("Not enough pulses for fluctuation spectrum.")
    if np.sum(offpulse_mask) < 5:
        fluc_warnings.append("Not enough off-pulse bins for baseline subtraction and noise estimate.")

    profile_for_modulation = mean_profile
    if I_baseline_subtracted.shape[0] > 1:
        var_profile = np.nanvar(I_baseline_subtracted, axis=0, ddof=1)
    else:
        var_profile = np.zeros(I_baseline_subtracted.shape[1])
    if np.sum(offpulse_mask) >= 5 and full_I_baseline_subtracted.shape[0] > 1:
        noise_var = float(np.nanmean(np.nanvar(full_I_baseline_subtracted[:, offpulse_mask], axis=0, ddof=1)))
    else:
        noise_var = 0.0
    intrinsic_var = np.maximum(var_profile - noise_var, 0.0)
    modulation = np.full_like(profile_for_modulation, np.nan, dtype=float)
    selected_on_mask = full_on_mask[start_idx:end_idx]
    modulation_valid = selected_on_mask & np.isfinite(profile_for_modulation) & (profile_for_modulation > EPSILON)
    modulation[modulation_valid] = np.sqrt(intrinsic_var[modulation_valid]) / profile_for_modulation[modulation_valid]

    lrfs_freq, lrfs_power, lrfs_spectrum = _compute_lrfs_complex(I_baseline_subtracted, None, remove_mean_profile=True)
    positive_lrfs_mask = lrfs_freq > 0
    lrfs_freq = lrfs_freq[positive_lrfs_mask]
    lrfs_power = lrfs_power[positive_lrfs_mask]
    lrfs_spectrum = lrfs_spectrum[positive_lrfs_mask]
    integrated_lrfs_power = np.nansum(lrfs_power, axis=1) if lrfs_power.size else np.array([])
    lrfs_p3_estimate = None
    lrfs_f_peak = None
    lrfs_previous_p3_estimate = None
    lrfs_previous_f_peak = None
    lrfs_peak_phase = np.full(I_baseline_subtracted.shape[1], np.nan, dtype=float)
    if lrfs_freq.size and integrated_lrfs_power.size:
        absolute_mask = np.isfinite(integrated_lrfs_power) & (lrfs_freq > EPSILON)
        if np.any(absolute_mask):
            local_freq = lrfs_freq[absolute_mask]
            local_power = integrated_lrfs_power[absolute_mask]
            peak = int(np.nanargmax(local_power))
            lrfs_f_peak = float(local_freq[peak])
            lrfs_p3_estimate = float(1.0 / lrfs_f_peak) if lrfs_f_peak > EPSILON else None
            peak_indices = np.flatnonzero(absolute_mask)
            if peak_indices.size and lrfs_spectrum.size:
                peak_row = lrfs_spectrum[int(peak_indices[peak])]
                lrfs_peak_phase = np.degrees(np.angle(peak_row))
        previous_mask = (lrfs_freq >= 0.04) & (lrfs_freq <= 0.5) & np.isfinite(integrated_lrfs_power)
        if np.any(previous_mask):
            local_freq = lrfs_freq[previous_mask]
            local_power = integrated_lrfs_power[previous_mask]
            peak = int(np.nanargmax(local_power))
            lrfs_previous_f_peak = float(local_freq[peak])
            lrfs_previous_p3_estimate = float(1.0 / lrfs_previous_f_peak) if lrfs_previous_f_peak > EPSILON else None
    lrfs_feature_candidates = _lrfs_local_peak_candidates(lrfs_freq, integrated_lrfs_power)

    f3, f2, power_2dfs = _compute_2dfs(I_baseline_subtracted, None)
    positive_f3 = (f3 > 0) & (f3 <= 0.5)
    f3_positive = f3[positive_f3]
    power_2dfs_positive = power_2dfs[positive_f3, :]
    integrated_2dfs_longitude_frequency = np.nansum(power_2dfs_positive, axis=0) if power_2dfs_positive.size else np.array([])
    estimate = _estimate_p2_p3(f3, f2, power_2dfs)

    centers, sliding_freq, sliding_power = _compute_sliding_lrfs(I_baseline_subtracted, None)
    if sliding_freq.size:
        sliding_mask = sliding_freq > 0
        sliding_freq = sliding_freq[sliding_mask]
        sliding_power = sliding_power[:, sliding_mask]
    if sliding_freq.size and sliding_power.size:
        p3_mask = (sliding_freq >= 0.02) & (sliding_freq <= 0.5)
        p3_values = []
        f3_values = []
        peak_power = []
        for spectrum in sliding_power:
            local_power = spectrum[p3_mask]
            local_freq = sliding_freq[p3_mask]
            if local_power.size == 0 or not np.isfinite(local_power).any():
                f3_values.append(np.nan)
                p3_values.append(np.nan)
                peak_power.append(np.nan)
                continue
            peak = int(np.nanargmax(local_power))
            fpeak = float(local_freq[peak])
            f3_values.append(fpeak)
            p3_values.append(1.0 / fpeak if fpeak > EPSILON else np.nan)
            peak_power.append(float(local_power[peak]))
    else:
        p3_values = []
        f3_values = []
        peak_power = []

    sliding_2dfs = _compute_sliding_2dfs(I_baseline_subtracted, None)
    profile_stabilisation = _profile_stabilisation_diagnostics(I_baseline_subtracted)
    acf_psd = _acf_psd_diagnostics(pulse_energies)
    trial_null_fraction = _trial_null_fraction_diagnostics(pulse_energies, off_energy_reference)
    adp = _adp_diagnostics(I_baseline_subtracted)

    phase_indices = _select_even_indices(len(selected_phase), 320)
    lrfs_freq_indices = _select_even_indices(len(lrfs_freq), 180)
    f2_indices = _select_even_indices(len(f2), 280)
    f3_indices = _select_even_indices(len(f3_positive), 180)
    sliding_center_indices = _select_even_indices(len(centers), 220)
    sliding_freq_indices = _select_even_indices(len(sliding_freq), 180)
    profile_stabilisation_indices = _select_even_indices(len(profile_stabilisation["pulse_count"]), 120)
    acf_lag_indices = _select_even_indices(len(acf_psd["lag"]), 220)
    psd_freq_indices = _select_even_indices(len(acf_psd["frequency"]), 220)

    default_start, default_end = float(on_pulse[0]), float(on_pulse[1])
    return {
        "obs_id": obs_id,
        "start_phase": float(start_phase),
        "end_phase": float(end_phase),
        "on_pulse": {"start": default_start, "end": default_end},
        "phase_axis": _finite_list(selected_phase),
        "pulse_number": params.pulse_number.tolist(),
        "pulse_energy_distribution": {
            "bin_centers": _finite_list(energy_centers),
            "density": _finite_list(energy_counts),
            "bin_count": int(energy_bin_count),
            "bin_rule": "freedman-diaconis",
            "normalization": normalization,
            "normalization_factor": _finite_or_none(normalization_factor),
            "on_pulse_energy": _finite_list(plot_pulse_energies),
            "off_pulse_energy": _finite_list(plot_off_energy_reference),
            "raw_on_pulse_energy": _finite_list(pulse_energies),
            "raw_off_pulse_energy": _finite_list(off_energy_reference),
            "on_pulse_peak_intensity": _finite_list(pulse_peak_intensity),
            "off_pulse_peak_intensity": _finite_list(off_peak_intensity_reference),
            "mean_profile_peak": _finite_or_none(mean_profile_peak),
            "warnings": distribution_warnings,
            "debug": {
                "mode": normalization,
                "on_bin_count": int(I_baseline_subtracted.shape[1]),
                "active_on_bin_count": int(np.sum(full_on_mask)),
                "off_bin_count": int(np.sum(offpulse_mask)),
                "i0_finite_fraction": _finite_or_none(np.isfinite(full_I_baseline_subtracted).mean()),
                "scale": _finite_or_none(normalization_factor),
                "profile_peak": _finite_or_none(mean_profile_peak),
                "value_on": _distribution_stats(on_distribution_values),
                "value_off": _distribution_stats(off_distribution_values),
                "plotted_on": _distribution_stats(plot_pulse_energies),
                "plotted_off": _distribution_stats(plot_off_energy_reference),
            },
            "description": (
                "Measures pulse-to-pulse variability in baseline-subtracted total emitted energy "
                "integrated over the selected on-pulse window and compares it against matched "
                "off-pulse noise-reference windows."
            ),
        },
        "intensity_histogram": {
            "phase_axis": _finite_list(selected_phase[phase_indices]),
            "bin_centers": _finite_list(bin_centers),
            "hist2d": _finite_matrix(hist2d[:, phase_indices]),
            "log10_hist2d": _finite_matrix(log10_hist2d[:, phase_indices]),
            "log_hist2d": _finite_matrix(log10_hist2d[:, phase_indices]),
            "mean_profile": _finite_list(mean_profile[phase_indices]),
            "warnings": intensity_warnings,
            "metadata": {
                "n_pulses": int(params.num_pulses),
                "n_selected_phase_bins": int(I_baseline_subtracted.shape[1]),
                "n_intensity_bins": int(quantity_bins),
                "i_min": _finite_or_none(vmin),
                "i_max": _finite_or_none(vmax),
                "finite_value_count": int(intensity_values.size),
                "fraction_outside_robust_display_range": _finite_or_none(outside_display_range),
            },
        },
        "fluctuation_spectrum": {
            "phase_axis": _finite_list(selected_phase[phase_indices]),
            "frequency": _finite_list(lrfs_freq[lrfs_freq_indices]),
            "log_power": _finite_matrix(_log_power(lrfs_power[np.ix_(lrfs_freq_indices, phase_indices)])),
            "fft_phase": _finite_list(lrfs_peak_phase[phase_indices]),
            "mean_profile": _finite_list(profile_for_modulation[phase_indices]),
            "modulation_index": _finite_list(modulation[phase_indices]),
            "integrated_spectrum": _finite_list(integrated_lrfs_power[lrfs_freq_indices]) if integrated_lrfs_power.size else [],
            "p3_estimate": _finite_or_none(lrfs_p3_estimate) if lrfs_p3_estimate is not None else None,
            "f_peak": _finite_or_none(lrfs_f_peak) if lrfs_f_peak is not None else None,
            "previous_p3_estimate": _finite_or_none(lrfs_previous_p3_estimate) if lrfs_previous_p3_estimate is not None else None,
            "previous_f_peak": _finite_or_none(lrfs_previous_f_peak) if lrfs_previous_f_peak is not None else None,
            "p3_candidates": lrfs_feature_candidates,
            "warnings": fluc_warnings,
            "metadata": {
                "n_pulses": int(params.num_pulses),
                "n_phase_bins": int(I_baseline_subtracted.shape[1]),
                "noise_var": _finite_or_none(noise_var),
                "modulation_index_finite_count": int(np.sum(np.isfinite(modulation))),
                "df": _finite_or_none(1.0 / params.num_pulses) if params.num_pulses else None,
                "max_plotted_frequency": 0.5,
                "absolute_peak_rule": "max integrated LRFS power over all positive fluctuation frequencies",
                "low_frequency_excluded_peak_rule": "max integrated LRFS power for 0.04 <= f <= 0.5 cycles/P1",
                "candidate_peak_rule": "ranked local maxima in the integrated LRFS spectrum; intended to expose 1/P3 features and harmonics separately from broad red-noise power",
            },
        },
        "two_d_fluctuation_spectrum": {
            "f2": _finite_list(f2[f2_indices]),
            "f3": _finite_list(f3_positive[f3_indices]),
            "log_power": _finite_matrix(_log_power(power_2dfs_positive[np.ix_(f3_indices, f2_indices)])),
            "integrated_longitude_frequency_power": _finite_list(integrated_2dfs_longitude_frequency[f2_indices]) if integrated_2dfs_longitude_frequency.size else [],
            "estimate": estimate,
        },
        "longitude_resolved_fluctuation_spectrum": {
            "phase_axis": _finite_list(selected_phase[phase_indices]),
            "frequency": _finite_list(lrfs_freq[lrfs_freq_indices]),
            "log_power": _finite_matrix(_log_power(lrfs_power[np.ix_(lrfs_freq_indices, phase_indices)])),
            "integrated_spectrum": _finite_list(integrated_lrfs_power[lrfs_freq_indices]) if integrated_lrfs_power.size else [],
            "mean_profile": _finite_list(mean_profile[phase_indices]),
            "p3_estimate": lrfs_p3_estimate,
        },
        "p3_evolution": {
            "centers": _finite_list(centers[sliding_center_indices]),
            "frequency": _finite_list(sliding_freq[sliding_freq_indices]),
            "log_power": _finite_matrix(_log_power(sliding_power[np.ix_(sliding_center_indices, sliding_freq_indices)])) if sliding_power.size else [],
            "f3": _finite_list(np.asarray(f3_values)[sliding_center_indices]) if len(f3_values) else [],
            "P3": _finite_list(np.asarray(p3_values)[sliding_center_indices]) if len(p3_values) else [],
            "peak_power": _finite_list(np.asarray(peak_power)[sliding_center_indices]) if len(peak_power) else [],
            "sliding_2dfs_centers": _finite_list(sliding_2dfs["centers"]),
            "sliding_2dfs_f3": _finite_list(sliding_2dfs["f3"]),
            "sliding_2dfs_f2": _finite_list(sliding_2dfs["f2"]),
            "sliding_2dfs_P3": _finite_list(sliding_2dfs["P3"]),
            "sliding_2dfs_P2_bins": _finite_list(sliding_2dfs["P2_bins"]),
            "sliding_2dfs_peak_power": _finite_list(sliding_2dfs["peak_power"]),
        },
        "profile_stabilisation": {
            "pulse_count": _finite_list(profile_stabilisation["pulse_count"][profile_stabilisation_indices]),
            "correlation": _finite_list(profile_stabilisation["correlation"][profile_stabilisation_indices]),
            "one_minus_correlation": _finite_list(profile_stabilisation["one_minus_correlation"][profile_stabilisation_indices]),
            "reference": _finite_list(profile_stabilisation["reference"][profile_stabilisation_indices]),
            "description": "Correlation between the cumulative average profile and the final average profile over the selected phase window.",
        },
        "acf_psd": {
            "lag": _finite_list(acf_psd["lag"][acf_lag_indices]),
            "acf": _finite_list(acf_psd["acf"][acf_lag_indices]),
            "frequency": _finite_list(acf_psd["frequency"][psd_freq_indices]),
            "psd": _finite_list(acf_psd["psd"][psd_freq_indices]),
            "description": "Autocorrelation and power spectral density of the baseline-subtracted on-pulse energy sequence.",
        },
        "trial_null_fraction": {
            "threshold_sigma": _finite_list(trial_null_fraction["threshold_sigma"]),
            "null_fraction": _finite_list(trial_null_fraction["null_fraction"]),
            "default_threshold_sigma": _finite_or_none(trial_null_fraction["default_threshold_sigma"]),
            "default_null_fraction": _finite_or_none(trial_null_fraction["default_null_fraction"]),
            "off_rms": _finite_or_none(trial_null_fraction["off_rms"]),
            "description": "Trial null fraction from the fraction of on-pulse energies below a threshold expressed in matched off-pulse RMS units.",
        },
        "adp": {
            "phase_lag_bins": _finite_list(adp["phase_lag_bins"]),
            "correlation": _finite_list(adp["correlation"]),
            "description": "Adjacent-pulse drift profile: correlation between consecutive pulses as a function of phase-bin lag.",
        },
    }

# Legacy functions for backward compatibility
def plot_I_heatmap(data, start_phase, end_phase, obs_id):
    return plot_all_heatmaps(data, start_phase, end_phase, obs_id)['I']

def plot_Q_heatmap(data, start_phase, end_phase, obs_id):
    return plot_all_heatmaps(data, start_phase, end_phase, obs_id)['Q']

def plot_U_heatmap(data, start_phase, end_phase, obs_id):
    return plot_all_heatmaps(data, start_phase, end_phase, obs_id)['U']

def plot_V_heatmap(data, start_phase, end_phase, obs_id):
    return plot_all_heatmaps(data, start_phase, end_phase, obs_id)['V']

def plot_poincare_aitoff_at_phase(data, on_pulse, cphase, obs_id):
    params = _as_polarimetry_precompute(data, on_pulse)
    cbin = int(np.argmin(np.abs(params.phase_axis - cphase)))

    pa_val = params.PA_rad[:, cbin]
    ea_val = params.pulse_EA_rad[:, cbin]

    lon = 2 * pa_val
    lat = 2 * ea_val
    lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi

    return {
        "lon": lon,
        "lat": lat,
        "pulse_number": params.pulse_number,
        "PA": params.PA_deg[:, cbin],
        "EA": params.pulse_EA_deg[:, cbin],
        "I": params.I[:, cbin],
        "Q": params.Q[:, cbin],
        "U": params.U[:, cbin],
        "V": params.V[:, cbin],
        "p_frac": params.pulse_p_frac[:, cbin],
        "l_frac": params.pulse_l_frac[:, cbin],
        "v_frac": params.pulse_v_frac[:, cbin],
        "absv_frac": params.pulse_abs_vfrac[:, cbin],
    }


def phase_slice_histogram_single(data, left_phase, mid_phase, right_phase, on_pulse, obs_id, quantity_key, default_bins=200, sigma_threshold=3.0):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses
    specs, warnings, common_metadata = _phase_slice_histogram_specs(params, sigma_threshold)

    if quantity_key not in specs:
        return {"error": f"Unknown quantity {quantity_key}"}

    quantity, name, valid_mask, x_limits, validity_rule = specs[quantity_key]
    phase_values = [left_phase, mid_phase, right_phase]
    phase_bins = [np.argmin(np.abs(phase_axis - val)) for val in phase_values]

    def compute_bin_count(values):
        values = values[np.isfinite(values)]
        if values.size < 2:
            return default_bins
        val_iqr = _iqr(values)
        if val_iqr > 0:
            bin_width = 2 * val_iqr / (len(values) ** (1 / 3))
            range_ = np.ptp(values)
            return int(np.clip(range_ / bin_width if bin_width else default_bins, 20, 300))
        return default_bins

    phase_entries = []
    for phase_bin, phase_val in zip(phase_bins, phase_values):
        raw_values = quantity[:, phase_bin]
        valid = valid_mask[:, phase_bin] if np.shape(valid_mask) == np.shape(quantity) else np.isfinite(raw_values)
        finite_mask = np.isfinite(raw_values)
        values = raw_values[finite_mask & valid]
        bin_count = compute_bin_count(values)
        vmin, vmax = _adaptive_display_range(values, x_limits)

        counts, edges = np.histogram(values, bins=bin_count, range=(vmin, vmax))
        outside_range = int(np.sum((values < vmin) | (values > vmax))) if values.size else 0

        phase_entries.append({
            "phase_value": float(phase_val),
            "phase_bin_index": int(phase_bin),
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "x_limits": x_limits,
            "stats": {
                "min": _finite_or_none(np.nanmin(values)) if values.size else None,
                "max": _finite_or_none(np.nanmax(values)) if values.size else None,
                "mean": _finite_or_none(np.nanmean(values)) if values.size else None,
                "std": _finite_or_none(np.nanstd(values)) if values.size else None,
                "num_pulses": int(num_pulses),
                "finite_values": int(np.sum(finite_mask)),
                "valid_values": int(values.size),
                "masked_fraction": _finite_or_none(1.0 - (values.size / np.sum(finite_mask))) if np.sum(finite_mask) else None,
                "outside_display_range": int(outside_range),
                "fraction_outside_display_range": _finite_or_none(outside_range / values.size) if values.size else None,
            },
        })

    return {
        "obs_id": obs_id,
        "phase_values": [float(p) for p in phase_values],
        "phase_bins": [int(p) for p in phase_bins],
        "quantity": {
            "key": quantity_key,
            "name": name,
            "phase_slices": phase_entries,
            "warnings": warnings,
            "metadata": {
                **common_metadata,
                "validity_rule": validity_rule,
                "x_range_mode": "physical-default",
                "count_scale": "raw-count",
            },
        },
    }

def plot_phase_slice_histograms_by_phase(data, left_phase, mid_phase, right_phase, on_pulse, obs_id, default_bins=200, return_data=False, sigma_threshold=3.0):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses

    phase_values = [left_phase, mid_phase, right_phase]
    phase_bins = [np.argmin(np.abs(phase_axis - val)) for val in phase_values]

    quantity_specs, warnings, common_metadata = _phase_slice_histogram_specs(params, sigma_threshold)
    quantity_items = [(key, *quantity_specs[key]) for key in PHASE_SLICE_QUANTITY_KEYS]

    def compute_bin_count(values):
        values = values[np.isfinite(values)]
        if values.size < 2:
            return default_bins
        val_iqr = _iqr(values)
        if val_iqr > 0:
            bin_width = 2 * val_iqr / (len(values) ** (1 / 3))
            range_ = np.ptp(values)
            return int(np.clip(range_ / bin_width if bin_width else default_bins, 20, 300))
        return default_bins

    result = None
    if return_data:
        result = {
            "obs_id": obs_id,
            "phase_values": [float(p) for p in phase_values],
            "phase_bins": [int(p) for p in phase_bins],
            "quantities": [],
        }
    else:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axs = plt.subplots(len(quantity_items), len(phase_bins), figsize=(20, 15), constrained_layout=True)

    for row_idx, (quantity_key, quantity, name, valid_mask, x_limits, validity_rule) in enumerate(quantity_items):
        phase_entries = []
        for col_idx, (phase_bin, phase_val) in enumerate(zip(phase_bins, phase_values)):
            raw_values = quantity[:, phase_bin]
            valid = valid_mask[:, phase_bin] if np.shape(valid_mask) == np.shape(quantity) else np.isfinite(raw_values)
            finite_mask = np.isfinite(raw_values)
            values = raw_values[finite_mask & valid]
            bin_count = compute_bin_count(values)
            vmin, vmax = _adaptive_display_range(values, x_limits)

            if return_data:
                counts, edges = np.histogram(values, bins=bin_count, range=(vmin, vmax))
                outside_range = int(np.sum((values < vmin) | (values > vmax))) if values.size else 0

                phase_entries.append({
                    "phase_value": float(phase_val),
                    "phase_bin_index": int(phase_bin),
                    "bin_edges": edges.tolist(),
                    "counts": counts.tolist(),
                    "x_limits": x_limits,
                    "stats": {
                        "min": _finite_or_none(np.nanmin(values)) if values.size else None,
                        "max": _finite_or_none(np.nanmax(values)) if values.size else None,
                        "mean": _finite_or_none(np.nanmean(values)) if values.size else None,
                        "std": _finite_or_none(np.nanstd(values)) if values.size else None,
                        "num_pulses": int(num_pulses),
                        "finite_values": int(np.sum(finite_mask)),
                        "valid_values": int(values.size),
                        "masked_fraction": _finite_or_none(1.0 - (values.size / np.sum(finite_mask))) if np.sum(finite_mask) else None,
                        "outside_display_range": int(outside_range),
                        "fraction_outside_display_range": _finite_or_none(outside_range / values.size) if values.size else None,
                    },
                })
                continue

            ax = axs[row_idx, col_idx]
            ax.hist(values, bins=bin_count, range=(vmin, vmax), color='steelblue', alpha=0.8)
            ax.set_title(f"{name}\nPhase = {phase_val:.2f}")
            ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.set_xlim(vmin, vmax)

        if return_data:
            result["quantities"].append({
                "key": quantity_key,
                "name": name,
                "phase_slices": phase_entries,
                "warnings": warnings,
                "metadata": {
                    **common_metadata,
                    "validity_rule": validity_rule,
                    "x_range_mode": "physical-default",
                    "count_scale": "raw-count",
                },
            })

    if return_data:
        return result

# New: compute a single polarisation histogram payload for one quantity
def polarisation_histogram_single(data, start_phase, end_phase, on_pulse, obs_id, quantity_key, base_quantity_bins=200, sigma_threshold=3.0):
    """
    quantity_key in {"PA", "EA", "P/I", "L/I", "|V/I|", "V/I", "I", "Q", "U", "V"}
    """
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses

    default_start, default_end = params.on_pulse
    on_pulse_mask = (phase_axis >= float(default_start)) & (phase_axis <= float(default_end))
    off_pulse_mask = ~on_pulse_mask
    warnings = []

    if np.sum(off_pulse_mask) < 5:
        warnings.append("Not enough off-pulse bins for baseline subtraction and noise masks.")
        baseline_I = np.zeros((num_pulses, 1), dtype=float)
        baseline_Q = np.zeros((num_pulses, 1), dtype=float)
        baseline_U = np.zeros((num_pulses, 1), dtype=float)
        baseline_V = np.zeros((num_pulses, 1), dtype=float)
    else:
        baseline_I = np.nanmean(params.I[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_Q = np.nanmean(params.Q[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_U = np.nanmean(params.U[:, off_pulse_mask], axis=1, keepdims=True)
        baseline_V = np.nanmean(params.V[:, off_pulse_mask], axis=1, keepdims=True)

    I0 = params.I - baseline_I
    Q0 = params.Q - baseline_Q
    U0 = params.U - baseline_U
    V0 = params.V - baseline_V

    def _offpulse_sigma(values):
        if np.sum(off_pulse_mask) < 5:
            return EPSILON
        finite = np.asarray(values[:, off_pulse_mask], dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return EPSILON
        return _safe_scalar(np.nanstd(finite))

    sigma_I = _offpulse_sigma(I0)
    sigma_Q = _offpulse_sigma(Q0)
    sigma_U = _offpulse_sigma(U0)
    sigma_V = _offpulse_sigma(V0)
    sigma_L = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2))
    sigma_P = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2 + sigma_V ** 2))
    sigma_threshold = float(sigma_threshold) if np.isfinite(float(sigma_threshold)) else 3.0

    L_raw = np.sqrt(Q0 ** 2 + U0 ** 2)
    P_raw = np.sqrt(Q0 ** 2 + U0 ** 2 + V0 ** 2)
    L_debiased = _debias_polarisation(L_raw, sigma_L)
    P_debiased = _debias_polarisation(P_raw, sigma_P)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_frac = np.divide(P_debiased, I0, out=np.full_like(P_debiased, np.nan, dtype=float), where=np.abs(I0) > EPSILON)
        l_frac = np.divide(L_debiased, I0, out=np.full_like(L_debiased, np.nan, dtype=float), where=np.abs(I0) > EPSILON)
        v_frac = np.divide(V0, I0, out=np.full_like(V0, np.nan, dtype=float), where=np.abs(I0) > EPSILON)

    PA_deg = np.degrees(0.5 * np.arctan2(U0, Q0))
    EA_deg = np.degrees(0.5 * np.arctan2(V0, L_debiased))
    intensity_valid = np.abs(I0) > sigma_threshold * sigma_I
    linear_valid = L_raw > sigma_threshold * sigma_L
    total_pol_valid = P_raw > sigma_threshold * sigma_P

    quantity_map = {
        "PA": (PA_deg, "PA [deg]", linear_valid, (-90.0, 90.0)),
        "EA": (EA_deg, "EA [deg]", total_pol_valid, (-45.0, 45.0)),
        "P/I": (p_frac, "P/I", intensity_valid, (0.0, 1.5)),
        "L/I": (l_frac, "L/I", intensity_valid, (0.0, 1.5)),
        "|V/I|": (np.abs(v_frac), "|V/I|", intensity_valid, (0.0, 1.5)),
        "V/I": (v_frac, "V/I", intensity_valid, (-1.5, 1.5)),
        "I": (I0, "I", np.isfinite(I0), None),
        "Q": (Q0, "Q", np.isfinite(Q0), None),
        "U": (U0, "U", np.isfinite(U0), None),
        "V": (V0, "V", np.isfinite(V0), None),
    }
    if quantity_key not in quantity_map:
        return {"error": f"Unknown quantity {quantity_key}"}

    quantity, label, valid_mask, default_range = quantity_map[quantity_key]

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase, end_side="right")
    selected_phase_axis = phase_axis[start_idx:end_idx]
    selected_phase_bins = end_idx - start_idx

    quantity_bins = max(50, min(base_quantity_bins, selected_phase_bins)) if selected_phase_bins > 0 else 50

    if selected_phase_bins <= 0:
        return {
            "obs_id": obs_id,
            "start_phase": float(start_phase),
            "end_phase": float(end_phase),
            "on_pulse": {"start": float(default_start), "end": float(default_end)},
            "quantity_bins": int(quantity_bins),
            "phase_axis": [],
            "quantities": [],
            "warning": "No phase bins selected; check start_phase/end_phase",
        }

    q = quantity.T[start_idx:end_idx]
    valid = valid_mask.T[start_idx:end_idx] if np.shape(valid_mask) == np.shape(quantity) else np.isfinite(q)
    if default_range is not None:
        q_min, q_max = default_range
        y_range_mode = "physical-default"
    elif q.size == 0:
        q_min, q_max = 0.0, 1.0
    else:
        finite_q = q[np.isfinite(q)]
        if finite_q.size:
            q_min, q_max = float(np.nanmin(finite_q)), float(np.nanmax(finite_q))
        else:
            q_min, q_max = 0.0, 1.0
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min == q_max:
            q_min = 0.0
            q_max = 1.0
        if q_min == q_max:
            q_max = q_min + 1e-3
        y_range_mode = "finite-minmax"

    hist2d = np.zeros((quantity_bins, selected_phase_bins))
    bin_edges = np.linspace(q_min, q_max, quantity_bins + 1)
    total_finite_values = 0
    total_valid_values = 0
    total_outside_range = 0
    for i in range(selected_phase_bins):
        row = q[i] if q.size else np.array([])
        row_valid = valid[i] if np.size(valid) else np.array([], dtype=bool)
        finite_mask = np.isfinite(row)
        total_finite_values += int(np.sum(finite_mask))
        values = row[finite_mask & row_valid]
        total_valid_values += int(values.size)
        if values.size:
            total_outside_range += int(np.sum((values < q_min) | (values > q_max)))
        hist, _ = np.histogram(values, bins=bin_edges)
        hist2d[:, i] = hist

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    extent = [selected_phase_axis[0], selected_phase_axis[-1], bin_centers[0], bin_centers[-1]]
    log10_hist2d = np.full_like(hist2d, np.nan, dtype=float)
    mask = hist2d > 0
    log10_hist2d[mask] = np.log10(hist2d[mask])

    return {
        "obs_id": obs_id,
        "start_phase": float(start_phase),
        "end_phase": float(end_phase),
        "on_pulse": {"start": float(default_start), "end": float(default_end)},
        "quantity": label,
        "quantity_key": quantity_key,
        "is_fraction": quantity_key in {"P/I", "L/I", "|V/I|", "V/I"},
        "quantity_bins": int(quantity_bins),
        "phase_axis": selected_phase_axis.tolist(),
        "hist2d": _finite_matrix(hist2d),
        "log10_hist2d": _finite_matrix(log10_hist2d),
        "log_hist2d": _finite_matrix(log10_hist2d),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "extent": [float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])],
        "q_min": float(q_min),
        "q_max": float(q_max),
        "lowfrac": None,
        "num_pulses": int(num_pulses),
        "warnings": warnings,
        "metadata": {
            "histogram_log": "log10",
            "zero_count_bins": "NaN",
            "sigma_threshold": float(sigma_threshold),
            "sigma_I": _finite_or_none(sigma_I),
            "sigma_Q": _finite_or_none(sigma_Q),
            "sigma_U": _finite_or_none(sigma_U),
            "sigma_V": _finite_or_none(sigma_V),
            "sigma_L": _finite_or_none(sigma_L),
            "sigma_P": _finite_or_none(sigma_P),
            "validity_rule": {
                "PA": "L_raw > sigma_threshold * sqrt(sigma_Q^2 + sigma_U^2)",
                "EA": "P_raw > sigma_threshold * sqrt(sigma_Q^2 + sigma_U^2 + sigma_V^2)",
                "fractions": "abs(I0) > sigma_threshold * sigma_I",
            }.get(quantity_key if quantity_key in {"PA", "EA"} else "fractions", "finite"),
            "y_range_mode": y_range_mode,
            "finite_value_count": int(total_finite_values),
            "valid_value_count": int(total_valid_values),
            "masked_fraction": _finite_or_none(1.0 - (total_valid_values / total_finite_values)) if total_finite_values else None,
            "fraction_outside_display_range": _finite_or_none(total_outside_range / total_valid_values) if total_valid_values else None,
            "uses_rician_debiased_LP": True,
        },
    }


def _rvm_pa_deg(phase, alpha_deg, beta_deg, phi0, psi0_deg):
    phase = np.asarray(phase, dtype=float)
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    zeta = alpha + beta
    phase_rad = 2.0 * np.pi * (phase - phi0)
    numerator = np.sin(alpha) * np.sin(phase_rad)
    denominator = (np.sin(zeta) * np.cos(alpha)) - (np.cos(zeta) * np.sin(alpha) * np.cos(phase_rad))
    pa = psi0_deg + np.degrees(np.arctan2(numerator, denominator))
    return ((pa + 90.0) % 180.0) - 90.0


def _pa_uncertainty_deg_from_l_snr(l_snr):
    l_snr = np.asarray(l_snr, dtype=float)
    out = np.full_like(l_snr, np.nan, dtype=float)
    finite = np.isfinite(l_snr) & (l_snr > EPSILON)
    high = finite & (l_snr >= 10.0)
    low = finite & ~high
    # High-S/N asymptotic form from Everett & Weisberg-style propagation.
    out[high] = 28.65 / l_snr[high]
    # For lower S/N, use a conservative smooth approximation to the 68% PA
    # interval; the exact NKC PDF integration can replace this without changing
    # the payload contract.
    out[low] = np.clip(28.65 / np.maximum(l_snr[low], EPSILON), 3.0, 45.0)
    return out


def rvm_fit_payload(data, start_phase, end_phase, on_pulse, obs_id, phase_bins=96, pa_bins=120, sigma_threshold=3.0):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase, end_side="right")
    if start_idx >= end_idx:
        return {
            "obs_id": obs_id,
            "start_phase": float(start_phase),
            "end_phase": float(end_phase),
            "on_pulse": {"start": float(params.on_pulse[0]), "end": float(params.on_pulse[1])},
            "warning": "No phase bins selected; check start_phase/end_phase.",
        }

    phase = phase_axis[start_idx:end_idx]
    pa_pulses = params.PA_deg[:, start_idx:end_idx]
    l_snr_pulses = params.L_sigma[:, start_idx:end_idx]
    valid_pulses = np.isfinite(pa_pulses) & (l_snr_pulses > float(sigma_threshold))

    phase_bin_count = max(16, min(int(phase_bins), len(phase)))
    if phase_bin_count == len(phase):
        hist_phase = phase
        hist_counts = np.zeros((int(pa_bins), len(phase)), dtype=float)
        pa_edges = np.linspace(-90.0, 90.0, int(pa_bins) + 1)
        for idx in range(len(phase)):
            values = pa_pulses[:, idx][valid_pulses[:, idx]]
            hist_counts[:, idx], _ = np.histogram(values, bins=pa_edges)
    else:
        phase_edges = np.linspace(float(phase[0]), float(phase[-1]), phase_bin_count + 1)
        hist_phase = 0.5 * (phase_edges[:-1] + phase_edges[1:])
        pa_edges = np.linspace(-90.0, 90.0, int(pa_bins) + 1)
        hist_counts = np.zeros((int(pa_bins), phase_bin_count), dtype=float)
        for idx in range(phase_bin_count):
            phase_mask = (phase >= phase_edges[idx]) & (phase < phase_edges[idx + 1] if idx < phase_bin_count - 1 else phase <= phase_edges[idx + 1])
            values = pa_pulses[:, phase_mask][valid_pulses[:, phase_mask]]
            hist_counts[:, idx], _ = np.histogram(values, bins=pa_edges)

    log10_hist = np.full_like(hist_counts, np.nan, dtype=float)
    count_mask = hist_counts > 0
    log10_hist[count_mask] = np.log10(hist_counts[count_mask])
    pa_centers = 0.5 * (pa_edges[:-1] + pa_edges[1:])

    fit_phase = phase
    fit_pa = params.mean_PA_deg[start_idx:end_idx]
    fit_l_snr = params.mean_L_sigma[start_idx:end_idx]
    fit_sigma = _pa_uncertainty_deg_from_l_snr(fit_l_snr)
    fit_valid = np.isfinite(fit_phase) & np.isfinite(fit_pa) & np.isfinite(fit_sigma) & (fit_l_snr > float(sigma_threshold))

    warnings = []
    fit = None
    if least_squares is None:
        warnings.append("SciPy is unavailable; RVM fit could not be computed.")
    elif np.sum(fit_valid) < 5:
        warnings.append("Too few PA bins pass the linear-polarisation S/N threshold for a stable RVM fit.")
    else:
        x = fit_phase[fit_valid]
        y = fit_pa[fit_valid]
        sigma = np.clip(fit_sigma[fit_valid], 1.0, 45.0)
        weight = 1.0 / sigma

        pa_mid = float(np.nanmedian(y))
        fit_l_snr_valid = np.asarray(fit_l_snr[fit_valid], dtype=float)
        initial_phi0 = float(x[np.nanargmax(fit_l_snr_valid)]) if fit_l_snr_valid.size else float(np.nanmedian(x))
        guesses = [
            [45.0, 0.0, initial_phi0, pa_mid],
            [80.0, 2.0, initial_phi0, pa_mid],
            [120.0, -2.0, initial_phi0, pa_mid],
            [30.0, 5.0, initial_phi0, pa_mid],
        ]
        bounds = ([1.0, -60.0, float(start_phase), -180.0], [179.0, 60.0, float(end_phase), 180.0])

        def residual(theta):
            model = _rvm_pa_deg(x, theta[0], theta[1], theta[2], theta[3])
            direct = _angle_residual_deg(y, model)
            opm = _angle_residual_deg(y, model + 90.0)
            chosen = np.where(np.abs(opm) < np.abs(direct), opm, direct)
            return chosen * weight

        best = None
        for guess in guesses:
            guess = np.asarray(guess, dtype=float)
            guess[2] = np.clip(guess[2], bounds[0][2], bounds[1][2])
            try:
                result = least_squares(residual, guess, bounds=bounds, max_nfev=4000)
            except Exception:
                continue
            if best is None or result.cost < best.cost:
                best = result

        if best is None:
            warnings.append("RVM minimisation failed.")
        else:
            theta = best.x
            residual_values = residual(theta)
            dof = max(int(residual_values.size) - int(theta.size), 1)
            chi2 = float(np.sum(residual_values ** 2))
            model_fit = _rvm_pa_deg(fit_phase, theta[0], theta[1], theta[2], theta[3])
            fit = {
                "alpha_deg": _finite_or_none(theta[0]),
                "beta_deg": _finite_or_none(theta[1]),
                "zeta_deg": _finite_or_none(theta[0] + theta[1]),
                "phi0": _finite_or_none(theta[2]),
                "psi0_deg": _finite_or_none(((theta[3] + 90.0) % 180.0) - 90.0),
                "chi2": _finite_or_none(chi2),
                "reduced_chi2": _finite_or_none(chi2 / dof),
                "num_fit_points": int(np.sum(fit_valid)),
                "dof": int(dof),
                "phase": _finite_list(fit_phase),
                "pa_model": _finite_list(model_fit),
                "pa_model_opm": _finite_list(((model_fit + 90.0 + 90.0) % 180.0) - 90.0),
                "method": "weighted least squares with PA residuals wrapped modulo 180 deg; each point may use the direct or 90 deg OPM branch",
            }

    return {
        "obs_id": obs_id,
        "start_phase": float(start_phase),
        "end_phase": float(end_phase),
        "on_pulse": {"start": float(params.on_pulse[0]), "end": float(params.on_pulse[1])},
        "phase_axis": _finite_list(hist_phase),
        "pa_bin_centers": _finite_list(pa_centers),
        "hist2d": _finite_matrix(hist_counts),
        "log10_hist2d": _finite_matrix(log10_hist),
        "fit_points": {
            "phase": _finite_list(fit_phase[fit_valid]),
            "pa": _finite_list(fit_pa[fit_valid]),
            "pa_err": _finite_list(fit_sigma[fit_valid]),
            "linear_snr": _finite_list(fit_l_snr[fit_valid]),
        },
        "fit": fit,
        "warnings": warnings,
        "metadata": {
            "sigma_threshold": float(sigma_threshold),
            "pa_error_rule": "28.65/L_SNR for L_SNR >= 10; conservative low-SNR approximation otherwise",
            "opm_handling": "RVM fit uses the smaller residual between the primary track and a 90 degree orthogonal branch.",
            "empirical_histogram": "The PA density map is empirical; model curves are overlaid without modifying the histogram counts.",
        },
    }

def plot_polarisation_stacks(data, start_phase, end_phase, on_pulse, obs_id, return_data=False):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses
    quantity_specs = _polarisation_stack_specs(params)

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase)
    selected_phase_axis = phase_axis[start_idx:end_idx]

    if return_data:
        default_start, default_end = params.on_pulse
        payload = {
            "obs_id": obs_id,
            "start_phase": float(start_phase),
            "end_phase": float(end_phase),
            "on_pulse": {"start": float(default_start), "end": float(default_end)},
            "phase_axis": selected_phase_axis.tolist(),
            "pulse_number": list(range(num_pulses)),
            "quantities": [],
        }
    
    if return_data:
        for quantity_key in POLARISATION_STACK_KEYS:
            quantity, label = quantity_specs[quantity_key]
            q = quantity[:, start_idx:end_idx]
            q_min, q_max = np.nanmin(q), np.nanmax(q)
            if q_min == q_max:
                pad = max(abs(q_min) * 0.1, 1e-3)
                q_min -= pad
                q_max += pad

            payload["quantities"].append({
                "key": quantity_key,
                "name": label,
                "data": q.tolist(),
                "vmin": float(q_min),
                "vmax": float(q_max),
            })
        return payload


def polarisation_stack_single(data, start_phase, end_phase, on_pulse, obs_id, quantity_key):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    quantity_specs = _polarisation_stack_specs(params)

    if quantity_key not in quantity_specs:
        return {"error": f"Unknown quantity {quantity_key}"}

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase)
    selected_phase_axis = phase_axis[start_idx:end_idx]
    quantity, label = quantity_specs[quantity_key]
    q = quantity[:, start_idx:end_idx]
    finite_q = q[np.isfinite(q)] if q.size else np.array([])
    if finite_q.size:
        q_min = float(np.nanmin(finite_q))
        q_max = float(np.nanmax(finite_q))
    else:
        q_min, q_max = 0.0, 1.0
    if np.isfinite(q_min) and np.isfinite(q_max) and q_min == q_max:
        pad = max(abs(q_min) * 0.1, 1e-3)
        q_min -= pad
        q_max += pad

    default_start, default_end = params.on_pulse
    return {
        "obs_id": obs_id,
        "start_phase": float(start_phase),
        "end_phase": float(end_phase),
        "on_pulse": {"start": float(default_start), "end": float(default_end)},
        "phase_axis": selected_phase_axis.tolist(),
        "pulse_number": params.pulse_number.tolist(),
        "quantity": {
            "key": quantity_key,
            "name": label,
            "data": _finite_matrix(q),
            "vmin": q_min,
            "vmax": q_max,
        },
    }


def _json_row(row):
    row = np.asarray(row)
    if np.isfinite(row).all():
        values = row.tolist()
    else:
        values = [float(value) if np.isfinite(value) else None for value in row]
    return json.dumps(values, separators=(",", ":"), allow_nan=False)


def _json_value(value):
    if isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_dumps(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def iter_polarisation_stacks_json(data, start_phase, end_phase, on_pulse, obs_id):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    quantity_specs = _polarisation_stack_specs(params)

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase)
    selected_phase_axis = phase_axis[start_idx:end_idx]
    default_start, default_end = params.on_pulse

    yield "{"
    yield f'"obs_id":{_json_dumps(obs_id)},'
    yield f'"start_phase":{_json_dumps(float(start_phase))},'
    yield f'"end_phase":{_json_dumps(float(end_phase))},'
    yield f'"on_pulse":{_json_dumps({"start": float(default_start), "end": float(default_end)})},'
    yield f'"phase_axis":{_json_dumps(selected_phase_axis.tolist())},'
    yield f'"pulse_number":{_json_dumps(params.pulse_number.tolist())},'
    yield '"quantities":['

    for quantity_index, quantity_key in enumerate(POLARISATION_STACK_KEYS):
        quantity, label = quantity_specs[quantity_key]
        q = quantity[:, start_idx:end_idx]
        if q.size:
            q_min = float(np.nanmin(q))
            q_max = float(np.nanmax(q))
        else:
            q_min, q_max = 0.0, 1.0
        if np.isfinite(q_min) and np.isfinite(q_max) and q_min == q_max:
            pad = max(abs(q_min) * 0.1, 1e-3)
            q_min -= pad
            q_max += pad

        if quantity_index:
            yield ","
        yield "{"
        yield f'"key":{_json_dumps(quantity_key)},'
        yield f'"name":{_json_dumps(label)},'
        yield '"data":['
        for row_index, row in enumerate(q):
            if row_index:
                yield ","
            yield _json_row(row)
        yield f'],"vmin":{_json_dumps(_json_value(q_min))},"vmax":{_json_dumps(_json_value(q_max))}'
        yield "}"

    yield "]}"


# --- Poincare sphere + polarisation fractions/angles (ported from old_modules/functions.py) ---
def find_radius(points):
    """
    Radius of the circle on the unit sphere passing through 3 points.
    """
    p1, p2, p3 = [np.array(p) / np.linalg.norm(p) for p in points]

    normal = np.cross(p2 - p1, p3 - p1)
    nrm = np.linalg.norm(normal)
    if nrm == 0:
        return np.nan
    normal /= nrm

    d = abs(np.dot(p1, normal))
    d = np.clip(d, 0.0, 1.0)
    return np.sqrt(1.0 - d ** 2)


def compute_polarisation_parameters(I, Q, U, V, phase_axis, on_pulse):
    start, end = on_pulse
    on_mask = (phase_axis >= start) & (phase_axis <= end)
    off_mask = ~on_mask

    sigma_off = _safe_scalar(np.std(I[off_mask])) if np.any(off_mask) else EPSILON
    threshold = np.min(I[on_mask]) if np.any(on_mask) else np.min(I)

    Q_sq, U_sq, V_sq = Q ** 2, U ** 2, V ** 2
    L = np.sqrt(Q_sq + U_sq)
    P = np.sqrt(Q_sq + U_sq + V_sq)
    L_true = _debias_polarisation(L, sigma_off)
    P_true = _debias_polarisation(P, sigma_off)

    p_frac = _fraction(P_true, I, threshold)
    l_frac = _fraction(L_true, I, threshold)
    v_frac = _fraction(V, I, threshold)
    absv_frac = np.abs(v_frac)

    PA_rad = 0.5 * np.arctan2(U, Q)
    EA_rad = 0.5 * np.arctan2(V, L_true)
    PA = np.degrees(PA_rad)
    EA = np.degrees(EA_rad)

    dPA = _normalised_gradient(PA, phase_axis)
    lon = 2 * PA_rad
    lat = 2 * EA_rad
    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)
    roc = _radius_of_curvature_from_xyz(x, y, z)

    return dict(
        I=I,
        Q=Q,
        U=U,
        V=V,
        L=L_true,
        P=P_true,
        p_frac=p_frac,
        l_frac=l_frac,
        v_frac=v_frac,
        absv_frac=absv_frac,
        PA=PA,
        EA=EA,
        dPA=dPA,
        x=x,
        y=y,
        z=z,
        radius_of_curvature=roc,
        roc_phase=phase_axis,
    )


def _polarisation_entry(params, pulse_index):
    sigma_I = _safe_offpulse_std(params.I, params.off_pulse_mask)
    sigma_Q = _safe_offpulse_std(params.Q, params.off_pulse_mask)
    sigma_U = _safe_offpulse_std(params.U, params.off_pulse_mask)
    sigma_V = _safe_offpulse_std(params.V, params.off_pulse_mask)
    sigma_L = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2))
    sigma_P = _safe_scalar(np.sqrt(sigma_Q ** 2 + sigma_U ** 2 + sigma_V ** 2))

    if pulse_index == 0:
        n_scale = np.sqrt(max(params.num_pulses, 1))
        mean_sigma_I = sigma_I / n_scale
        mean_sigma_Q = sigma_Q / n_scale
        mean_sigma_U = sigma_U / n_scale
        mean_sigma_V = sigma_V / n_scale
        mean_sigma_L = sigma_L / n_scale
        mean_sigma_P = sigma_P / n_scale
        return dict(
            I=params.I_mean,
            Q=params.Q_mean,
            U=params.U_mean,
            V=params.V_mean,
            L=params.mean_L_true,
            P=params.mean_P_true,
            p_frac=params.mean_p_frac,
            l_frac=params.mean_l_frac,
            v_frac=params.mean_v_frac,
            absv_frac=params.mean_abs_vfrac,
            PA=params.mean_PA_deg,
            EA=params.mean_EA_deg,
            dPA=params.mean_dPA_dphi,
            x=params.mean_x,
            y=params.mean_y,
            z=params.mean_z,
            radius_of_curvature=params.mean_radius_of_curvature,
            roc_phase=params.roc_phase,
            I_err=_constant_like(params.I_mean, mean_sigma_I),
            Q_err=_constant_like(params.Q_mean, mean_sigma_Q),
            U_err=_constant_like(params.U_mean, mean_sigma_U),
            V_err=_constant_like(params.V_mean, mean_sigma_V),
            p_frac_err=_fraction_error(params.mean_P_true, params.I_mean, mean_sigma_P, mean_sigma_I),
            l_frac_err=_fraction_error(params.mean_L_true, params.I_mean, mean_sigma_L, mean_sigma_I),
            v_frac_err=_fraction_error(params.V_mean, params.I_mean, mean_sigma_V, mean_sigma_I),
            absv_frac_err=_fraction_error(params.V_mean, params.I_mean, mean_sigma_V, mean_sigma_I),
            PA_err=_angle_error_from_snr(params.mean_L, mean_sigma_L),
            EA_err=_angle_error_from_snr(params.mean_P, mean_sigma_P),
        )

    pulse_idx = pulse_index - 1
    pulse_sigma_I = _safe_offpulse_std(params.I[pulse_idx], params.off_pulse_mask)
    pulse_sigma_Q = _safe_offpulse_std(params.Q[pulse_idx], params.off_pulse_mask)
    pulse_sigma_U = _safe_offpulse_std(params.U[pulse_idx], params.off_pulse_mask)
    pulse_sigma_V = _safe_offpulse_std(params.V[pulse_idx], params.off_pulse_mask)
    pulse_sigma_L = _safe_scalar(np.sqrt(pulse_sigma_Q ** 2 + pulse_sigma_U ** 2))
    pulse_sigma_P = _safe_scalar(np.sqrt(pulse_sigma_Q ** 2 + pulse_sigma_U ** 2 + pulse_sigma_V ** 2))
    return dict(
        I=params.I[pulse_idx],
        Q=params.Q[pulse_idx],
        U=params.U[pulse_idx],
        V=params.V[pulse_idx],
        L=params.pulse_L_true[pulse_idx],
        P=params.pulse_P_true[pulse_idx],
        p_frac=params.pulse_p_frac[pulse_idx],
        l_frac=params.pulse_l_frac[pulse_idx],
        v_frac=params.pulse_v_frac[pulse_idx],
        absv_frac=params.pulse_abs_vfrac[pulse_idx],
        PA=params.PA_deg[pulse_idx],
        EA=params.pulse_EA_deg[pulse_idx],
        dPA=params.pulse_dPA_dphi[pulse_idx],
        x=params.pulse_x[pulse_idx],
        y=params.pulse_y[pulse_idx],
        z=params.pulse_z[pulse_idx],
        radius_of_curvature=params.pulse_radius_of_curvature[pulse_idx],
        roc_phase=params.roc_phase,
        I_err=_constant_like(params.I[pulse_idx], pulse_sigma_I),
        Q_err=_constant_like(params.Q[pulse_idx], pulse_sigma_Q),
        U_err=_constant_like(params.U[pulse_idx], pulse_sigma_U),
        V_err=_constant_like(params.V[pulse_idx], pulse_sigma_V),
        p_frac_err=_fraction_error(params.pulse_P_true[pulse_idx], params.I[pulse_idx], pulse_sigma_P, pulse_sigma_I),
        l_frac_err=_fraction_error(params.pulse_L_true[pulse_idx], params.I[pulse_idx], pulse_sigma_L, pulse_sigma_I),
        v_frac_err=_fraction_error(params.V[pulse_idx], params.I[pulse_idx], pulse_sigma_V, pulse_sigma_I),
        absv_frac_err=_fraction_error(params.V[pulse_idx], params.I[pulse_idx], pulse_sigma_V, pulse_sigma_I),
        PA_err=_angle_error_from_snr(params.L[pulse_idx], pulse_sigma_L),
        EA_err=_angle_error_from_snr(params.P[pulse_idx], pulse_sigma_P),
    )


def build_polarisation_dataset(data, on_pulse):
    """
    Build derived-parameter dataset.

    Output index meaning:
      0 -> integrated profile
      1 -> first subpulse
      2 -> second subpulse
      ...
    """

    params = _as_polarimetry_precompute(data, on_pulse)
    dataset = [_polarisation_entry(params, idx) for idx in range(params.num_pulses + 1)]
    return dataset, params.phase_axis


def get_pulse_parameters(dataset, pulse_index):
    """
    Parameters for a given pulse index.

    pulse_index = 0  -> integrated profile
    pulse_index >= 1 -> individual subpulses
    """

    return dataset[pulse_index]


def _constant_like(values, scalar):
    values = np.asarray(values, dtype=float)
    return np.full_like(values, float(scalar), dtype=float)


def _safe_offpulse_std(values, offpulse_mask):
    values = np.asarray(values, dtype=float)
    if not np.any(offpulse_mask):
        return EPSILON
    finite = values[..., offpulse_mask]
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return EPSILON
    return _safe_scalar(np.nanstd(finite))


def _fraction_error(numerator, denominator, sigma_numerator, sigma_denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > EPSILON)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.sqrt((sigma_numerator / denominator[valid]) ** 2 + ((numerator[valid] * sigma_denominator) / (denominator[valid] ** 2)) ** 2)
    return out


def _angle_error_from_snr(amplitude, sigma_amplitude):
    amplitude = np.asarray(amplitude, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.divide(amplitude, sigma_amplitude, out=np.full_like(amplitude, np.nan, dtype=float), where=float(sigma_amplitude) > EPSILON)
    return _pa_uncertainty_deg_from_l_snr(snr)


def build_polarisation_payload(data, start_phase, end_phase, on_pulse, max_pulses=None, pulse_index=None):
    def _tolist_with_none(arr):
        # Replace NaN/inf with None for valid JSON serialization
        return np.where(np.isfinite(arr), arr, None).tolist()

    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    top_pulse_power = get_top_pulse_power_summary(params, 10)

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase, end_side="right")

    # Early return for invalid phase range
    if start_idx >= end_idx:
        return {
            "start_phase": float(start_phase),
            "end_phase": float(end_phase),
            "on_pulse": {"start": float(params.on_pulse[0]), "end": float(params.on_pulse[1])},
            "phase_axis": [],
            "num_pulses": int(params.num_pulses),
            "top_pulse_power": top_pulse_power,
            "dataset": [],
            "warning": "No phase bins selected; adjust start_phase/end_phase",
        }

    phase_slice = phase_axis[start_idx:end_idx]
    total_subpulses = params.num_pulses
    pulse_energies = get_pulse_energies(params)
    top_pulse_indices = []
    pulse_selection = "all"

    # Determine which pulse indices to include. 0 is the integrated profile;
    # 1..num_pulses are individual subpulses.
    if pulse_index is not None:
        indices = [int(pulse_index)]
        pulse_selection = "selected"
    elif max_pulses is None:
        indices = range(total_subpulses + 1)
    else:
        max_pulses = max(0, min(int(max_pulses), total_subpulses))
        top_pulse_indices = get_top_pulse_indices(params, max_pulses).tolist()
        indices = [0] + [idx + 1 for idx in top_pulse_indices]
        pulse_selection = "top_power"

    # Build payload efficiently
    payload_dataset = []
    for idx in indices:
        entry = _polarisation_entry(params, idx)
        sliced = {"pulse_index": idx}
        if idx > 0:
            pulse_zero_index = idx - 1
            sliced["pulse_number"] = int(pulse_zero_index)
            sliced["pulse_power"] = float(pulse_energies[pulse_zero_index])
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                sliced[key] = _tolist_with_none(val[start_idx:end_idx])
            else:
                sliced[key] = val
        payload_dataset.append(sliced)

    return {
        "start_phase": float(start_phase),
        "end_phase": float(end_phase),
        "on_pulse": {"start": float(params.on_pulse[0]), "end": float(params.on_pulse[1])},
        "phase_axis": phase_slice.tolist(),
        "num_pulses": int(params.num_pulses),
        "pulse_selection": pulse_selection,
        "top_pulse_indices": [int(idx) for idx in top_pulse_indices],
        "top_pulse_power": top_pulse_power,
        "dataset": payload_dataset,
    }
