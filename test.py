import numpy as np
from dataclasses import dataclass

DEBIAS_THRESHOLD = 1.57
EPSILON = 1e-6
STOKES_LABELS = ("I", "Q", "U", "V")


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
        off_pulse_std = _safe_scalar(np.std(base.I_mean[off_pulse_mask]))
        pulse_off_pulse_std = np.std(base.I[:, off_pulse_mask], axis=1, keepdims=True)
    else:
        off_pulse_std = EPSILON
        pulse_off_pulse_std = np.full((base.num_pulses, 1), EPSILON)
    pulse_off_pulse_std = np.maximum(pulse_off_pulse_std, EPSILON)

    threshold = float(np.min(base.I_mean[on_pulse_mask])) if np.any(on_pulse_mask) else float(np.min(base.I_mean))
    if np.any(on_pulse_mask):
        pulse_threshold = np.min(base.I[:, on_pulse_mask], axis=1, keepdims=True)
    else:
        pulse_threshold = np.min(base.I, axis=1, keepdims=True)

    L = np.sqrt(base.Q ** 2 + base.U ** 2)
    P = np.sqrt(base.Q ** 2 + base.U ** 2 + base.V ** 2)
    L_sigma = L / off_pulse_std
    P_sigma = P / off_pulse_std
    L_mask = L_sigma >= DEBIAS_THRESHOLD
    P_mask = P_sigma >= DEBIAS_THRESHOLD
    L_true = _debias_polarisation(L, off_pulse_std)
    P_true = _debias_polarisation(P, off_pulse_std)

    p_frac = _fraction(P_true, base.I, threshold)
    l_frac = _fraction(L_true, base.I, threshold)
    v_frac = _fraction(base.V, base.I, threshold)
    abs_vfrac = np.abs(v_frac)
    PA_rad = 0.5 * np.arctan2(base.U, base.Q)
    PA_deg = np.degrees(PA_rad)
    EA_rad = 0.5 * np.arctan2(base.V, L_true)
    EA_deg = np.degrees(EA_rad)
    dPA_dphi = _normalised_gradient(PA_deg, base.phase_axis, axis=-1)

    pulse_L_true = _debias_polarisation(L, pulse_off_pulse_std)
    pulse_P_true = _debias_polarisation(P, pulse_off_pulse_std)
    pulse_p_frac = _fraction(pulse_P_true, base.I, pulse_threshold)
    pulse_l_frac = _fraction(pulse_L_true, base.I, pulse_threshold)
    pulse_v_frac = _fraction(base.V, base.I, pulse_threshold)
    pulse_abs_vfrac = np.abs(pulse_v_frac)
    pulse_EA_rad = 0.5 * np.arctan2(base.V, pulse_L_true)
    pulse_EA_deg = np.degrees(pulse_EA_rad)
    pulse_dPA_dphi = _normalised_gradient(PA_deg, base.phase_axis, axis=-1, per_profile=True)
    pulse_lon = 2 * PA_rad
    pulse_lat = 2 * pulse_EA_rad
    pulse_cos_lat = np.cos(pulse_lat)
    pulse_x = pulse_cos_lat * np.cos(pulse_lon)
    pulse_y = pulse_cos_lat * np.sin(pulse_lon)
    pulse_z = np.sin(pulse_lat)
    pulse_radius_of_curvature = _radius_of_curvature_from_xyz(pulse_x, pulse_y, pulse_z)

    mean_L = np.sqrt(base.Q_mean ** 2 + base.U_mean ** 2)
    mean_P = np.sqrt(base.Q_mean ** 2 + base.U_mean ** 2 + base.V_mean ** 2)
    mean_L_sigma = mean_L / off_pulse_std
    mean_P_sigma = mean_P / off_pulse_std
    mean_L_mask = mean_L_sigma >= DEBIAS_THRESHOLD
    mean_P_mask = mean_P_sigma >= DEBIAS_THRESHOLD
    mean_L_true = _debias_polarisation(mean_L, off_pulse_std)
    mean_P_true = _debias_polarisation(mean_P, off_pulse_std)
    mean_p_frac = _fraction(mean_P_true, base.I_mean, threshold)
    mean_l_frac = _fraction(mean_L_true, base.I_mean, threshold)
    mean_v_frac = _fraction(base.V_mean, base.I_mean, threshold)
    mean_abs_vfrac = np.abs(mean_v_frac)
    mean_PA_rad = 0.5 * np.arctan2(base.U_mean, base.Q_mean)
    mean_PA_deg = np.degrees(mean_PA_rad)
    mean_EA_rad = 0.5 * np.arctan2(base.V_mean, mean_L_true)
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
        I=base.I,
        Q=base.Q,
        U=base.U,
        V=base.V,
        I_mean=base.I_mean,
        Q_mean=base.Q_mean,
        U_mean=base.U_mean,
        V_mean=base.V_mean,
        mean_profiles=base.mean_profiles,
        I0=base.I0,
        I_over_I0=base.I_over_I0,
        I_mean_over_I0=base.I_mean_over_I0,
        on_pulse=(default_start, default_end),
        on_pulse_mask=on_pulse_mask,
        off_pulse_mask=off_pulse_mask,
        off_pulse_std=off_pulse_std,
        threshold=threshold,
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

    return lon, lat

def plot_phase_slice_histograms_by_phase(data, left_phase, mid_phase, right_phase, on_pulse, obs_id, default_bins=200, return_data=False):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses

    phase_values = [left_phase, mid_phase, right_phase]
    phase_bins = [np.argmin(np.abs(phase_axis - val)) for val in phase_values]

    quantities = [params.p_frac, params.l_frac, params.abs_vfrac, params.v_frac, params.PA_deg, params.EA_deg]
    quantity_names = ["P/I", "L/I", "|V/I|", "V/I", "PA [deg]", "EA [deg]"]

    def compute_bin_count(values):
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

        fig, axs = plt.subplots(len(quantities), len(phase_bins), figsize=(20, 15), constrained_layout=True)

    for row_idx, (quantity, name) in enumerate(zip(quantities, quantity_names)):
        phase_entries = []
        for col_idx, (phase_bin, phase_val) in enumerate(zip(phase_bins, phase_values)):
            values = quantity[:, phase_bin]
            bin_count = compute_bin_count(values)
            vmin = float(values.min())
            vmax = float(values.max())
            if vmin == vmax:
                pad = max(abs(vmin) * 0.1, 0.5)
                vmin -= pad
                vmax += pad

            if return_data:
                counts, edges = np.histogram(values, bins=bin_count, range=(vmin, vmax))
                x_limits = None
                if row_idx < 3:
                    x_limits = [0.0, 1.0]
                elif row_idx == 3:
                    x_limits = [-1.0, 1.0]

                phase_entries.append({
                    "phase_value": float(phase_val),
                    "phase_bin_index": int(phase_bin),
                    "bin_edges": edges.tolist(),
                    "counts": counts.tolist(),
                    "x_limits": x_limits,
                    "stats": {
                        "min": vmin,
                        "max": vmax,
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "num_pulses": int(num_pulses),
                    },
                })
                continue

            ax = axs[row_idx, col_idx]
            ax.hist(values, bins=bin_count, color='steelblue', alpha=0.8)
            ax.set_title(f"{name}\nPhase = {phase_val:.2f}")
            ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

            if row_idx < 3:
                ax.set_xlim(0, 1)
            if row_idx == 3:
                ax.set_xlim(-1, 1)

        if return_data:
            result["quantities"].append({
                "name": name,
                "phase_slices": phase_entries,
            })

    if return_data:
        return result

# New: compute a single polarisation histogram payload for one quantity
def polarisation_histogram_single(data, start_phase, end_phase, on_pulse, obs_id, quantity_key, base_quantity_bins=200):
    """
    quantity_key in {"PA", "EA", "P/I", "L/I", "|V/I|", "V/I", "I", "dPA"}
    """
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses

    quantity_map = {
        "PA": (params.PA_deg, "PA [deg]", False),
        "EA": (params.EA_deg, "EA [deg]", False),
        "P/I": (params.p_frac, "P/I", True),
        "L/I": (params.l_frac, "L/I", True),
        "|V/I|": (params.abs_vfrac, "|V/I|", True),
        "V/I": (params.v_frac, "V/I", True),
        "I": (params.I, "I", False),
        "dPA": (params.dPA_dphi, "Normalised PA Derivative", False),
    }
    if quantity_key not in quantity_map:
        return {"error": f"Unknown quantity {quantity_key}"}

    quantity, label, is_fraction = quantity_map[quantity_key]

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase, end_side="right")
    selected_phase_axis = phase_axis[start_idx:end_idx]
    selected_phase_bins = end_idx - start_idx

    default_start, default_end = params.on_pulse
    max_I = np.max(params.I_mean)
    lowfrac = params.threshold / max_I if max_I != 0 else 0
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
    if q.size == 0:
        q_min, q_max = 0.0, 1.0
    else:
        q_min, q_max = np.min(q), np.max(q)
        if q_min == q_max:
            q_max = q_min + 1e-3

    hist2d = np.zeros((quantity_bins, selected_phase_bins))
    for i in range(selected_phase_bins):
        row = q[i] if q.size else np.array([])
        row = row[np.isfinite(row)]
        if row.size == 0:
            hist = np.zeros(quantity_bins)
            bin_edges = np.linspace(q_min, q_max, quantity_bins + 1)
            hist2d[:, i] = hist
            continue
        if is_fraction:
            nonzero_values = row[np.abs(row) >= lowfrac]
            if len(nonzero_values) > 0:
                hist, bin_edges = np.histogram(nonzero_values, bins=quantity_bins, range=(q_min, q_max))
            else:
                hist = np.zeros(quantity_bins)
                bin_edges = np.linspace(q_min, q_max, quantity_bins + 1)
        else:
            hist, bin_edges = np.histogram(row, bins=quantity_bins, range=(q_min, q_max))
        hist2d[:, i] = hist

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    extent = [selected_phase_axis[0], selected_phase_axis[-1], bin_centers[0], bin_centers[-1]]
    log_hist2d = np.zeros_like(hist2d, dtype=float)
    mask = hist2d >= 1
    log_hist2d[mask] = np.log(hist2d[mask])

    return {
        "obs_id": obs_id,
        "start_phase": float(start_phase),
        "end_phase": float(end_phase),
        "on_pulse": {"start": float(default_start), "end": float(default_end)},
        "quantity": label,
        "quantity_key": quantity_key,
        "is_fraction": is_fraction,
        "quantity_bins": int(quantity_bins),
        "phase_axis": selected_phase_axis.tolist(),
        "hist2d": hist2d.tolist(),
        "log_hist2d": log_hist2d.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "extent": [float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])],
        "q_min": float(q_min),
        "q_max": float(q_max),
        "lowfrac": float(lowfrac),
        "num_pulses": int(num_pulses),
    }

def plot_polarisation_stacks(data, start_phase, end_phase, on_pulse, obs_id, return_data=False):
    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis
    num_pulses = params.num_pulses
    quantities = [params.PA_deg, params.EA_deg, params.p_frac, params.l_frac, params.abs_vfrac, params.v_frac]
    labels = ["PA [deg]", "EA [deg]", "P/I", "L/I", "|V/I|", "V/I"]

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
        for idx, (quantity, label) in enumerate(zip(quantities, labels)):
            q = quantity[:, start_idx:end_idx]
            q_min, q_max = np.nanmin(q), np.nanmax(q)
            if q_min == q_max:
                pad = max(abs(q_min) * 0.1, 1e-3)
                q_min -= pad
                q_max += pad

            payload["quantities"].append({
                "name": label,
                "data": q.tolist(),
                "vmin": float(q_min),
                "vmax": float(q_max),
            })
        return payload


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
    if pulse_index == 0:
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
        )

    pulse_idx = pulse_index - 1
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


def build_polarisation_payload(data, start_phase, end_phase, on_pulse, max_pulses=None):
    def _tolist_with_none(arr):
        # Replace NaN/inf with None for valid JSON serialization
        return np.where(np.isfinite(arr), arr, None).tolist()

    params = _as_polarimetry_precompute(data, on_pulse)
    phase_axis = params.phase_axis

    start_idx, end_idx = _phase_bounds(phase_axis, start_phase, end_phase, end_side="right")

    # Early return for invalid phase range
    if start_idx >= end_idx:
        return {
            "start_phase": float(start_phase),
            "end_phase": float(end_phase),
            "on_pulse": {"start": float(params.on_pulse[0]), "end": float(params.on_pulse[1])},
            "phase_axis": [],
            "num_pulses": int(params.num_pulses),
            "dataset": [],
            "warning": "No phase bins selected; adjust start_phase/end_phase",
        }

    phase_slice = phase_axis[start_idx:end_idx]
    total_subpulses = params.num_pulses

    # Determine which pulse indices to include
    if max_pulses is None:
        indices = range(total_subpulses + 1)
    else:
        max_pulses = max(0, min(int(max_pulses), total_subpulses))
        indices = [0] + list(range(1, max_pulses + 1))

    # Build payload efficiently
    payload_dataset = []
    for idx in indices:
        entry = _polarisation_entry(params, idx)
        sliced = {"pulse_index": idx}
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
        "dataset": payload_dataset,
    }
