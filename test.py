import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# Helper function to compute common polarisation parameters efficiently
def compute_common_stokes_params(data, phase_axis, on_pulse):
    """Compute commonly used Stokes parameters efficiently to avoid redundant calculations"""
    I = data[:, 0, :]
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]
    
    default_start, default_end = on_pulse
    on_pulse_mask = (phase_axis >= default_start) & (phase_axis <= default_end)
    off_pulse_mask = ~on_pulse_mask
    
    I_mean = I.mean(axis=0)
    off_pulse_std = np.std(I_mean[off_pulse_mask]) if np.any(off_pulse_mask) else 1e-6
    off_pulse_std = off_pulse_std if off_pulse_std != 0 else 1e-6
    
    threshold = np.min(I_mean[on_pulse_mask]) if np.any(on_pulse_mask) else np.min(I_mean)
    
    # Compute L_true
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / off_pulse_std
    mask = L_sigma >= 1.57
    L_true[mask] = off_pulse_std * np.sqrt(L_sigma[mask]**2 - 1)
    
    # Compute P_true
    P = np.sqrt(Q**2 + U**2 + V**2)
    P_sigma = P / off_pulse_std
    P_true = np.zeros_like(P)
    mask = P_sigma >= 1.57
    P_true[mask] = off_pulse_std * np.sqrt(P_sigma[mask]**2 - 1)
    
    return {
        'I': I, 'Q': Q, 'U': U, 'V': V,
        'I_mean': I_mean,
        'L_true': L_true,
        'P_true': P_true,
        'off_pulse_std': off_pulse_std,
        'threshold': threshold,
        'on_pulse_mask': on_pulse_mask,
        'off_pulse_mask': off_pulse_mask
    }

def return_xyz_interactive_poincare_sphere(data, start_phase, end_phase, on_pulse, obs_id):
    num_pulses, _, num_bins = data.shape

    print(f"Number of pulses: {num_pulses}, Number of bins: {num_bins}")
    phase_axis = np.linspace(0, 1, num_bins)

    I = data[:, 0, :].mean(axis=0)
    Q = data[:, 1, :].mean(axis=0)
    U = data[:, 2, :].mean(axis=0)
    V = data[:, 3, :].mean(axis=0)

    default_start, default_end = on_pulse
    on_pulse_mask = (phase_axis >= default_start) & (phase_axis <= default_end)
    off_pulse_mask = ~on_pulse_mask
    sigma_off = np.std(I[off_pulse_mask])
    L = np.sqrt(Q**2 + U**2)
    L_true = np.zeros_like(L)
    L_sigma = L / sigma_off
    mask = L_sigma >= 1.57
    L_true[mask] = sigma_off * np.sqrt(L_sigma[mask]**2 - 1)

    threshold = np.min(I[on_pulse_mask])
    P = np.sqrt(Q**2 + U**2 + V**2)
    P_sigma = P / sigma_off
    P_true = np.zeros_like(P)
    mask = P_sigma >= 1.57
    P_true[mask] = sigma_off * np.sqrt(P_sigma[mask]**2 - 1)
    p_frac = np.where(I >= threshold, P_true/I , 0)

    PA = 0.5 * np.arctan2(U, Q)
    EA = 0.5 * np.arctan2(V, L_true)

    lon = 2 * PA
    lat = 2 * EA
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    start_idx = np.searchsorted(phase_axis, start_phase)
    end_idx = np.searchsorted(phase_axis, end_phase)

    x = x[start_idx:end_idx]
    y = y[start_idx:end_idx]
    z = z[start_idx:end_idx]
    
    return x, y, z # This will be sent to the frontend for interactive plotting

def get_all_profiles(data, start_phase, end_phase):
    """Get all 4 Stokes profiles efficiently in one call"""
    pulse_phase = np.linspace(0, 1, data.shape[2])
    
    profiles = {}
    for idx, label in enumerate(['I', 'Q', 'U', 'V']):
        mean_profile = data[:, idx, :].mean(axis=0)
        profiles[label] = {'x': pulse_phase, 'y': mean_profile}
    
    return profiles

def get_I_profile(data, start_phase, end_phase):
    pulse_phase = np.linspace(0, 1, data.shape[2])
    mean_profile = data[:, 0, :].mean(axis=0)
    return pulse_phase, mean_profile

def get_Q_profile(data, start_phase, end_phase):
    pulse_phase = np.linspace(0, 1, data.shape[2])
    mean_profile = data[:, 1, :].mean(axis=0)
    return pulse_phase, mean_profile

def get_U_profile(data, start_phase, end_phase):
    pulse_phase = np.linspace(0, 1, data.shape[2])
    mean_profile = data[:, 2, :].mean(axis=0)
    return pulse_phase, mean_profile

def get_V_profile(data, start_phase, end_phase):
    pulse_phase = np.linspace(0, 1, data.shape[2])
    mean_profile = data[:, 3, :].mean(axis=0)
    return pulse_phase, mean_profile

# Unified function to compute all heatmaps in one pass
def plot_all_heatmaps(data, start_phase, end_phase, obs_id):
    """Compute all four Stokes heatmaps (I, Q, U, V) efficiently in a single pass."""
    # Compute common parameters once
    pulse_phase = np.linspace(0, 1, data.shape[2])
    pulse_number = np.arange(data.shape[0])
    start_idx = np.searchsorted(pulse_phase, start_phase, side='left')
    end_idx = np.searchsorted(pulse_phase, end_phase, side='right')
    phase_slice = pulse_phase[start_idx:end_idx]
    
    # Compute all four Stokes heatmaps
    heatmaps = {}
    labels = ['I', 'Q', 'U', 'V']
    
    for stokes_idx, label in enumerate(labels):
        heatmap_data = data[:, stokes_idx, start_idx:end_idx]
        vmin = heatmap_data.min()
        vmax = heatmap_data.max()
        
        heatmaps[label] = {
            'pulse_phase': phase_slice,
            'pulse_number': pulse_number,
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
    num_pulses, _, num_bins = data.shape
    phase_axis = np.linspace(0, 1, num_bins)
    cbin = np.argmin(np.abs(phase_axis - cphase))

    default_start, default_end = on_pulse
    on_pulse_mask = (phase_axis >= default_start) & (phase_axis <= default_end)
    off_pulse_mask = ~on_pulse_mask

    # Vectorized computation for all pulses at once
    I = data[:, 0, :]  # shape: (num_pulses, num_bins)
    Q = data[:, 1, :]
    U = data[:, 2, :]
    V = data[:, 3, :]
    
    # Compute off_pulse_std for each pulse
    off_pulse_std = np.std(I[:, off_pulse_mask], axis=1, keepdims=True)  # shape: (num_pulses, 1)
    off_pulse_std = np.where(off_pulse_std == 0, 1e-6, off_pulse_std)
    
    # Vectorized L_true computation
    L = np.sqrt(Q**2 + U**2)
    L_sigma = L / off_pulse_std
    L_true = np.zeros_like(L)
    mask = L_sigma >= 1.57
    L_true[mask] = off_pulse_std.repeat(num_bins, axis=1)[mask] * np.sqrt(L_sigma[mask]**2 - 1)
    
    # Vectorized PA and EA
    PA = 0.5 * np.arctan2(U, Q)
    EA = 0.5 * np.arctan2(V, L_true)
    
    # Extract values at the specific bin for all pulses
    pa_val = PA[:, cbin]
    ea_val = EA[:, cbin]
    
    lon = 2 * pa_val
    lat = 2 * ea_val
    lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi

    return lon, lat

def plot_phase_slice_histograms_by_phase(data, left_phase, mid_phase, right_phase, on_pulse, obs_id, default_bins=200, return_data=False):
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)
    
    phase_values = [left_phase, mid_phase, right_phase]
    phase_bins = [np.argmin(np.abs(phase_axis - val)) for val in phase_values]

    # Use optimized helper function
    params = compute_common_stokes_params(data, phase_axis, on_pulse)
    I = params['I']
    threshold = params['threshold']
    L_true = params['L_true']
    P_true = params['P_true']
    Q = params['Q']
    U = params['U']
    V = params['V']

    p_frac = np.where(I >= threshold, P_true / I, 0)
    l_frac = np.where(I >= threshold, L_true / I, 0)
    v_frac = np.where(I >= threshold, V / I, 0)
    absv_frac = np.where(I >= threshold, np.abs(V / I), 0)
    PA = 0.5 * np.arctan2(U, Q) * 180 / np.pi
    EA = 0.5 * np.arctan2(V, L_true) * 180 / np.pi

    quantities = [p_frac, l_frac, absv_frac, v_frac, PA, EA]
    quantity_names = ["P/I", "L/I", "|V/I|", "V/I", "PA [deg]", "EA [deg]"]

    def compute_bin_count(values):
        val_iqr = iqr(values)
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
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Use optimized helper function
    params = compute_common_stokes_params(data, phase_axis, on_pulse)
    I = params['I']
    Q = params['Q']
    U = params['U']
    V = params['V']
    L_true = params['L_true']
    P_true = params['P_true']
    threshold = params['threshold']
    I_mean = params['I_mean']
    
    p_frac = np.where(I >= threshold, P_true/I , 0)
    l_frac = np.where(I >= threshold, L_true / I, 0)
    v_frac = np.where(I >= threshold, V / I, 0)
    absv_frac = np.where(I >= threshold, np.abs(V / I), 0)
    PA = 0.5 * np.arctan2(U, Q) * 180 / np.pi
    EA = 0.5 * np.arctan2(V, L_true) * 180 / np.pi
    dPA_dphi = np.gradient(PA, phase_axis, axis=-1)
    dPA_dphi = dPA_dphi / (np.max(np.abs(dPA_dphi)) if np.max(np.abs(dPA_dphi)) != 0 else 1)

    quantity_map = {
        "PA": (PA, "PA [deg]", False),
        "EA": (EA, "EA [deg]", False),
        "P/I": (p_frac, "P/I", True),
        "L/I": (l_frac, "L/I", True),
        "|V/I|": (absv_frac, "|V/I|", True),
        "V/I": (v_frac, "V/I", True),
        "I": (I, "I", False),
        "dPA": (dPA_dphi, "Normalised PA Derivative", False),
    }
    if quantity_key not in quantity_map:
        return {"error": f"Unknown quantity {quantity_key}"}

    quantity, label, is_fraction = quantity_map[quantity_key]

    start_idx = np.searchsorted(phase_axis, start_phase, side='left')
    end_idx = np.searchsorted(phase_axis, end_phase, side='right')
    selected_phase_axis = phase_axis[start_idx:end_idx]
    selected_phase_bins = end_idx - start_idx

    default_start, default_end = on_pulse
    max_I = np.max(I_mean)
    lowfrac = threshold/max_I if max_I != 0 else 0
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
    num_pulses, _, num_phase_bins = data.shape
    phase_axis = np.linspace(0, 1, num_phase_bins)

    # Use optimized helper function
    params = compute_common_stokes_params(data, phase_axis, on_pulse)
    I = params['I']
    Q = params['Q']
    U = params['U']
    V = params['V']
    L_true = params['L_true']
    P_true = params['P_true']
    threshold = params['threshold']

    p_frac = np.where(I >= threshold, P_true / I, 0)
    l_frac = np.where(I >= threshold, L_true / I, 0)
    v_frac = np.where(I >= threshold, V / I, 0)
    absv_frac = np.where(I >= threshold, np.abs(V / I), 0)
    PA = 0.5 * np.arctan2(U, Q) * 180 / np.pi
    EA = 0.5 * np.arctan2(V, L_true) * 180 / np.pi
    quantities = [PA, EA, p_frac, l_frac, absv_frac, v_frac]
    labels = ["PA [deg]", "EA [deg]", "P/I", "L/I", "|V/I|", "V/I"]

    start_idx = np.searchsorted(phase_axis, start_phase)
    end_idx = np.searchsorted(phase_axis, end_phase)
    selected_phase_axis = phase_axis[start_idx:end_idx]

    if return_data:
        default_start, default_end = on_pulse
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


def _debias_polarisation(amplitude, sigma_off, threshold=1.57):
    """Helper function to debias polarisation amplitudes."""
    debiased = np.zeros_like(amplitude)
    sigma_ratio = amplitude / sigma_off
    mask = sigma_ratio >= threshold
    debiased[mask] = sigma_off * np.sqrt(sigma_ratio[mask] ** 2 - 1)
    return debiased


def compute_polarisation_parameters(I, Q, U, V, phase_axis, on_pulse):
    start, end = on_pulse
    on_mask = (phase_axis >= start) & (phase_axis <= end)
    off_mask = ~on_mask

    sigma_off = np.std(I[off_mask]) if np.any(off_mask) else 1e-6
    sigma_off = max(sigma_off, 1e-6)  # Ensure numerical stability
    threshold = np.min(I[on_mask]) if np.any(on_mask) else np.min(I)

    # Compute Stokes amplitudes
    Q_sq, U_sq, V_sq = Q ** 2, U ** 2, V ** 2
    L = np.sqrt(Q_sq + U_sq)
    P = np.sqrt(Q_sq + U_sq + V_sq)

    # Debias using helper function
    L_true = _debias_polarisation(L, sigma_off)
    P_true = _debias_polarisation(P, sigma_off)

    # Compute fractions efficiently
    I_safe = np.where(I >= threshold, I, np.inf)  # Avoid division by zero
    p_frac = np.where(I >= threshold, P_true / I_safe, 0.0)
    l_frac = np.where(I >= threshold, L_true / I_safe, 0.0)
    v_frac = np.where(I >= threshold, V / I_safe, 0.0)
    absv_frac = np.abs(v_frac)

    # Compute angles
    PA_rad = 0.5 * np.arctan2(U, Q)
    EA_rad = 0.5 * np.arctan2(V, L_true)
    PA = PA_rad * 180 / np.pi
    EA = EA_rad * 180 / np.pi

    # Normalize PA derivative
    dPA = np.gradient(PA, phase_axis)
    max_dPA = np.nanmax(np.abs(dPA))
    if max_dPA > 0:
        dPA /= max_dPA

    # Compute Poincare sphere coordinates (optimized)
    lon = 2 * PA_rad
    lat = 2 * EA_rad
    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)

    # Compute radius of curvature
    points = np.column_stack((x, y, z))
    roc = np.full(len(points), np.nan)
    for i in range(len(points) - 2):
        roc[i + 1] = find_radius(points[i:i + 3])

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


def build_polarisation_dataset(data, on_pulse):
    """
    Build derived-parameter dataset.

    Output index meaning:
      0 -> integrated profile
      1 -> first subpulse
      2 -> second subpulse
      ...
    """

    npulse, _, nphase = data.shape
    phase_axis = np.linspace(0, 1, nphase)

    # Preallocate list for better performance
    dataset = [None] * (npulse + 1)

    # Compute integrated profile (mean across all pulses)
    I_mean = data[:, 0, :].mean(axis=0)
    Q_mean = data[:, 1, :].mean(axis=0)
    U_mean = data[:, 2, :].mean(axis=0)
    V_mean = data[:, 3, :].mean(axis=0)
    dataset[0] = compute_polarisation_parameters(I_mean, Q_mean, U_mean, V_mean, phase_axis, on_pulse)

    # Compute individual pulse parameters
    for p in range(npulse):
        dataset[p + 1] = compute_polarisation_parameters(
            data[p, 0, :], data[p, 1, :], data[p, 2, :], data[p, 3, :],
            phase_axis, on_pulse
        )

    return dataset, phase_axis


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

    dataset, phase_axis = build_polarisation_dataset(data, on_pulse)

    start_idx = np.searchsorted(phase_axis, start_phase, side="left")
    end_idx = np.searchsorted(phase_axis, end_phase, side="right")

    # Early return for invalid phase range
    if start_idx >= end_idx:
        return {
            "start_phase": float(start_phase),
            "end_phase": float(end_phase),
            "on_pulse": {"start": float(on_pulse[0]), "end": float(on_pulse[1])},
            "phase_axis": [],
            "num_pulses": int(data.shape[0]),
            "dataset": [],
            "warning": "No phase bins selected; adjust start_phase/end_phase",
        }

    phase_slice = phase_axis[start_idx:end_idx]
    total_subpulses = len(dataset) - 1

    # Determine which pulse indices to include
    if max_pulses is None:
        indices = list(range(len(dataset)))
    else:
        max_pulses = max(0, min(int(max_pulses), total_subpulses))
        indices = [0] + list(range(1, max_pulses + 1))

    # Build payload efficiently
    payload_dataset = []
    for idx in indices:
        entry = dataset[idx]
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
        "on_pulse": {"start": float(on_pulse[0]), "end": float(on_pulse[1])},
        "phase_axis": phase_slice.tolist(),
        "num_pulses": int(data.shape[0]),
        "dataset": payload_dataset,
    }