from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
from test import plot_poincare_aitoff_at_phase, return_xyz_interactive_poincare_sphere
from test import get_all_profiles
from test import plot_all_heatmaps
from test import (
    plot_phase_slice_histograms_by_phase,
    polarisation_histogram_single,
    build_polarisation_payload,
    precompute_polarimetry,
    precompute_stokes,
)
from test import plot_polarisation_stacks
from fastapi.responses import JSONResponse
import asyncio
import hashlib
import psutil
import threading
from collections import OrderedDict
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import ProcessCollector

try:
    ProcessCollector()
except ValueError:
    # Uvicorn reloads or imported test runs can register this collector already.
    pass

app = FastAPI(title="Pulsar Polarimetry API")

# Instrument the app for Prometheus metrics
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pulsar-p-re-spidar-react-js.vercel.app", "https://psrweb.jb.man.ac.uk"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CACHE_ITEMS = 3
_DATA_CACHE = OrderedDict()
_STOKES_CACHE = OrderedDict()
_POLARIMETRY_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _cache_get(cache, key):
    with _CACHE_LOCK:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value


def _cache_set(cache, key, value, max_items=MAX_CACHE_ITEMS):
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)


def _cache_sizes():
    with _CACHE_LOCK:
        return {
            "raw_data": len(_DATA_CACHE),
            "stokes": len(_STOKES_CACHE),
            "polarimetry": len(_POLARIMETRY_CACHE),
        }


def _normalise_on_pulse(on_pulse):
    return (float(on_pulse[0]), float(on_pulse[1]))


def _load_numpy_from_bytes(content):
    loaded = np.load(io.BytesIO(content))
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = loaded.files[0]
            return loaded[key]
        finally:
            loaded.close()
    return loaded


async def load_numpy_data(file: UploadFile):
    """Load and parse numpy file asynchronously, reusing cached arrays when possible."""
    data, _ = await load_numpy_data_with_key(file)
    return data


async def load_numpy_data_with_key(file: UploadFile):
    content = await file.read()
    data_key = hashlib.sha256(content).hexdigest()
    cached = _cache_get(_DATA_CACHE, data_key)
    if cached is not None:
        return cached, data_key

    data = await asyncio.to_thread(_load_numpy_from_bytes, content)
    _cache_set(_DATA_CACHE, data_key, data)
    return data, data_key


async def load_stokes_precompute(file: UploadFile):
    data, data_key = await load_numpy_data_with_key(file)
    cached = _cache_get(_STOKES_CACHE, data_key)
    if cached is not None:
        return cached, data_key

    stokes = await asyncio.to_thread(precompute_stokes, data)
    _cache_set(_STOKES_CACHE, data_key, stokes)
    return stokes, data_key


async def load_polarimetry_precompute(file: UploadFile, on_pulse):
    stokes, data_key = await load_stokes_precompute(file)
    on_pulse = _normalise_on_pulse(on_pulse)
    cache_key = (data_key, on_pulse)
    cached = _cache_get(_POLARIMETRY_CACHE, cache_key)
    if cached is not None:
        return cached, data_key

    polarimetry = await asyncio.to_thread(precompute_polarimetry, stokes, on_pulse)
    _cache_set(_POLARIMETRY_CACHE, cache_key, polarimetry)
    return polarimetry, data_key


def _serialise_profile(profile):
    return {"x": profile["x"].tolist(), "y": profile["y"].tolist()}


def _serialise_heatmap(heatmap):
    return {
        "pulse_phase": heatmap["pulse_phase"].tolist(),
        "pulse_number": heatmap["pulse_number"].tolist(),
        "heatmap_data": heatmap["heatmap_data"].tolist(),
        "vmin": heatmap["vmin"],
        "vmax": heatmap["vmax"],
        "label": heatmap["label"],
        "obs_id": heatmap["obs_id"],
    }


def _bytes_to_mb(value):
    return round(value / 1024 / 1024, 2)


def _optional_mb(obj, attr):
    value = getattr(obj, attr, None)
    return _bytes_to_mb(value) if value is not None else None


def _process_io_stats(process):
    try:
        io_counters = process.io_counters()
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return None

    return {
        "read_count": getattr(io_counters, "read_count", None),
        "write_count": getattr(io_counters, "write_count", None),
        "read_mb": _optional_mb(io_counters, "read_bytes"),
        "write_mb": _optional_mb(io_counters, "write_bytes"),
    }


def _process_handle_count(process):
    try:
        return process.num_handles()
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        pass

    try:
        return process.num_fds()
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return None


@app.get("/", summary="Health check")
async def root() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/stats", summary="Get app process stats")
async def get_stats():
    # Get process info for this app
    process = psutil.Process()
    process_memory = process.memory_info()
    try:
        process_memory_full = process.memory_full_info()
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        process_memory_full = None

    process_cpu = process.cpu_percent(interval=0.1)
    system_cpu = psutil.cpu_percent(interval=None)
    system_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    disk_usage = psutil.disk_usage(".")

    # Get process threads
    threads = process.num_threads()

    # Get process open files/connections (if any)
    try:
        open_files = len(process.open_files())
    except:
        open_files = 0

    try:
        connections = len(process.net_connections())
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        connections = 0

    return {
        "process_memory_rss_mb": _bytes_to_mb(process_memory.rss),
        "process_memory_vms_mb": _bytes_to_mb(process_memory.vms),
        "process_memory_uss_mb": _optional_mb(process_memory_full, "uss") if process_memory_full else None,
        "process_memory_percent": round(process.memory_percent(), 2),
        "process_cpu_percent": process_cpu,
        "process_threads": threads,
        "process_open_files": open_files,
        "process_connections": connections,
        "process_handles": _process_handle_count(process),
        "process_io": _process_io_stats(process),
        "process_status": process.status(),
        "process_create_time": process.create_time(),
        "system_cpu_percent": system_cpu,
        "system_cpu_count_logical": psutil.cpu_count(logical=True),
        "system_cpu_count_physical": psutil.cpu_count(logical=False),
        "system_memory": {
            "total_mb": _bytes_to_mb(system_memory.total),
            "available_mb": _bytes_to_mb(system_memory.available),
            "used_mb": _bytes_to_mb(system_memory.used),
            "percent": system_memory.percent,
        },
        "system_swap": {
            "total_mb": _bytes_to_mb(swap_memory.total),
            "used_mb": _bytes_to_mb(swap_memory.used),
            "free_mb": _bytes_to_mb(swap_memory.free),
            "percent": swap_memory.percent,
        },
        "disk_usage": {
            "total_mb": _bytes_to_mb(disk_usage.total),
            "used_mb": _bytes_to_mb(disk_usage.used),
            "free_mb": _bytes_to_mb(disk_usage.free),
            "percent": disk_usage.percent,
        },
        "cache_items": _cache_sizes(),
    }

@app.post("/export_poincare_data", summary="Fetch pulsar details")
async def export_poincare_data(file: UploadFile = File(...), start_phase: float = 0.0, end_phase: float = 1.0, on_pulse_start: float = 0.0, on_pulse_end: float = 1.0):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file, on_pulse)

    response = await asyncio.to_thread(
        return_xyz_interactive_poincare_sphere,
        precomputed, start_phase, end_phase, on_pulse, file.filename
    )
    return {"x": response[0].tolist(), "y": response[1].tolist(), "z": response[2].tolist()}

@app.post("/export_profiles", summary="Fetch profiles")
async def export_profiles(file: UploadFile = File(...), start_phase: float = 0.0, end_phase: float = 1.0):
    precomputed, _ = await load_stokes_precompute(file)

    profiles = await asyncio.to_thread(get_all_profiles, precomputed, start_phase, end_phase)

    return {label: _serialise_profile(profiles[label]) for label in ("I", "Q", "U", "V")}

@app.post("/export_heatmaps", summary="Fetch heatmaps")
async def export_heatmaps(file: UploadFile = File(...), start_phase: float = 0.0, end_phase: float = 1.0):
    precomputed, _ = await load_stokes_precompute(file)

    obs_id = file.filename

    heatmaps = await asyncio.to_thread(plot_all_heatmaps, precomputed, start_phase, end_phase, obs_id)

    return {label: _serialise_heatmap(heatmaps[label]) for label in ("I", "Q", "U", "V")}

@app.post("/poincare_sphere_aitoff_fixedphase", summary="Fetch Poincare sphere data for Aitoff projection with fixed phase value")
async def poincare_sphere_aitoff_fixedphase(
    file: UploadFile = File(...),
    phase_value: float = 0.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    obs_id: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file, on_pulse)
    lon_arr, lat_array = await asyncio.to_thread(
        plot_poincare_aitoff_at_phase, precomputed, on_pulse, phase_value, obs_id or "uploaded"
    )

    return {"lon": lon_arr.tolist(), "lat": lat_array.tolist()}


@app.post("/phase_slice_histograms", summary="Phase-slice histograms for multiple polarisation quantities")
async def phase_slice_histograms(
    file: UploadFile = File(...),
    left_phase: float = 0.0,
    mid_phase: float = 0.5,
    right_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    default_bins: int = 200,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file, on_pulse)
    payload = await asyncio.to_thread(
        plot_phase_slice_histograms_by_phase,
        precomputed,
        left_phase,
        mid_phase,
        right_phase,
        on_pulse,
        file.filename,
        default_bins,
        True,
    )

    return JSONResponse(content=payload)


@app.post(
    "/polarisation_preprocess",
    summary="Preprocess Poincare-sphere coords and polarisation fractions/angles",
)
async def polarisation_preprocess(
    file: UploadFile = File(...),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    max_pulses: int | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file, on_pulse)
    payload = await asyncio.to_thread(
        build_polarisation_payload,
        precomputed,
        start_phase,
        end_phase,
        on_pulse,
        max_pulses,
    )

    return JSONResponse(content=payload)

# One route that serves a single quantity; you can call it for each of the 8 quantities
# quantity values: PA, EA, P/I, L/I, |V/I|, V/I, I, dPA
@app.post("/polarisation_histogram", summary="Single polarisation histogram for one quantity")
async def polarisation_histogram_single_endpoint(
    quantity: str,
    file: UploadFile = File(...),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    base_quantity_bins: int = 200,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file, on_pulse)
    payload = await asyncio.to_thread(
        polarisation_histogram_single,
        precomputed,
        start_phase,
        end_phase,
        on_pulse,
        file.filename,
        quantity,
        base_quantity_bins,
    )

    return JSONResponse(content=payload)


@app.post("/polarisation_stacks", summary="Pulse-phase stacks for polarisation quantities")
async def polarisation_stacks_endpoint(
    file: UploadFile = File(...),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file, on_pulse)
    payload = await asyncio.to_thread(
        plot_polarisation_stacks,
        precomputed,
        start_phase,
        end_phase,
        on_pulse,
        file.filename,
        True,
    )

    return JSONResponse(content=payload)

