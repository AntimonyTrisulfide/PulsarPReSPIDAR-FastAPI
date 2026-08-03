from urllib import request

from fastapi import FastAPI, HTTPException, Request
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import numpy as np
from test import plot_poincare_aitoff_at_phase, return_xyz_interactive_poincare_sphere
from test import get_all_profiles
from test import plot_all_heatmaps
from test import (
    phase_slice_histogram_single,
    plot_phase_slice_histograms_by_phase,
    polarisation_histogram_single,
    polarisation_stack_single,
    build_polarisation_payload,
    precompute_polarimetry,
    precompute_stokes,
)
from test import iter_polarisation_stacks_json
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import hashlib
import psutil
import threading
from collections import OrderedDict
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import ProcessCollector
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse, urlunparse
from urllib.request import Request as UrlRequest, urlopen

try:
    ProcessCollector()
except ValueError:
    # Uvicorn reloads or imported test runs can register this collector already.
    pass

app = FastAPI(title="Pulsar Polarimetry API")

MEERTIME_HOST = "psrweb.jb.man.ac.uk"
MEERTIME_PATH_PREFIX = "/meertime/singlepulse/"
MEERTIME_ALLOWED_SUFFIXES = (".npz", "/pipeline_info.json")
MEERTIME_CHUNK_SIZE = 1024 * 1024

# Instrument the app for Prometheus metrics
Instrumentator().instrument(app).expose(app)

def _get_cors_origins():
    default_origins = [
        "https://pulsar-p-re-spidar-react-js.vercel.app",
        "https://psrweb.jb.man.ac.uk",
        "http://localhost:5173",
        "http://localhost:4173",
    ]
    configured_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return default_origins + configured_origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Upstream-Authorization",
        "X-MeerTime-Authorization",
    ],
)

MAX_CACHE_ITEMS = max(1, int(os.getenv("MAX_CACHE_ITEMS", "1")))
_DATA_CACHE = OrderedDict()
_STOKES_CACHE = OrderedDict()
_POLARIMETRY_CACHE = OrderedDict()
_DATASET_META_CACHE = OrderedDict()
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
            "metadata": len(_DATASET_META_CACHE),
            "max_cache_items": MAX_CACHE_ITEMS,
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


def _require_file(file: UploadFile | None):
    if file is None:
        raise HTTPException(status_code=400, detail="Either file upload or data_key is required")
    return file


def _require_cached(cache, data_key: str, label: str):
    cached = _cache_get(cache, data_key)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"{label} for data_key {data_key[:12]} is not cached; upload/prepare the dataset again",
        )
    return cached


def _resolve_obs_id(file: UploadFile | None, data_key: str | None, fallback="uploaded"):
    if data_key:
        meta = _cache_get(_DATASET_META_CACHE, data_key)
        if meta and meta.get("filename"):
            return meta["filename"]
    if file is not None and file.filename:
        return file.filename
    return fallback


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


async def load_stokes_precompute_from_data(data, data_key: str):
    cached = _cache_get(_STOKES_CACHE, data_key)
    if cached is not None:
        return cached, data_key

    stokes = await asyncio.to_thread(precompute_stokes, data)
    _cache_set(_STOKES_CACHE, data_key, stokes)
    return stokes, data_key


async def load_stokes_precompute_from_key(data_key: str):
    cached = _cache_get(_STOKES_CACHE, data_key)
    if cached is not None:
        return cached, data_key

    data = _require_cached(_DATA_CACHE, data_key, "Raw dataset")
    return await load_stokes_precompute_from_data(data, data_key)


async def load_stokes_precompute(file: UploadFile | None = None, data_key: str | None = None):
    if data_key:
        return await load_stokes_precompute_from_key(data_key)

    file = _require_file(file)
    data, data_key = await load_numpy_data_with_key(file)
    return await load_stokes_precompute_from_data(data, data_key)


async def load_polarimetry_precompute_from_stokes(stokes, data_key: str, on_pulse):
    on_pulse = _normalise_on_pulse(on_pulse)
    cache_key = (data_key, on_pulse)
    cached = _cache_get(_POLARIMETRY_CACHE, cache_key)
    if cached is not None:
        return cached, data_key

    polarimetry = await asyncio.to_thread(precompute_polarimetry, stokes, on_pulse)
    _cache_set(_POLARIMETRY_CACHE, cache_key, polarimetry)
    return polarimetry, data_key


async def load_polarimetry_precompute(
    file: UploadFile | None = None,
    on_pulse=(0.0, 1.0),
    data_key: str | None = None,
):
    stokes, data_key = await load_stokes_precompute(file=file, data_key=data_key)
    return await load_polarimetry_precompute_from_stokes(stokes, data_key, on_pulse)


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


def _is_allowed_meertime_url(raw_url):
    parsed = urlparse(raw_url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc.lower() != MEERTIME_HOST:
        return False
    if not parsed.path.startswith(MEERTIME_PATH_PREFIX):
        return False
    return parsed.path.endswith(MEERTIME_ALLOWED_SUFFIXES)


def _iter_upstream_chunks(upstream):
    try:
        while True:
            chunk = upstream.read(MEERTIME_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    finally:
        upstream.close()


def _stream_meertime_response(upstream):
    content_type = upstream.headers.get("content-type") or "application/octet-stream"
    status_code = getattr(upstream, "status", getattr(upstream, "code", 200))
    return StreamingResponse(
        _iter_upstream_chunks(upstream),
        status_code=status_code,
        media_type=content_type,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/", summary="Health check")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/meertime-proxy", summary="Proxy allowed MeerTime files")
async def meertime_proxy(request: Request, url: str | None = None):
    if not url:
        raise HTTPException(
            status_code=400,
            detail="Missing url query parameter",
        )

    try:
        is_allowed = _is_allowed_meertime_url(url)
    except ValueError:
        is_allowed = False

    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail="Only MeerTime .npz and pipeline_info.json files are allowed",
        )

    authorization = (
        request.headers.get("x-upstream-authorization")
        or request.headers.get("x-meertime-authorization")
        or request.headers.get("authorization")
    )

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing upstream authorization",
        )

    parsed = urlparse(url)

    # FastAPI decodes %3A in the query parameter. Re-encode the path before
    # sending it to the MeerTime Apache server.
    encoded_path = quote(
        unquote(parsed.path),
        safe="/",
    )

    upstream_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            encoded_path,
            parsed.params,
            parsed.query,
            "",
        )
    )

    upstream_request = UrlRequest(
        upstream_url,
        headers={
            "Authorization": authorization,
            "Accept": "*/*",
            "User-Agent": "MeerTime-FastAPI-Proxy/1.0",
        },
        method="GET",
    )

    try:
        upstream = await asyncio.to_thread(
            urlopen,
            upstream_request,
            timeout=60,
        )

    except HTTPError as error:
        body = error.read()

        print(
            f"MeerTime upstream HTTP error: "
            f"status={error.code}, url={upstream_url}"
        )

        return StreamingResponse(
            io.BytesIO(body),
            status_code=error.code,
            media_type=error.headers.get(
                "content-type",
                "application/octet-stream",
            ),
            headers={
                "Cache-Control": "no-store",
            },
        )

    except (OSError, TimeoutError, URLError) as error:
        print(
            f"MeerTime proxy request failed: "
            f"url={upstream_url}, error={error}"
        )

        raise HTTPException(
            status_code=502,
            detail="MeerTime proxy request failed",
        ) from error

    return _stream_meertime_response(upstream)

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


@app.post("/prepare_dataset", summary="Upload and cache a dataset once for follow-up plot requests")
async def prepare_dataset(
    file: UploadFile = File(...),
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    data, data_key = await load_numpy_data_with_key(file)
    stokes, _ = await load_stokes_precompute_from_data(data, data_key)
    await load_polarimetry_precompute_from_stokes(stokes, data_key, on_pulse)

    meta = {
        "filename": file.filename or data_key,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "on_pulse": {"start": float(on_pulse_start), "end": float(on_pulse_end)},
    }
    _cache_set(_DATASET_META_CACHE, data_key, meta)

    return {
        "data_key": data_key,
        **meta,
        "cache_items": _cache_sizes(),
    }

@app.post("/export_poincare_data", summary="Fetch pulsar details")
async def export_poincare_data(file: UploadFile | None = File(None), start_phase: float = 0.0, end_phase: float = 1.0, on_pulse_start: float = 0.0, on_pulse_end: float = 1.0, data_key: str | None = None):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)

    response = await asyncio.to_thread(
        return_xyz_interactive_poincare_sphere,
        precomputed, start_phase, end_phase, on_pulse, _resolve_obs_id(file, data_key)
    )
    return {"x": response[0].tolist(), "y": response[1].tolist(), "z": response[2].tolist()}

@app.post("/export_profiles", summary="Fetch profiles")
async def export_profiles(file: UploadFile | None = File(None), start_phase: float = 0.0, end_phase: float = 1.0, data_key: str | None = None):
    precomputed, _ = await load_stokes_precompute(file=file, data_key=data_key)

    profiles = await asyncio.to_thread(get_all_profiles, precomputed, start_phase, end_phase)

    return {label: _serialise_profile(profiles[label]) for label in ("I", "Q", "U", "V")}

@app.post("/export_heatmaps", summary="Fetch heatmaps")
async def export_heatmaps(file: UploadFile | None = File(None), start_phase: float = 0.0, end_phase: float = 1.0, data_key: str | None = None):
    precomputed, _ = await load_stokes_precompute(file=file, data_key=data_key)

    obs_id = _resolve_obs_id(file, data_key)

    heatmaps = await asyncio.to_thread(plot_all_heatmaps, precomputed, start_phase, end_phase, obs_id)

    return {label: _serialise_heatmap(heatmaps[label]) for label in ("I", "Q", "U", "V")}

@app.post("/poincare_sphere_aitoff_fixedphase", summary="Fetch Poincare sphere data for Aitoff projection with fixed phase value")
async def poincare_sphere_aitoff_fixedphase(
    file: UploadFile | None = File(None),
    phase_value: float = 0.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    obs_id: str | None = None,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    lon_arr, lat_array = await asyncio.to_thread(
        plot_poincare_aitoff_at_phase, precomputed, on_pulse, phase_value, obs_id or _resolve_obs_id(file, data_key)
    )

    return {"lon": lon_arr.tolist(), "lat": lat_array.tolist()}


@app.post("/phase_slice_histograms", summary="Phase-slice histograms for multiple polarisation quantities")
async def phase_slice_histograms(
    file: UploadFile | None = File(None),
    left_phase: float = 0.0,
    mid_phase: float = 0.5,
    right_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    default_bins: int = 200,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    payload = await asyncio.to_thread(
        plot_phase_slice_histograms_by_phase,
        precomputed,
        left_phase,
        mid_phase,
        right_phase,
        on_pulse,
        _resolve_obs_id(file, data_key),
        default_bins,
        True,
    )

    return JSONResponse(content=payload)


@app.post("/phase_slice_histogram", summary="Single phase-slice histogram for one polarisation quantity")
async def phase_slice_histogram_endpoint(
    quantity: str,
    file: UploadFile | None = File(None),
    left_phase: float = 0.0,
    mid_phase: float = 0.5,
    right_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    default_bins: int = 200,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    payload = await asyncio.to_thread(
        phase_slice_histogram_single,
        precomputed,
        left_phase,
        mid_phase,
        right_phase,
        on_pulse,
        _resolve_obs_id(file, data_key),
        quantity,
        default_bins,
    )

    return JSONResponse(content=payload)


@app.post(
    "/polarisation_params",
    summary="Preprocess Poincare-sphere coords and polarisation fractions/angles",
)
async def polarisation_params(
    file: UploadFile | None = File(None),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    max_pulses: int | None = None,
    pulse_index: int | None = None,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    if pulse_index is not None and (pulse_index < 0 or pulse_index > precomputed.num_pulses):
        raise HTTPException(
            status_code=400,
            detail=f"pulse_index must be between 0 and {precomputed.num_pulses}; 0 is the integrated profile",
        )
    payload = await asyncio.to_thread(
        build_polarisation_payload,
        precomputed,
        start_phase,
        end_phase,
        on_pulse,
        max_pulses,
        pulse_index,
    )

    return JSONResponse(content=payload)

# One route that serves a single quantity; you can call it for each of the 8 quantities
# quantity values: PA, EA, P/I, L/I, |V/I|, V/I, I, dPA
@app.post("/polarisation_histogram", summary="Single polarisation histogram for one quantity")
async def polarisation_histogram_single_endpoint(
    quantity: str,
    file: UploadFile | None = File(None),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    base_quantity_bins: int = 200,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    payload = await asyncio.to_thread(
        polarisation_histogram_single,
        precomputed,
        start_phase,
        end_phase,
        on_pulse,
        _resolve_obs_id(file, data_key),
        quantity,
        base_quantity_bins,
    )

    return JSONResponse(content=payload)


@app.post("/polarisation_stacks", summary="Pulse-phase stacks for polarisation quantities")
async def polarisation_stacks_endpoint(
    file: UploadFile | None = File(None),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    return StreamingResponse(
        iter_polarisation_stacks_json(
            precomputed,
            start_phase,
            end_phase,
            on_pulse,
            _resolve_obs_id(file, data_key),
        ),
        media_type="application/json",
    )


@app.post("/polarisation_stack", summary="Pulse-phase stack for one polarisation quantity")
async def polarisation_stack_endpoint(
    quantity: str,
    file: UploadFile | None = File(None),
    start_phase: float = 0.0,
    end_phase: float = 1.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    data_key: str | None = None,
):
    on_pulse = (on_pulse_start, on_pulse_end)
    precomputed, _ = await load_polarimetry_precompute(file=file, on_pulse=on_pulse, data_key=data_key)
    payload = await asyncio.to_thread(
        polarisation_stack_single,
        precomputed,
        start_phase,
        end_phase,
        on_pulse,
        _resolve_obs_id(file, data_key),
        quantity,
    )

    return JSONResponse(content=payload)

