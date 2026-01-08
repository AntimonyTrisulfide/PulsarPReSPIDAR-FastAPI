from fastapi import FastAPI, HTTPException, Request
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import io
import numpy as np
from test import plot_poincare_aitoff_at_phase, return_xyz_interactive_poincare_sphere
from test import get_all_profiles
from test import plot_all_heatmaps
from test import (
    plot_phase_slice_histograms_by_phase,
    polarisation_histogram_single,
    build_polarisation_payload,
)
from test import plot_polarisation_stacks
from fastapi.responses import JSONResponse
import asyncio
from functools import lru_cache
import hashlib
import httpx

app = FastAPI(title="Pulsar Polarimetry API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to load numpy data
async def load_numpy_data(file: UploadFile):
    """Load and parse numpy file asynchronously"""
    content = await file.read()
    # Run CPU-intensive np.load in thread pool
    data = await asyncio.to_thread(np.load, io.BytesIO(content))
    if isinstance(data, np.lib.npyio.NpzFile):
        key = list(data.keys())[0]
        data = data[key]
    return data

@app.get("/", summary="Health check")
async def root() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/proxy", summary="Proxy requests to external servers")
async def proxy_request(url: str, request: Request):
    """
    Proxy endpoint that forwards requests to external URLs (e.g., MeerTime server).
    Accepts a url query parameter and forwards the authorization header.
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is required")
    
    # Extract authorization header from incoming request
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    
    headers = {"Authorization": auth_header}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            
            # Forward the response back to the client
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to remote server: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.post("/export_poincare_data", summary="Fetch pulsar details")
async def export_poincare_data(file: UploadFile = File(...), start_phase: float = 0.0, end_phase: float = 1.0, on_pulse_start: float = 0.0, on_pulse_end: float = 1.0):
    # Load numpy file npz or npy
    data = await load_numpy_data(file)

    # Run computation in thread pool to avoid blocking
    response = await asyncio.to_thread(
        return_xyz_interactive_poincare_sphere,
        data, start_phase, end_phase, (on_pulse_start, on_pulse_end), file.filename
    )
    return {"x": response[0].tolist(), "y": response[1].tolist(), "z": response[2].tolist()}

@app.post("/export_profiles", summary="Fetch profiles")
async def export_profiles(file: UploadFile = File(...), start_phase: float = 0.0, end_phase: float = 1.0):
    # Load numpy file npz or npy
    data = await load_numpy_data(file)

    # Get all profiles in one optimized call
    profiles = await asyncio.to_thread(get_all_profiles, data, start_phase, end_phase)
    
    I_profile = {"x": profiles['I']['x'].tolist(), "y": profiles['I']['y'].tolist()}
    Q_profile = {"x": profiles['Q']['x'].tolist(), "y": profiles['Q']['y'].tolist()}
    U_profile = {"x": profiles['U']['x'].tolist(), "y": profiles['U']['y'].tolist()}
    V_profile = {"x": profiles['V']['x'].tolist(), "y": profiles['V']['y'].tolist()}

    return {"I": I_profile, "Q": Q_profile, "U": U_profile, "V": V_profile}

@app.post("/export_heatmaps", summary="Fetch heatmaps")
async def export_heatmaps(file: UploadFile = File(...), start_phase: float = 0.0, end_phase: float = 1.0):
    # Load numpy file npz or npy
    data = await load_numpy_data(file)

    obs_id = file.filename

    # Compute all heatmaps in one efficient pass with async
    heatmaps = await asyncio.to_thread(plot_all_heatmaps, data, start_phase, end_phase, obs_id)
    
    # Convert to JSON-serializable format
    I_heatmap_data = {"pulse_phase": heatmaps['I']['pulse_phase'].tolist(), "pulse_number": heatmaps['I']['pulse_number'].tolist(), "heatmap_data": heatmaps['I']['heatmap_data'].tolist(), "vmin": heatmaps['I']['vmin'], "vmax": heatmaps['I']['vmax'], "label": heatmaps['I']['label'], "obs_id": heatmaps['I']['obs_id']}
    Q_heatmap_data = {"pulse_phase": heatmaps['Q']['pulse_phase'].tolist(), "pulse_number": heatmaps['Q']['pulse_number'].tolist(), "heatmap_data": heatmaps['Q']['heatmap_data'].tolist(), "vmin": heatmaps['Q']['vmin'], "vmax": heatmaps['Q']['vmax'], "label": heatmaps['Q']['label'], "obs_id": heatmaps['Q']['obs_id']}
    U_heatmap_data = {"pulse_phase": heatmaps['U']['pulse_phase'].tolist(), "pulse_number": heatmaps['U']['pulse_number'].tolist(), "heatmap_data": heatmaps['U']['heatmap_data'].tolist(), "vmin": heatmaps['U']['vmin'], "vmax": heatmaps['U']['vmax'], "label": heatmaps['U']['label'], "obs_id": heatmaps['U']['obs_id']}
    V_heatmap_data = {"pulse_phase": heatmaps['V']['pulse_phase'].tolist(), "pulse_number": heatmaps['V']['pulse_number'].tolist(), "heatmap_data": heatmaps['V']['heatmap_data'].tolist(), "vmin": heatmaps['V']['vmin'], "vmax": heatmaps['V']['vmax'], "label": heatmaps['V']['label'], "obs_id": heatmaps['V']['obs_id']}

    return {"I": I_heatmap_data, "Q": Q_heatmap_data, "U": U_heatmap_data, "V": V_heatmap_data}

@app.post("/poincare_sphere_aitoff_fixedphase", summary="Fetch Poincare sphere data for Aitoff projection with fixed phase value")
async def poincare_sphere_aitoff_fixedphase(
    file: UploadFile = File(...),
    phase_value: float = 0.0,
    on_pulse_start: float = 0.0,
    on_pulse_end: float = 1.0,
    obs_id: str | None = None,
):
    data = await load_numpy_data(file)

    on_pulse = (on_pulse_start, on_pulse_end)
    lon_arr, lat_array = await asyncio.to_thread(
        plot_poincare_aitoff_at_phase, data, on_pulse, phase_value, obs_id or "uploaded"
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
    data = await load_numpy_data(file)

    on_pulse = (on_pulse_start, on_pulse_end)
    payload = await asyncio.to_thread(
        plot_phase_slice_histograms_by_phase,
        data,
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
    data = await load_numpy_data(file)

    on_pulse = (on_pulse_start, on_pulse_end)
    payload = await asyncio.to_thread(
        build_polarisation_payload,
        data,
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
    data = await load_numpy_data(file)

    on_pulse = (on_pulse_start, on_pulse_end)
    payload = await asyncio.to_thread(
        polarisation_histogram_single,
        data,
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
    data = await load_numpy_data(file)

    on_pulse = (on_pulse_start, on_pulse_end)
    payload = await asyncio.to_thread(
        plot_polarisation_stacks,
        data,
        start_phase,
        end_phase,
        on_pulse,
        file.filename,
        True,
    )

    return JSONResponse(content=payload)


