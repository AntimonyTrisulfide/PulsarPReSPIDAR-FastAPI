from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch
import torchaudio
import pathlib
import io
import shutil
import traceback
import uuid
from typing import Optional
from app.model import UNet
from app.processor import ExternalPreprocessedDataset, ExternalPreprocessor

app = FastAPI()

allowed_origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Static Files Setup ----------------------------- #
OUTPUT_DIR = pathlib.Path("reconstructed_audio")
OUTPUT_DIR.mkdir(exist_ok=True)
PUBLIC_BASE_URL = "http://127.0.0.1:8001"
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# ----------------------------- Model Setup ----------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=16).to(device)

model_path = pathlib.Path("app/model_weights.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------- Inference Logic ----------------------------- #
def reconstruct_and_save_audio(model, dataset, preprocessor, save_dir=OUTPUT_DIR, device='cpu'):
    import torchaudio.transforms as T

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    track_urls = []  # Flat list retained for backward compatibility
    track_payloads = []

    for track_dir in dataset.track_dirs:
        track_name = track_dir.name
        track_output_dir = save_dir / track_name
        track_output_dir.mkdir(exist_ok=True)
        current_track_stems = []

        mix_data = torch.load(track_dir / 'mix.pt')
        mix_specs = mix_data['spectrogram']
        mix_phases = mix_data['phases']
        n_chunks = mix_specs.shape[0]

        reconstructed_sources = {name: [] for name in dataset.all_source_names}
        istft = T.InverseSpectrogram(n_fft=preprocessor.n_fft, hop_length=preprocessor.hop_length)

        for chunk_idx in range(n_chunks):
            mix_spec_chunk = mix_specs[chunk_idx].unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                mask_logits = model(mix_spec_chunk)
                masks = torch.sigmoid(mask_logits).squeeze(0).cpu()

            for src_idx, source_name in enumerate(dataset.all_source_names):
                masked_spec = masks[src_idx] * mix_spec_chunk.squeeze().cpu()
                phase = mix_phases[chunk_idx]
                complex_spec = masked_spec * torch.exp(1j * phase)
                reconstructed_audio = istft(complex_spec.unsqueeze(0)).squeeze(0)
                reconstructed_sources[source_name].append(reconstructed_audio)

        for source_name, audio_chunks in reconstructed_sources.items():
            full_audio = torch.cat(audio_chunks, dim=0)
            save_path = track_output_dir / f"{source_name}_reconstructed.wav"
            torchaudio.save(save_path, full_audio.unsqueeze(0), preprocessor.sr)

            # Build the public URL
            url = f"{PUBLIC_BASE_URL}/output/{track_name}/{source_name}_reconstructed.wav"
            track_urls.append(url)
            current_track_stems.append({"name": source_name, "url": url})

        track_payloads.append({"cache_id": track_name, "stems": current_track_stems})

    primary_track = track_payloads[0] if track_payloads else {"cache_id": None, "stems": []}

    return {
        "status": "success",
        "cache_id": primary_track["cache_id"],
        "files": track_urls,
        "stems": primary_track["stems"],
    }

# ----------------------------- Inference Pipeline ----------------------------- #
def inference_pipeline(temp_input_path, device, track_id):
    output_dir_preprocessed = pathlib.Path("preprocessed_output")
    output_dir_preprocessed.mkdir(exist_ok=True)

    try:
        preprocessor = ExternalPreprocessor(temp_input_path, output_dir_preprocessed)
        track_output_dir = preprocessor.preprocess()

        # Ensure the downstream track directory aligns with the requested cache identifier
        if track_output_dir.name != track_id:
            desired_dir = track_output_dir.parent / track_id
            if desired_dir.exists():
                shutil.rmtree(desired_dir, ignore_errors=True)
            track_output_dir.rename(desired_dir)

        source_names = [
            'Bass', 'Brass', 'Chromatic Percussion', 'Drums', 'Ethnic',
            'Guitar', 'Organ', 'Percussive', 'Piano', 'Pipe', 'Reed',
            'Sound Effects', 'Strings', 'Strings (continued)',
            'Synth Lead', 'Synth Pad'
        ]

        dataset = ExternalPreprocessedDataset(output_dir_preprocessed, source_names)
        result = reconstruct_and_save_audio(model, dataset, preprocessor, device=device)
        result["cache_id"] = track_id
        return result

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

    finally:
        try:
            if temp_input_path.exists():
                temp_input_path.unlink()
                print(f"🧹 Deleted temporary input file: {temp_input_path}")

            if output_dir_preprocessed.exists():
                shutil.rmtree(output_dir_preprocessed, ignore_errors=True)
                print(f"🧹 Deleted preprocessed folder: {output_dir_preprocessed}")

        except Exception as cleanup_error:
            print(f"[Warning] Cleanup failed: {cleanup_error}")

# ----------------------------- FastAPI Endpoint ----------------------------- #
def build_cached_payload(track_id):
    track_dir = OUTPUT_DIR / track_id
    if not track_dir.exists():
        return None

    stems = []
    for wav_file in sorted(track_dir.glob("*.wav")):
        stem_name = wav_file.stem.replace("_reconstructed", "")
        url = f"{PUBLIC_BASE_URL}/output/{track_id}/{wav_file.name}"
        stems.append({"name": stem_name, "url": url})

    if not stems:
        return None

    return {
        "files": [stem["url"] for stem in stems],
        "stems": stems,
    }


@app.post("/infer")
async def infer_audio(file: UploadFile = File(...), cache_id: Optional[str] = Form(None)):
    print("Received file:", file.filename if file else "No file")
    if not file:
        return {"error": "No file received"}

    track_id = cache_id or uuid.uuid4().hex

    if cache_id:
        cached_payload = build_cached_payload(track_id)
        if cached_payload:
            return JSONResponse({
                "status": "cached",
                "cache_id": track_id,
                **cached_payload,
            })

    input_bytes = await file.read()
    print("File size:", len(input_bytes), "bytes")
    input_audio, sr = torchaudio.load(io.BytesIO(input_bytes))

    temp_input_path = pathlib.Path(f"{track_id}.wav")
    torchaudio.save(temp_input_path, input_audio, sr)

    result = inference_pipeline(temp_input_path, device, track_id)
    return JSONResponse(result)
