import torch
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import math

class ExternalPreprocessor:
    def __init__(self, input_file, output_dir, chunk_duration=4.0,
                 sr=16000, n_fft=2048, hop_length=512):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.sr = sr
        self.chunk_samples = int(chunk_duration * sr)
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Spectrogram (complex)
        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None  # keep complex-valued STFT output
        )

    def preprocess(self):
        """Preprocess the external track"""
        # Load mix: shape -> (channels, samples)
        mix, sample_rate = torchaudio.load(self.input_file)

        if sample_rate != self.sr:
            # Resample if necessary
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sr)
            mix = resampler(mix)

        # Convert mix to spectrograms and phases
        mix_specs, mix_phases = self._audio_to_specs(mix)

        # Create output directory for the track
        track_output_dir = self.output_dir / self.input_file.stem
        track_output_dir.mkdir(exist_ok=True)

        # Save mix
        torch.save({
            'spectrogram': mix_specs,   # (n_chunks, freq, time)
            'phases': mix_phases,      # (n_chunks, freq, time)
            'track_name': self.input_file.stem,
            'type': 'mix',
            'sr': self.sr,
            'chunk_samples': self.chunk_samples
        }, track_output_dir / 'mix.pt')

        print(f"Preprocessing complete for {self.input_file.name}!")
        return track_output_dir

    def _audio_to_specs(self, audio):
        """
        Convert audio (channels, samples) -> chunked spectrogram magnitudes and phases.
        Now: Pads the LAST chunk instead of trimming it.
        """
        # Convert to mono
        if audio.ndim == 2 and audio.shape[0] > 1:
            waveform = torch.mean(audio, dim=0)
        else:
            waveform = audio.squeeze(0)

        total_samples = waveform.shape[0]

        # Compute number of chunks (ceil)
        n_chunks = math.ceil(total_samples / self.chunk_samples)

        # Pad waveform so it fits exactly n_chunks * chunk_samples
        required_samples = n_chunks * self.chunk_samples
        pad_amount = required_samples - total_samples

        if pad_amount > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Now reshape into (n_chunks, chunk_samples)
        waveform = waveform[:required_samples]
        audio_chunks = waveform.view(n_chunks, self.chunk_samples)

        specs = []
        phases = []

        for i in range(n_chunks):
            chunk = audio_chunks[i].unsqueeze(0)  # (1, chunk_samples)
            complex_spec = self.stft(chunk)[0]    # (freq, time), complex

            magnitude = torch.abs(complex_spec)
            phase = torch.angle(complex_spec)

            specs.append(magnitude)
            phases.append(phase)

        return torch.stack(specs), torch.stack(phases)


# new dataset class for external data (unchanged except small safety tweak)
class ExternalPreprocessedDataset(Dataset):
    def __init__(self, preprocessed_dir, source_names):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.track_dirs = sorted([d for d in self.preprocessed_dir.iterdir() if d.is_dir()])
        self.all_source_names = source_names

        # Precompute chunk counts
        self.track_chunk_counts = []
        for track_dir in self.track_dirs:
            mix_data = torch.load(track_dir / 'mix.pt')
            n_chunks = int(mix_data['spectrogram'].shape[0])
            self.track_chunk_counts.append(n_chunks)

        self.total_chunks = sum(self.track_chunk_counts)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        for track_dir, n_chunks in zip(self.track_dirs, self.track_chunk_counts):
            if idx < n_chunks:
                mix_data = torch.load(track_dir / 'mix.pt')
                mix_spec = mix_data['spectrogram'][idx]
                return mix_spec
            idx -= n_chunks
        raise IndexError("Index out of range")
