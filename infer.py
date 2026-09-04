# -*- coding: utf-8 -*-
"""
Standalone Inference Script for Voice Deepfake Detection.
Usage:
    python infer.py path/to/audio.wav [path/to/voice_deepfake_detector.pth]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly


# ------------------------------------------------------------
# Spectrogram & Model Architecture (Matches Trained v3 Model)
# ------------------------------------------------------------

SAMPLE_RATE = 16_000
AUDIO_DURATION = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)
N_MELS = 128
HOP_LENGTH = 256
F_MIN = 20
F_MAX = SAMPLE_RATE // 2


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        left_idx = np.where((fft_freqs >= left) & (fft_freqs <= center))[0]
        if center > left:
            filterbank[m - 1, left_idx] = (fft_freqs[left_idx] - left) / (center - left)
        right_idx = np.where((fft_freqs >= center) & (fft_freqs <= right))[0]
        if right > center:
            filterbank[m - 1, right_idx] = (right - fft_freqs[right_idx]) / (right - center)

    return torch.tensor(filterbank, dtype=torch.float32)


MEL_FILTERBANK_1024 = create_mel_filterbank(SAMPLE_RATE, 1024, N_MELS, F_MIN, F_MAX)


class MultiResolutionSpectrogram(nn.Module):
    def __init__(self, hop_length=HOP_LENGTH, n_mels=N_MELS, mel_filterbank=MEL_FILTERBANK_1024):
        super().__init__()
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.register_buffer("window_512", torch.hann_window(512))
        self.register_buffer("window_1024", torch.hann_window(1024))
        self.register_buffer("window_2048", torch.hann_window(2048))
        self.register_buffer("mel_filterbank", mel_filterbank)

    def forward(self, waveform):
        with torch.no_grad():
            waveform = waveform.float()

            stft_1024 = torch.stft(
                waveform, n_fft=1024, hop_length=self.hop_length, win_length=1024,
                window=self.window_1024, center=True, return_complex=True
            )
            power_1024 = torch.abs(stft_1024) ** 2
            mel = torch.matmul(self.mel_filterbank, power_1024)
            log_mel = torch.log(torch.clamp(mel, min=1e-5))
            log_mel = (log_mel - log_mel.mean(dim=(-2, -1), keepdim=True)) / (log_mel.std(dim=(-2, -1), keepdim=True) + 1e-5)

            stft_512 = torch.stft(
                waveform, n_fft=512, hop_length=self.hop_length, win_length=512,
                window=self.window_512, center=True, return_complex=True
            )
            power_512 = torch.abs(stft_512) ** 2
            lin_512 = F.adaptive_avg_pool2d(power_512.unsqueeze(1), (self.n_mels, power_512.shape[-1])).squeeze(1)
            log_lin_512 = torch.log(torch.clamp(lin_512, min=1e-5))
            log_lin_512 = (log_lin_512 - log_lin_512.mean(dim=(-2, -1), keepdim=True)) / (log_lin_512.std(dim=(-2, -1), keepdim=True) + 1e-5)

            stft_2048 = torch.stft(
                waveform, n_fft=2048, hop_length=self.hop_length, win_length=2048,
                window=self.window_2048, center=True, return_complex=True
            )
            power_2048 = torch.abs(stft_2048) ** 2
            lin_2048 = F.adaptive_avg_pool2d(power_2048.unsqueeze(1), (self.n_mels, power_2048.shape[-1])).squeeze(1)
            log_lin_2048 = torch.log(torch.clamp(lin_2048, min=1e-5))
            log_lin_2048 = (log_lin_2048 - log_lin_2048.mean(dim=(-2, -1), keepdim=True)) / (log_lin_2048.std(dim=(-2, -1), keepdim=True) + 1e-5)

            return torch.stack([log_mel, log_lin_512, log_lin_2048], dim=1)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=(3, 3), dropout=0.0):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, tuple) else (kernel_size // 2, kernel_size // 2)
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.se(self.conv2(self.conv1(x))) + self.shortcut(x))


class MultiDomainDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=(5, 5), stride=2, padding=2),
            ResidualBlock(32, 32, stride=1, kernel_size=(3, 3)),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, kernel_size=(5, 3)),
            ResidualBlock(64, 64, stride=1, kernel_size=(3, 3)),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, kernel_size=(3, 5)),
            ResidualBlock(128, 128, stride=1, kernel_size=(5, 3)),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, kernel_size=(3, 3)),
            ResidualBlock(256, 256, stride=1, kernel_size=(3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        mean_feat = x.mean(dim=(-2, -1))
        std_feat = x.std(dim=(-2, -1))
        max_feat = F.adaptive_max_pool2d(x, (1, 1)).flatten(1)

        pooled = torch.cat([mean_feat, std_feat, max_feat], dim=1)
        return self.classifier(pooled).squeeze(-1)


# ------------------------------------------------------------
# Inference Function
# ------------------------------------------------------------

def load_audio_file(path):
    """
    Load audio from file with multi-backend fallback:
    1. soundfile (WAV, FLAC, OGG, etc.)
    2. torchaudio (M4A, AAC, MP3, etc.)
    3. ffmpeg subprocess (any audio/video format)
    """
    audio, sr = None, None

    # Backend 1: soundfile
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
    except Exception:
        pass

    # Backend 2: torchaudio
    if audio is None:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                audio = waveform.mean(dim=0).cpu().numpy()
            else:
                audio = waveform.squeeze().cpu().numpy()
        except Exception:
            pass

    # Backend 3: ffmpeg subprocess
    if audio is None:
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-nostdin", "-v", "error",
                "-i", str(path),
                "-f", "f32le",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "pipe:1"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio = np.frombuffer(proc.stdout, dtype=np.float32)
            sr = SAMPLE_RATE
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file '{path}'. Supported formats include .wav, .mp3, .m4a, .aac, .flac, .ogg. Details: {e}"
            )

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if sr != SAMPLE_RATE:
        gcd = math.gcd(int(sr), int(SAMPLE_RATE))
        up, down = SAMPLE_RATE // gcd, sr // gcd
        audio = resample_poly(audio, up, down).astype(np.float32)

    if len(audio) > 0:
        audio = audio - np.mean(audio)

    max_abs = np.max(np.abs(audio)) if len(audio) else 0.0
    if max_abs > 1e-8:
        audio = audio / max_abs

    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]

    return torch.from_numpy(audio.astype(np.float32))


def run_inference(audio_path, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if audio_path is None or str(audio_path).startswith("-") or str(audio_path).endswith(".json"):
        raise ValueError(
            f"Invalid audio path: '{audio_path}'. Please pass a valid path to an audio file (e.g. run_inference('sample.wav'))."
        )

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

    # Locate checkpoint
    if model_path is None or str(model_path).endswith(".json"):
        possible_paths = [
            Path("./trained_models/voice_deepfake_detector.pth"),
            Path("/kaggle/working/voice_deepfake_detector/voice_deepfake_detector.pth"),
            Path("./voice_deepfake_detector.pth"),
            Path("./checkpoints/best_voice_deepfake_model.pth"),
        ]
        model_path = next((p for p in possible_paths if p.exists()), None)

    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            "Could not find voice_deepfake_detector.pth. Please provide the path as second argument."
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    threshold = float(checkpoint.get("threshold", 0.0509))

    model = MultiDomainDeepfakeDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mel_transform = MultiResolutionSpectrogram().to(device)
    mel_transform.eval()

    waveform = load_audio_file(audio_path).unsqueeze(0).to(device)

    with torch.no_grad():
        features = mel_transform(waveform)
        logits = model(features)
        fake_prob = torch.sigmoid(logits).item()

    real_prob = 1.0 - fake_prob
    is_fake = fake_prob >= threshold

    print("\n" + "=" * 60)
    print("VOICE DEEPFAKE DETECTION RESULT")
    print("=" * 60)
    print(f"File         : {audio_path}")
    print(f"Model Checkpoint: {model_path}")
    print(f"Threshold    : {threshold:.4f}")
    print("-" * 60)
    print(f"Probability FAKE (AI): {fake_prob * 100:.2f}%")
    print(f"Probability REAL (Human): {real_prob * 100:.2f}%")
    print("-" * 60)

    if is_fake:
        print("🚨 VERDICT: [ AI-GENERATED / SYNTHETIC VOICE ]")
    else:
        print("✅ VERDICT: [ REAL / BONAFIDE HUMAN VOICE ]")
    print("=" * 60 + "\n")

    return {
        "file": str(audio_path),
        "prediction": "FAKE" if is_fake else "REAL",
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Filter out IPython/Jupyter kernel arguments (e.g. -f /root/.../kernel-1234.json)
    args = [
        a for a in sys.argv[1:]
        if not a.startswith("-") and not a.endswith(".json")
    ]

    if len(args) == 0:
        print("\n" + "=" * 60)
        print("Voice Deepfake Detector — Inference Mode")
        print("=" * 60)
        print("Usage in Terminal:")
        print("    python infer.py path/to/audio.wav [path/to/model.pth]\n")
        print("Usage in Jupyter / Kaggle Notebook:")
        print("    from infer import run_inference")
        print("    result = run_inference('path/to/sample.wav')")
        print("=" * 60 + "\n")
    else:
        audio_file = args[0]
        ckpt_file = args[1] if len(args) > 1 and args[1].endswith((".pth", ".pt")) else None
        run_inference(audio_file, ckpt_file)

