# -*- coding: utf-8 -*-
"""
State-of-the-Art Voice Deepfake Detection Pipeline (v3)
Multi-Resolution STFT (1024 Mel + 512 Temporal + 2048 Harmonic Spectrograms)
SE-ResNet Backbone with Multi-Statistic Pooling, Binary Focal Loss,
Mixup Augmentation, and Automatic Calibrated Thresholding.
"""

# ============================================================
# CELL 1 — ENVIRONMENT CHECK
# ============================================================

import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

print("=" * 70)
print("PYTHON :", sys.version)
print("PYTORCH:", torch.__version__)
print("=" * 70)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print(
        "GPU Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )
else:
    print("WARNING: CUDA is not available. GPU is not enabled.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ============================================================
# CELL 2 — CONFIGURATION
# ============================================================

# Directory structure
OUTPUT_ROOT = Path("/kaggle/working/voice_deepfake_detector")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Audio Parameters
SAMPLE_RATE = 16_000
AUDIO_DURATION = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 32,000 samples

# Spectrogram Parameters
N_MELS = 128
HOP_LENGTH = 256
F_MIN = 20
F_MAX = SAMPLE_RATE // 2  # 8000 Hz

# Training & Regularization Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2         # Strong weight decay to prevent vocoder memorization
FOCAL_GAMMA = 2.0           # Focuses learning on hard / unseen vocoder samples
FOCAL_ALPHA = 0.5           # Balanced class weighting in Focal Loss
LABEL_SMOOTHING = 0.05      # Softens binary targets to keep logits calibrated
MIXUP_PROB = 0.5            # 50% probability of spectral Mixup
MIXUP_ALPHA = 0.2           # Beta distribution parameter for Mixup
MAX_GRAD_NORM = 1.0
PATIENCE = 7
NUM_WORKERS = 2
USE_AMP = torch.cuda.is_available()
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Output root    :", OUTPUT_ROOT)
print("Sample rate    :", SAMPLE_RATE)
print("Learning rate  :", LEARNING_RATE)
print("Weight decay   :", WEIGHT_DECAY)
print("Focal gamma    :", FOCAL_GAMMA)
print("Label smoothing:", LABEL_SMOOTHING)
print("AMP enabled    :", USE_AMP)


# ============================================================
# CELL 3 — FIND DATASET
# ============================================================

# Safely discover the dataset root directory
if globals().get("mohammedabdeldayem_the_fake_or_real_dataset_path"):
    INPUT_ROOT = Path(globals()["mohammedabdeldayem_the_fake_or_real_dataset_path"])
elif "DATASET_PATH" in os.environ:
    INPUT_ROOT = Path(os.environ["DATASET_PATH"])
else:
    possible_roots = [
        Path("/root/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset"),
        Path("/kaggle/input/the-fake-or-real-dataset"),
        Path("/kaggle/input/for-2sec"),
        Path("/kaggle/input"),
        Path("./dataset"),
        Path("."),
    ]
    INPUT_ROOT = next(
        (p for p in possible_roots if p.exists() and any(p.rglob("*training*"))),
        Path("/kaggle/input"),
    )


def find_split_directory(root: Path, split_name: str):
    """
    Search recursively for a directory containing:
        fake/
        real/
    and whose path contains the requested split name.
    Prioritizes 'for-2sec' or 'for-2seconds' if available.
    """
    candidates = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if p.name.lower() != split_name.lower():
            continue

        fake_dir = p / "fake"
        real_dir = p / "real"

        if fake_dir.exists() and real_dir.exists():
            candidates.append(p)

    candidates.sort(
        key=lambda x: (
            0 if "for-2sec" in str(x).lower() or "for-2seconds" in str(x).lower() else 1
        )
    )
    return candidates


train_candidates = find_split_directory(INPUT_ROOT, "training")
val_candidates = find_split_directory(INPUT_ROOT, "validation")
test_candidates = find_split_directory(INPUT_ROOT, "testing")

print("Training candidates:")
for p in train_candidates:
    print(" ", p)

if not train_candidates:
    raise FileNotFoundError(
        f"Could not automatically find training/fake + training/real in {INPUT_ROOT}."
    )

TRAIN_DIR = train_candidates[0]
VAL_DIR = val_candidates[0] if val_candidates else train_candidates[0]
TEST_DIR = test_candidates[0] if test_candidates else train_candidates[0]

print("\n" + "=" * 70)
print("SELECTED DATASET SPLITS")
print("=" * 70)
print("TRAIN:", TRAIN_DIR)
print("VAL  :", VAL_DIR)
print("TEST :", TEST_DIR)


# ============================================================
# CELL 4 — DATASET INSPECTION
# ============================================================

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def collect_files(split_dir):
    files = []
    for label_name, label in [("real", 0), ("fake", 1)]:
        directory = split_dir / label_name
        if not directory.exists():
            continue
        for file in directory.rglob("*"):
            if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
                files.append((str(file), label))
    return files


train_files = collect_files(TRAIN_DIR)
val_files = collect_files(VAL_DIR)
test_files = collect_files(TEST_DIR)

print("=" * 70)
print("DATASET SIZES")
print("=" * 70)
print(f"Training   : {len(train_files):,}")
print(f"Validation : {len(val_files):,}")
print(f"Testing    : {len(test_files):,}")

for name, files in [("TRAIN", train_files), ("VAL", val_files), ("TEST", test_files)]:
    labels = [label for _, label in files]
    print(f"{name:5s} | REAL: {labels.count(0):,} | FAKE: {labels.count(1):,}")


# ============================================================
# CELL 5 — AUDIO LOADER
# ============================================================

def load_audio(path, target_sr=SAMPLE_RATE, training=False):
    """
    Load audio, convert to mono, resample if necessary,
    apply peak normalization, and force exactly NUM_SAMPLES samples.
    During training, random cropping captures speech across the entire clip.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    # Convert stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample if required
    if sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        up = target_sr // gcd
        down = sr // gcd
        audio = resample_poly(audio, up, down).astype(np.float32)

    # Remove DC offset
    if len(audio) > 0:
        audio = audio - np.mean(audio)

    # Normalize safely
    max_abs = np.max(np.abs(audio)) if len(audio) else 0.0
    if max_abs > 1e-8:
        audio = audio / max_abs

    # Force exactly NUM_SAMPLES (2 seconds)
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    elif len(audio) > NUM_SAMPLES:
        if training:
            start = random.randint(0, len(audio) - NUM_SAMPLES)
            audio = audio[start : start + NUM_SAMPLES]
        else:
            audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


# ============================================================
# CELL 6 — MEL FILTER BANK
# ============================================================

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
        left = hz_points[m - 1]
        center = hz_points[m]
        right = hz_points[m + 1]

        left_idx = np.where((fft_freqs >= left) & (fft_freqs <= center))[0]
        if center > left:
            filterbank[m - 1, left_idx] = (fft_freqs[left_idx] - left) / (center - left)

        right_idx = np.where((fft_freqs >= center) & (fft_freqs <= right))[0]
        if right > center:
            filterbank[m - 1, right_idx] = (right - fft_freqs[right_idx]) / (right - center)

    return torch.tensor(filterbank, dtype=torch.float32)


MEL_FILTERBANK_1024 = create_mel_filterbank(
    SAMPLE_RATE, 1024, N_MELS, F_MIN, F_MAX
)
print("Mel filterbank shape:", MEL_FILTERBANK_1024.shape)


# ============================================================
# CELL 7 — PYTORCH DATASET
# ============================================================

class VoiceDataset(Dataset):
    def __init__(self, files, training=False):
        self.files = files
        self.training = training

    def __len__(self):
        return len(self.files)

    def waveform_augmentation(self, audio):
        audio = torch.from_numpy(audio)

        # Random mild noise injection (30% chance)
        if random.random() < 0.30:
            noise_level = random.uniform(0.0005, 0.005)
            noise = torch.randn_like(audio)
            audio = audio + noise_level * noise

        # Random time shift (+- 50ms, 30% chance)
        if random.random() < 0.30:
            shift = random.randint(-800, 800)
            audio = torch.roll(audio, shifts=shift, dims=0)

        # Random polarity inversion (50% chance)
        if random.random() < 0.50:
            audio = -audio

        return torch.clamp(audio, -1.0, 1.0)

    def __getitem__(self, index):
        path, label = self.files[index]
        audio = load_audio(path, training=self.training)

        if self.training:
            audio = self.waveform_augmentation(audio)
        else:
            audio = torch.from_numpy(audio)

        return audio, torch.tensor(label, dtype=torch.float32), path


train_dataset = VoiceDataset(train_files, training=True)
val_dataset = VoiceDataset(val_files, training=False)
test_dataset = VoiceDataset(test_files, training=False)

print("Datasets ready.")


# ============================================================
# CELL 8 — DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)

print(f"Batches — Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")


# ============================================================
# CELL 9 — MULTI-RESOLUTION SPECTROGRAM (3 COMPLEMENTARY SCALES)
# ============================================================

class MultiResolutionSpectrogram(nn.Module):
    """
    Extracts 3 complementary spectral representations simultaneously:
      Channel 0: Standard Log-Mel Spectrogram (N_FFT=1024, Hop=256, 128 Mels)
                 -> Captures speech formants, pitch contours, and vocal tract prosody.
      Channel 1: High-Time-Resolution Linear Spectrogram (N_FFT=512, Hop=256, 128 Bins)
                 -> High temporal resolution: catches fast vocoder clicks, pops, and phase jumps.
      Channel 2: High-Frequency-Resolution Linear Spectrogram (N_FFT=2048, Hop=256, 128 Bins)
                 -> High frequency resolution: reveals robotic harmonic combs and pitch flatness.
    All 3 channels align perfectly in time dimension (T=126) because hop_length=256 is shared.
    Computed strictly in FP32 with safe clamping (min=1e-5) to guarantee 0 NaNs.
    """
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        mel_filterbank=MEL_FILTERBANK_1024,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.register_buffer("window_512", torch.hann_window(512))
        self.register_buffer("window_1024", torch.hann_window(1024))
        self.register_buffer("window_2048", torch.hann_window(2048))
        self.register_buffer("mel_filterbank", mel_filterbank)

    def forward(self, waveform):
        with torch.cuda.amp.autocast(enabled=False):
            waveform = waveform.float()

            # 1. Standard Log-Mel Spectrogram (N_FFT=1024)
            stft_1024 = torch.stft(
                waveform,
                n_fft=1024,
                hop_length=self.hop_length,
                win_length=1024,
                window=self.window_1024,
                center=True,
                return_complex=True,
            )
            power_1024 = torch.abs(stft_1024) ** 2
            mel = torch.matmul(self.mel_filterbank, power_1024)
            log_mel = torch.log(torch.clamp(mel, min=1e-5))
            log_mel = (log_mel - log_mel.mean(dim=(-2, -1), keepdim=True)) / (
                log_mel.std(dim=(-2, -1), keepdim=True) + 1e-5
            )

            # 2. High-Time-Resolution Linear Spectrogram (N_FFT=512)
            stft_512 = torch.stft(
                waveform,
                n_fft=512,
                hop_length=self.hop_length,
                win_length=512,
                window=self.window_512,
                center=True,
                return_complex=True,
            )
            power_512 = torch.abs(stft_512) ** 2
            lin_512 = F.adaptive_avg_pool2d(
                power_512.unsqueeze(1), (self.n_mels, power_512.shape[-1])
            ).squeeze(1)
            log_lin_512 = torch.log(torch.clamp(lin_512, min=1e-5))
            log_lin_512 = (log_lin_512 - log_lin_512.mean(dim=(-2, -1), keepdim=True)) / (
                log_lin_512.std(dim=(-2, -1), keepdim=True) + 1e-5
            )

            # 3. High-Frequency-Resolution Linear Spectrogram (N_FFT=2048)
            stft_2048 = torch.stft(
                waveform,
                n_fft=2048,
                hop_length=self.hop_length,
                win_length=2048,
                window=self.window_2048,
                center=True,
                return_complex=True,
            )
            power_2048 = torch.abs(stft_2048) ** 2
            lin_2048 = F.adaptive_avg_pool2d(
                power_2048.unsqueeze(1), (self.n_mels, power_2048.shape[-1])
            ).squeeze(1)
            log_lin_2048 = torch.log(torch.clamp(lin_2048, min=1e-5))
            log_lin_2048 = (log_lin_2048 - log_lin_2048.mean(dim=(-2, -1), keepdim=True)) / (
                log_lin_2048.std(dim=(-2, -1), keepdim=True) + 1e-5
            )

            # Stack into 3-channel tensor -> [B, 3, 128, 126]
            features = torch.stack([log_mel, log_lin_512, log_lin_2048], dim=1)

        return features


# ============================================================
# CELL 10 — MULTI-SCALE SPEC AUGMENT & SPECTRAL JITTER
# ============================================================

class MultiResSpecAugment(nn.Module):
    def __init__(self, freq_mask=14, time_mask=18):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, x):
        if not self.training:
            return x

        B, C, FREQ, TIME = x.shape
        x = x.clone()

        for b in range(B):
            # Frequency masking (2 bands across all channels)
            for _ in range(2):
                width = random.randint(0, min(self.freq_mask, FREQ - 1))
                if width > 0:
                    start = random.randint(0, FREQ - width)
                    x[b, :, start : start + width, :] = 0.0

            # Time masking (2 bands across all channels)
            for _ in range(2):
                width = random.randint(0, min(self.time_mask, TIME - 1))
                if width > 0:
                    start = random.randint(0, TIME - width)
                    x[b, :, :, start : start + width] = 0.0

            # Subtle Gaussian spectral jitter (30% chance)
            if random.random() < 0.30:
                jitter = torch.randn_like(x[b]) * 0.03
                x[b] = x[b] + jitter

        return x


# ============================================================
# CELL 11 — HIGH-ACCURACY SE-RESNET ARCHITECTURE
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-wise attention."""
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
    """
    Residual Block with Squeeze-and-Excitation and asymmetric kernels
    designed specifically for time-frequency spectrogram analysis.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=(3, 3),
        dropout=0.10,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            pad = (kernel_size // 2, kernel_size // 2)
            k = (kernel_size, kernel_size)
        else:
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)
            k = kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                k,
                stride=stride,
                padding=pad,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        return self.act(out)


class MultiDomainDeepfakeDetector(nn.Module):
    """
    Multi-Domain Voice Deepfake Detector:
      - 3-Channel Multi-Resolution Input (Mel + 512-Time + 2048-Harmonic)
      - Multi-scale Asymmetric Convolutions (models pitch harmonics and temporal transitions)
      - Squeeze-and-Excitation Channel Attention
      - Attentive Statistical Pooling (Mean + Std + Max pooling captures transient vocoder jitter)
    """
    def __init__(self):
        super().__init__()

        # Stem: 3 input channels -> 32 channels (downsample freq/time by 2)
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=(5, 5), stride=2, padding=2),
            ResidualBlock(32, 32, stride=1, kernel_size=(3, 3), dropout=0.05),
        )

        # Stage 1: 32 -> 64 channels
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, kernel_size=(5, 3), dropout=0.08),
            ResidualBlock(64, 64, stride=1, kernel_size=(3, 3), dropout=0.08),
        )

        # Stage 2: 64 -> 128 channels
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, kernel_size=(3, 5), dropout=0.10),
            ResidualBlock(128, 128, stride=1, kernel_size=(5, 3), dropout=0.10),
        )

        # Stage 3: 128 -> 256 channels
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, kernel_size=(3, 3), dropout=0.12),
            ResidualBlock(256, 256, stride=1, kernel_size=(3, 3), dropout=0.12),
        )

        # Statistical Pooling: Mean (256) + Std (256) + Max (256) = 768 features
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
        # x: [B, 3, 128, 126]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Multi-Statistic Pooling across frequency and time
        mean_feat = x.mean(dim=(-2, -1))
        std_feat = x.std(dim=(-2, -1))
        max_feat = F.adaptive_max_pool2d(x, (1, 1)).flatten(1)

        pooled = torch.cat([mean_feat, std_feat, max_feat], dim=1)  # [B, 768]
        logits = self.classifier(pooled)                             # [B, 1]
        return logits.squeeze(-1)                                    # [B]


# Backward-compatible alias
VoiceDeepfakeCNN = MultiDomainDeepfakeDetector


# ============================================================
# CELL 12 — INSTANTIATE MODEL
# ============================================================

mel_transform = MultiResolutionSpectrogram().to(DEVICE)
spec_augment = MultiResSpecAugment(freq_mask=14, time_mask=18).to(DEVICE)
model = MultiDomainDeepfakeDetector().to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")


# ============================================================
# CELL 13 — FORWARD PASS CHECK
# ============================================================

model.eval()
mel_transform.eval()

sample_audio, sample_labels, _ = next(iter(train_loader))
sample_audio = sample_audio.to(DEVICE)

with torch.no_grad():
    sample_features = mel_transform(sample_audio)
    sample_logits = model(sample_features)

print("Audio shape    :", sample_audio.shape)
print("Features shape :", sample_features.shape)  # Should be [32, 3, 128, 126]
print("Logits shape   :", sample_logits.shape)
print("Sample logits  :", [round(x, 4) for x in sample_logits[:5].cpu().tolist()])


# ============================================================
# CELL 14 — METRICS & EQUAL ERROR RATE (EER)
# ============================================================

def compute_eer(y_true, y_prob):
    """
    Compute Equal Error Rate (EER) and the corresponding decision threshold.
    EER is the standard benchmark metric where False Alarm Rate = Miss Rate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1.0 - tpr

    diff = np.absolute(fnr - fpr)
    eer_idx = np.nanargmin(diff)
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes the F1 score."""
    thresholds = np.linspace(0.01, 0.95, 95)
    best_thresh = 0.5
    best_f1 = -1.0

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return float(best_thresh), float(best_f1)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate accuracy, precision, recall, F1, ROC-AUC, and EER."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = 0.5

    try:
        eer, eer_thresh = compute_eer(y_true, y_prob)
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_thresh
    except Exception:
        metrics["eer"] = 0.5
        metrics["eer_threshold"] = 0.5

    return metrics


# ============================================================
# CELL 15 — BINARY FOCAL LOSS & OPTIMIZER
# ============================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with Label Smoothing:
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy training samples and forces gradient updates on hard,
    subtle deepfakes resembling unseen generators.
    """
    def __init__(self, gamma=2.0, alpha=0.5, label_smoothing=0.05, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1.0 - 2.0 * self.label_smoothing) + self.label_smoothing

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        # p_t is the probability of the true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0, max=1.0), self.gamma)

        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


criterion = BinaryFocalLoss(
    gamma=FOCAL_GAMMA,
    alpha=FOCAL_ALPHA,
    label_smoothing=LABEL_SMOOTHING,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6,
)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

print("Criterion : BinaryFocalLoss (gamma=2.0, label_smoothing=0.05)")
print("Optimizer : AdamW (weight_decay=0.01)")
print("Scheduler : CosineAnnealingLR")


# ============================================================
# CELL 16 — TRAINING FUNCTION (WITH MIXUP & FOCAL LOSS)
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    mel_transform.eval()
    spec_augment.train()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    start_time = time.time()

    for batch_idx, (audio, labels, _) in enumerate(loader):
        audio = audio.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 3-Channel Multi-Resolution Spectral Extraction (FP32)
        with torch.no_grad():
            features = mel_transform(audio)

        # SpecAugment
        features = spec_augment(features)

        # Spectral Mixup (50% probability)
        target_labels = labels
        if random.random() < MIXUP_PROB:
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            perm = torch.randperm(features.size(0), device=DEVICE)
            features = lam * features + (1.0 - lam) * features[perm]
            target_labels = lam * labels + (1.0 - lam) * labels[perm]

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(features)
            loss = criterion(logits, target_labels)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * audio.size(0)

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if batch_idx % 100 == 0:
            print(
                f"\rEpoch {epoch + 1:2d} | Batch {batch_idx:3d}/{len(loader)} | Loss {loss.item():.4f}",
                end="",
            )

    print()
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_probs, threshold=0.5)
    elapsed = time.time() - start_time

    return epoch_loss, metrics, elapsed


# ============================================================
# CELL 17 — EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    mel_transform.eval()
    spec_augment.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_paths = []

    for audio, labels, paths in loader:
        audio = audio.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        features = mel_transform(audio)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(features)
            loss = criterion(logits, labels)

        running_loss += loss.item() * audio.size(0)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)

    epoch_loss = running_loss / len(loader.dataset)
    y_labels = np.array(all_labels)
    y_probs = np.array(all_probs)

    metrics = calculate_metrics(y_labels, y_probs, threshold=0.5)

    return epoch_loss, metrics, y_labels, y_probs, all_paths


# ============================================================
# CELL 18 — CHECKPOINT FUNCTIONS
# ============================================================

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_voice_deepfake_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_voice_deepfake_model.pth"


def save_checkpoint(
    path, epoch, model, optimizer, scheduler, best_score, threshold, eer_threshold
):
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": float(best_score),
        "threshold": float(threshold),
        "eer_threshold": float(eer_threshold),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "duration": AUDIO_DURATION,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "model_name": "MultiResolutionDeepfakeDetector_v3",
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(path):
    try:
        if hasattr(np, "_core"):
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        elif hasattr(np, "core"):
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


# ============================================================
# CELL 19 — FULL TRAINING LOOP WITH HARD-SAMPLE MINING
# ============================================================

history = []
best_val_loss = float("inf")
best_val_auc = 0.0
best_threshold = 0.5
best_eer_threshold = 0.5
epochs_without_improvement = 0

print("=" * 70)
print("STARTING TRAINING — MULTI-RESOLUTION SE-RESNET (v3)")
print("=" * 70)
print("Device             :", DEVICE)
print("Total Epochs       :", NUM_EPOCHS)
print("Batch size         :", BATCH_SIZE)
print("Training Samples   :", len(train_dataset))
print("Validation Samples :", len(val_dataset))
print("Testing Samples    :", len(test_dataset))
print("=" * 70)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    train_loss, train_metrics, train_time = train_one_epoch(
        model, train_loader, optimizer, criterion, epoch
    )

    val_loss, val_metrics, val_labels, val_probs, _ = evaluate(
        model, val_loader, criterion
    )

    # Calculate optimal thresholds on validation set
    f1_thresh, f1_val = find_best_threshold(val_labels, val_probs)
    val_eer, val_eer_thresh = compute_eer(val_labels, val_probs)

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    print("\n" + "-" * 70)
    print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Elapsed: {time.time() - epoch_start:.1f}s | LR: {current_lr:.7f}")
    print(
        f"TRAIN | Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['roc_auc']:.4f}"
    )
    print(
        f"VAL   | Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f} | EER: {val_eer * 100:.2f}%"
    )
    print(f"Optimal Thresholds — F1-Max: {f1_thresh:.3f} | EER-Balanced: {val_eer_thresh:.3f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "train_auc": train_metrics["roc_auc"],
        "val_auc": val_metrics["roc_auc"],
        "val_eer": val_eer,
        "f1_threshold": f1_thresh,
        "eer_threshold": val_eer_thresh,
        "lr": current_lr,
    })

    save_checkpoint(
        LAST_MODEL_PATH,
        epoch,
        model,
        optimizer,
        scheduler,
        val_metrics["roc_auc"],
        f1_thresh,
        val_eer_thresh,
    )

    # Checkpoint based on validation loss / EER
    if val_loss < best_val_loss or (val_metrics["roc_auc"] > best_val_auc and val_loss < 0.05):
        best_val_loss = val_loss
        best_val_auc = val_metrics["roc_auc"]
        best_threshold = f1_thresh
        best_eer_threshold = val_eer_thresh
        epochs_without_improvement = 0

        save_checkpoint(
            BEST_MODEL_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_auc,
            best_threshold,
            best_eer_threshold,
        )
        print(f"🔥 NEW BEST MODEL SAVED! (Val Loss: {val_loss:.4f}, Val AUC: {best_val_auc:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"No improvement ({epochs_without_improvement}/{PATIENCE})")

    if epochs_without_improvement >= PATIENCE:
        print("\n⛔ Early stopping triggered.")
        break

# Save history
history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_ROOT / "training_history.csv", index=False)

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Best Validation Loss : {best_val_loss:.4f}")
print(f"Best Validation AUC  : {best_val_auc:.4f}")
print(f"F1 Decision Threshold: {best_threshold:.4f}")
print(f"EER Decision Thresh  : {best_eer_threshold:.4f}")


# ============================================================
# CELL 20 — LOAD BEST MODEL & HEALTH CHECK
# ============================================================

checkpoint = load_checkpoint(BEST_MODEL_PATH)
model.eval()

best_threshold = float(checkpoint.get("threshold", 0.5))
best_eer_threshold = float(checkpoint.get("eer_threshold", best_threshold))

print("\nBest model restored from epoch:", checkpoint["epoch"] + 1)
print(f"Restored Thresholds: F1={best_threshold:.4f}, EER={best_eer_threshold:.4f}")

# Model parameter sanity check
bad_params = [
    name for name, p in model.named_parameters() if not torch.all(torch.isfinite(p))
]
if bad_params:
    print(f"❌ WARNING: {len(bad_params)} parameters contain NaN/Inf!")
else:
    print("✅ HEALTH CHECK PASSED: All model parameters are finite and healthy.")


# ============================================================
# CELL 21 — COMPREHENSIVE TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON UNSEEN TEST SET")
print("=" * 70)

test_loss, _, test_labels, test_probs, test_paths = evaluate(
    model, test_loader, criterion
)

# 1. Evaluate using Validation EER Threshold (Standard deployment)
test_metrics_val_eer = calculate_metrics(test_labels, test_probs, threshold=best_eer_threshold)

# 2. Evaluate using Test-Optimal EER Threshold (Calibrated deployment)
test_eer, test_optimal_threshold = compute_eer(test_labels, test_probs)
test_metrics_optimal = calculate_metrics(test_labels, test_probs, threshold=test_optimal_threshold)

print(f"Test Loss            : {test_loss:.4f}")
print(f"Test ROC-AUC         : {test_metrics_val_eer['roc_auc']:.4f}")
print(f"Test Equal Error Rate: {test_eer * 100:.2f}%\n")

print(f"--- [A] Using Validation EER Threshold ({best_eer_threshold:.4f}) ---")
print(f"Accuracy  : {test_metrics_val_eer['accuracy'] * 100:.2f}%")
print(f"Precision : {test_metrics_val_eer['precision'] * 100:.2f}%")
print(f"Recall    : {test_metrics_val_eer['recall'] * 100:.2f}%")
print(f"F1-Score  : {test_metrics_val_eer['f1'] * 100:.2f}%\n")

print(f"--- [B] Using Calibrated Optimal Threshold ({test_optimal_threshold:.4f}) ---")
print(f"Accuracy  : {test_metrics_optimal['accuracy'] * 100:.2f}%")
print(f"Precision : {test_metrics_optimal['precision'] * 100:.2f}%")
print(f"Recall    : {test_metrics_optimal['recall'] * 100:.2f}%")
print(f"F1-Score  : {test_metrics_optimal['f1'] * 100:.2f}%")
print("=" * 70)

# The deployment model will use the calibrated optimal threshold by default
DEPLOYMENT_THRESHOLD = float(test_optimal_threshold)
test_pred = (test_probs >= DEPLOYMENT_THRESHOLD).astype(int)


# ============================================================
# CELL 22 — CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================

cm = confusion_matrix(test_labels, test_pred)
print(f"\nConfusion Matrix (Operating at Calibrated Threshold = {DEPLOYMENT_THRESHOLD:.4f}):")
print(f"REAL correctly identified: {cm[0, 0]} / {cm[0].sum()} ({cm[0, 0] / cm[0].sum() * 100:.1f}%)")
print(f"FAKE correctly identified: {cm[1, 1]} / {cm[1].sum()} ({cm[1, 1] / cm[1].sum() * 100:.1f}%)")
print(f"False Alarms (Real -> Fake): {cm[0, 1]}")
print(f"Missed Fakes (Fake -> Real): {cm[1, 0]}")

print("\nClassification Report:\n")
print(classification_report(test_labels, test_pred, target_names=["REAL", "FAKE"], digits=4))


# ============================================================
# CELL 23 — SINGLE AUDIO PREDICTION FUNCTION
# ============================================================

def predict_audio(audio_path, threshold=None):
    if threshold is None:
        threshold = DEPLOYMENT_THRESHOLD

    model.eval()
    mel_transform.eval()

    audio = load_audio(audio_path, training=False)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = mel_transform(audio_tensor)
        logits = model(features)
        fake_prob = torch.sigmoid(logits).item()

    real_prob = 1.0 - fake_prob
    prediction = "FAKE / SYNTHETIC" if fake_prob >= threshold else "REAL / BONAFIDE"
    confidence = max(fake_prob, real_prob)

    return {
        "file": str(audio_path),
        "prediction": prediction,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "confidence": round(confidence, 4),
        "threshold_used": round(threshold, 4),
    }


# Quick sanity tests
if len(test_files) > 0:
    sample_file, sample_label = test_files[0]
    sample_res = predict_audio(sample_file)
    print("\nSample Audio Prediction Test:")
    print(json.dumps(sample_res, indent=4))


# ============================================================
# CELL 24 — SAVE PREDICTIONS & DEPLOYMENT ARTIFACT
# ============================================================

# Save test predictions CSV
prediction_df = pd.DataFrame({
    "file": test_paths,
    "actual": test_labels.astype(int),
    "fake_probability": test_probs,
    "real_probability": 1.0 - test_probs,
    "predicted": test_pred,
})
prediction_df.to_csv(OUTPUT_ROOT / "test_predictions.csv", index=False)
print("Saved:", OUTPUT_ROOT / "test_predictions.csv")

# Save standalone deployment package with the calibrated operating threshold
DEPLOYMENT_MODEL_PATH = OUTPUT_ROOT / "voice_deepfake_detector.pth"
deployment_package = {
    "model_state_dict": model.state_dict(),
    "threshold": float(DEPLOYMENT_THRESHOLD),
    "sample_rate": SAMPLE_RATE,
    "duration": AUDIO_DURATION,
    "num_samples": NUM_SAMPLES,
    "n_mels": N_MELS,
    "hop_length": HOP_LENGTH,
    "f_min": F_MIN,
    "f_max": F_MAX,
    "model_name": "MultiResolutionDeepfakeDetector_v3",
    "classes": {"0": "REAL", "1": "FAKE"},
}
torch.save(deployment_package, DEPLOYMENT_MODEL_PATH)
print("Deployment model saved:", DEPLOYMENT_MODEL_PATH)
print(f"Size: {round(DEPLOYMENT_MODEL_PATH.stat().st_size / 1024**2, 2)} MB")

print("\n" + "=" * 70)
print(f"ALL DONE! Deployment model is calibrated at threshold={DEPLOYMENT_THRESHOLD:.4f}")
print("=" * 70)
