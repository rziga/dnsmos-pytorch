from pathlib import Path

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from dnsmos_pytorch.dnsmos import (
    DNSMOSp808,
    DNSMOSp808Config,
    DNSMOSp835,
    DNSMOSp835Config,
)

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "original_weights"
_P808_ONNX = _ASSETS / "DNSMOS" / "model_v8.onnx"
_P835_ONNX = _ASSETS / "DNSMOS" / "sig_bak_ovr.onnx"
_P835_PERSONALIZED_ONNX = _ASSETS / "pDNSMOS" / "sig_bak_ovr.onnx"
_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = int(9.01 * _SAMPLE_RATE)
_HOP_SAMPLES = 16_000

def _audio_melspec(
    audio: np.ndarray,
    n_mels: int = 120,
    frame_size: int = 320,
    hop_length: int = 160,
    sr: int = 16000,
    to_db: bool = True,
) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def _apply_polyfit(raw: np.ndarray, polyfit: torch.Tensor) -> np.ndarray:
    coeffs = polyfit.numpy()
    return (
        coeffs[:, 0] * raw**3
        + coeffs[:, 1] * raw**2
        + coeffs[:, 2] * raw
        + coeffs[:, 3]
    )


def _iter_hops(audio: np.ndarray):
    for start in range(0, audio.shape[-1] - _WINDOW_SAMPLES + 1, _HOP_SAMPLES):
        yield audio[start : start + _WINDOW_SAMPLES]


@pytest.fixture(params=[_WINDOW_SAMPLES, 22 * _SAMPLE_RATE], ids=["9.01s", "22s"])
def waveform(request: pytest.FixtureRequest) -> torch.Tensor:
    rng = np.random.default_rng(1337)
    return torch.from_numpy(rng.random((1, 1, request.param), dtype=np.float32))


def test_p808_from_original_onnx_state_dict_shapes() -> None:
    loaded = DNSMOSp808.from_original_onnx(onnx.load(_P808_ONNX))
    fresh = DNSMOSp808(DNSMOSp808Config())
    assert loaded.state_dict().keys() == fresh.state_dict().keys()
    for key, tensor in loaded.state_dict().items():
        assert tensor.shape == fresh.state_dict()[key].shape, key


def test_p808_matches_original_onnx(waveform: torch.Tensor) -> None:
    model = DNSMOSp808.from_original_onnx(onnx.load(_P808_ONNX))
    model.eval()

    audio = waveform[0, 0].numpy()
    sess = ort.InferenceSession(str(_P808_ONNX))
    scores = []
    for seg in _iter_hops(audio):
        features = np.array(_audio_melspec(audio=seg[:-160]), dtype=np.float32)[
            np.newaxis, :, :
        ]
        scores.append(sess.run(None, {"input_1": features})[0])
    onnx_out = np.mean(scores, axis=0)

    with torch.no_grad():
        pytorch_out = model(waveform)

    torch.testing.assert_close(
        pytorch_out, torch.from_numpy(onnx_out), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize(
    ("onnx_path", "personalized"),
    [
        (_P835_ONNX, False),
        (_P835_PERSONALIZED_ONNX, True),
    ],
    ids=["regular", "personalized"],
)
def test_p835_from_original_onnx_state_dict_shapes(
    onnx_path: Path, personalized: bool
) -> None:
    loaded = DNSMOSp835.from_original_onnx(
        onnx.load(onnx_path), personalized=personalized
    )
    fresh = DNSMOSp835(DNSMOSp835Config(personalized=personalized))
    assert loaded.state_dict().keys() == fresh.state_dict().keys()
    for key, tensor in loaded.state_dict().items():
        assert tensor.shape == fresh.state_dict()[key].shape, key


@pytest.mark.parametrize(
    ("onnx_path", "personalized"),
    [
        (_P835_ONNX, False),
        (_P835_PERSONALIZED_ONNX, True),
    ],
    ids=["regular", "personalized"],
)
def test_p835_matches_original_onnx(
    waveform: torch.Tensor, onnx_path: Path, personalized: bool
) -> None:
    model = DNSMOSp835.from_original_onnx(
        onnx.load(onnx_path), personalized=personalized
    )
    model.eval()

    audio = waveform[0, 0].numpy()
    sess = ort.InferenceSession(str(onnx_path))
    scores = []
    for seg in _iter_hops(audio):
        onnx_raw = sess.run(None, {"input_1": seg[np.newaxis, :]})[0]
        scores.append(_apply_polyfit(onnx_raw, model.polyfit))
    onnx_out = np.mean(scores, axis=0)

    with torch.no_grad():
        pytorch_out = model(waveform)

    torch.testing.assert_close(
        pytorch_out, torch.from_numpy(onnx_out).to(pytorch_out.dtype), atol=1e-4, rtol=1e-4
    )
