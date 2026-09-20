import argparse  # noqa: INP001
from pathlib import Path
from typing import Literal

import librosa
import torch
from torch import nn

from dnsmos_pytorch import DNSMOSp808, DNSMOSp835

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_AUDIO = _REPO_ROOT / "assets" / "audio.wav"
_HUB_REPOS = {
    "p808": "DNSMOSp808",
    "p835": "DNSMOSp835",
    "p835-personalized": "DNSMOSp835-personalized",
}
ModelName = Literal["p808", "p835", "p835-personalized"]


def _print(message: str) -> None:
    print(message)  # noqa: T201


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _add_white_noise(
    waveform: torch.Tensor,
    snr_db: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    signal_power = waveform.square().mean().clamp_min(1e-12)
    noise_std = (signal_power / (10.0 ** (snr_db / 10.0))).sqrt()
    noise = torch.randn(
        waveform.shape,
        dtype=waveform.dtype,
        generator=generator,
    )
    return waveform + noise.to(waveform.device) * noise_std


def _load_model(model_name: ModelName, hub_user: str) -> nn.Module:
    repo_id = f"{hub_user}/{_HUB_REPOS[model_name]}"
    model_cls: type[nn.Module] = (
        DNSMOSp808 if model_name == "p808" else DNSMOSp835
    )
    return model_cls.from_pretrained(repo_id)


def _format_scores(scores: torch.Tensor) -> str:
    if scores.shape[-1] == 3: # noqa: PLR2004
        sig, bak, ovrl = scores.reshape(-1, 3)[0].tolist()
        return f"SIG: {sig:.4f}  BAK: {bak:.4f}  OVRL: {ovrl:.4f}"
    return f"MOS: {scores.reshape(-1)[0].item():.4f}"


def main() -> None:
    """Score clean and noisy audio with a DNSMOS Hub checkpoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Load audio and run DNSMOS (p808, p835, or p835 personalized) "
            "on the clean signal and a white-noise mix."
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=_DEFAULT_AUDIO,
        help="Path to a wav file (default: assets/audio.wav).",
    )
    parser.add_argument(
        "--model",
        choices=tuple(_HUB_REPOS),
        default="p808",
        help="Which DNSMOS checkpoint to run.",
    )
    parser.add_argument(
        "--hub-user",
        default="rziga",
        help="Hugging Face Hub username or org.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=10.0,
        help="SNR of the noisy mix in dB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the added white noise.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (auto, cpu, cuda, mps).",
    )
    args = parser.parse_args()
    device = _resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)

    audio, sample_rate = librosa.load(args.audio, sr=None, mono=True)
    waveform = torch.from_numpy(audio).to(torch.float32)[None, None]
    noisy = _add_white_noise(waveform, args.snr_db, generator=generator)
    waveform = waveform.to(device)
    noisy = noisy.to(device)

    model = _load_model(args.model, args.hub_user).eval().to(device)
    with torch.no_grad():
        clean_scores = model(waveform, sample_rate=sample_rate)
        noisy_scores = model(noisy, sample_rate=sample_rate)

    repo_id = f"{args.hub_user}/{_HUB_REPOS[args.model]}"
    _print(f"Model: {repo_id}")
    _print(f"Audio: {args.audio}")
    _print(f"Device: {device}")
    _print(f"Clean:  {_format_scores(clean_scores)}")
    _print(f"Noisy ({args.snr_db:g} dB SNR): {_format_scores(noisy_scores)}")


if __name__ == "__main__":
    main()
