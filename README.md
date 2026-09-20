# dnsmos-pytorch

PyTorch implementation of [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

## Install

Requires Python 3.12 or later.

```bash
pip install git+https://github.com/rziga/dnsmos-pytorch.git@v1.0.0
```

## Usage

Audio is expected as `[batch, 1, samples]`. Input is resampled to 16 kHz internally. Scores are averaged over overlapping ~9 s windows.

```python
import torch
import torchaudio
from dnsmos_pytorch import DNSMOSp808, DNSMOSp835

waveform, sample_rate = torchaudio.load("audio.wav")
waveform = waveform[:1].unsqueeze(0)

p808 = DNSMOSp808.from_pretrained("rziga/DNSMOSp808").eval()
with torch.no_grad():
    mos = p808(waveform, sample_rate=sample_rate)  # [batch, 1]

p835 = DNSMOSp835.from_pretrained("rziga/DNSMOSp835").eval()
# or DNSMOSp835.from_pretrained("rziga/DNSMOSp835-personalized")
with torch.no_grad():
    scores = p835(waveform, sample_rate=sample_rate)  # [batch, 3]
sig, bak, ovrl = scores[0]
```

`DNSMOSp808` predicts overall MOS. `DNSMOSp835` predicts SIG, BAK, and OVRL.

Weights: [DNSMOSp808](https://huggingface.co/rziga/DNSMOSp808), [DNSMOSp835](https://huggingface.co/rziga/DNSMOSp835), [DNSMOSp835-personalized](https://huggingface.co/rziga/DNSMOSp835-personalized).

## Dev

Install the project and development extras with:

```bash
uv sync --dev
```

### Tests

`tests/` contains pytest coverage for checkpoint save/load (`test_save_load.py`) and numerical parity against the original ONNX DNSMOS models (`test_onnx_parity.py`). The ONNX parity tests expect the original weights under `assets/original_weights/`.

Run tests with:

```bash
bash scripts/download_original_weights.sh
uv run pytest
```

### Scripts

`scripts/` has helpers used while developing and publishing the models:

- `download_original_weights.sh` fetches Microsoft's ONNX DNSMOS weights into `assets/original_weights/`.
- `run_dnsmos_demo.py` scores a wav file and a white-noise mix of it.
- `convert_original_weights_and_upload_to_hub.py` converts those ONNX weights to this package's format and uploads them to the Hugging Face Hub.

```bash
uv run python scripts/run_dnsmos_demo.py --audio path/to/audio.wav
```

## License

This repository is licensed under the [MIT License](LICENSE).

The original DNSMOS weights belong to Microsoft and remain subject to their license, available in the [DNS-Challenge repository](https://github.com/microsoft/DNS-Challenge).
