# dnsmos-pytorch

PyTorch implementation of [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

## Install

```bash
pip install git+https://github.com/rziga/dnsmos-pytorch.git@v1.0.0
```

## Usage

Audio is expected as `[batch, 1, samples]`. Input is resampled to 16 kHz internally.

```python
import torch
import torchaudio
from dnsmos_pytorch import DNSMOSp808, DNSMOSp835

waveform, sample_rate = torchaudio.load("audio.wav")
waveform = waveform[:1].unsqueeze(0)

p808 = DNSMOSp808.from_pretrained("rziga/DNSMOSp808").eval()
with torch.no_grad():
    mos = p808(waveform, sample_rate=sample_rate)  # [batch]

p835 = DNSMOSp835.from_pretrained("rziga/DNSMOSp835").eval()
with torch.no_grad():
    scores = p835(waveform, sample_rate=sample_rate)  # [batch, 3]
sig, bak, ovrl = scores[0]
```

`DNSMOSp808` predicts overall MOS. `DNSMOSp835` predicts SIG, BAK, and OVRL.

Weights: [DNSMOSp808](https://huggingface.co/rziga/DNSMOSp808), [DNSMOSp835](https://huggingface.co/rziga/DNSMOSp835), [DNSMOSp835-personalized](https://huggingface.co/rziga/DNSMOSp835-personalized).

## License

This repository is licensed under the [MIT License](LICENSE).

The original DNSMOS weights belong to Microsoft and remain subject to their license, available in the [DNS-Challenge repository](https://github.com/microsoft/DNS-Challenge).
