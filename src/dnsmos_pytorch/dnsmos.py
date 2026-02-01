import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import resample

if TYPE_CHECKING:
    from onnx import ModelProto


def _onnx_initializer_tensors(
    onnx_model: "ModelProto",
) -> dict[str, torch.Tensor]:
    from onnx import numpy_helper  # noqa: PLC0415

    return {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in onnx_model.graph.initializer
    }


def _matmul_to_conv1x1(weight: torch.Tensor) -> torch.Tensor:
    return weight.T.contiguous()[..., None, None]


def _get_to_samples(sample_rate: int) -> Callable[[float], int]:
    def to_samples(seconds: float) -> int:
        return int(seconds * sample_rate)

    return to_samples


@dataclass(frozen=True, kw_only=True)
class DNSMOSp808Config:
    """Configuration for DNSMOSp808 model.

    Args:
        sample_rate: int: sample rate expected by the model (don't change)
        window_size_seconds: float: window size in seconds
        hop_size_seconds: float: hop size in seconds
            (change to 9 for faster inference if you don't care about the spec)
        fft_window_size_seconds: float: FFT window size in seconds
        fft_hop_size_seconds: float: FFT hop size in seconds
        num_mel: int: number of mel bins

    """

    sample_rate: int = 16_000
    window_size_seconds: float = 9.01
    hop_size_seconds: float = 1.0
    fft_window_size_seconds: float = 0.02
    fft_hop_size_seconds: float = 0.01
    num_mel: int = 120


class DNSMOSp808(nn.Module, PyTorchModelHubMixin):
    """DNSMOSp808 model."""

    def __init__(self, config: DNSMOSp808Config) -> None:
        """Initialize the DNSMOSp808 model."""
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            ConvBlock(1, 32, 3, 1, pool=True),
            ConvBlock(32, 32, 3, 1, pool=True),
            ConvBlock(32, 32, 3, 1),
            ConvBlock(32, 32, 3, 1, pool=True),
            ConvBlock(32, 64, 3, 1),
            nn.AdaptiveMaxPool2d(1),
            ConvBlock(64, 64, 1, 0),
            ConvBlock(64, 64, 1, 0),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        to_samples = _get_to_samples(self.config.sample_rate)
        self.window_size = to_samples(self.config.window_size_seconds)
        self.hop_size = to_samples(self.config.hop_size_seconds)
        self.fft_window_size = to_samples(self.config.fft_window_size_seconds)
        self.fft_hop_size = to_samples(self.config.fft_hop_size_seconds)
        self.num_fft = self.fft_window_size + 1
        self.fft_window = nn.Buffer(
            torch.hann_window(self.num_fft, periodic=True),
            persistent=False,
        )
        self.mel_fbank = nn.Buffer(
            self._mel_filterbank(
                self.config.sample_rate,
                self.num_fft,
                self.config.num_mel,
            ),
            persistent=False,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        """Compute the DNSMOS of the input waveform.

        Args:
            waveform: [batch_size, num_channels == 1, input_num_samples]
            sample_rate: int

        Returns:
            [batch_size]

        """
        # Resample to target freq
        # [batch_size, num_channels, num_samples]
        x = resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=self.config.sample_rate,
        )

        # Pad so we have at least one window
        if x.shape[-1] < self.window_size:
            num_tiles = math.ceil(self.window_size / x.shape[-1])
            x = torch.tile(x, (1, 1, num_tiles))[..., : self.window_size]

        # Get overlapping windows of 9 seconds with 1 second hop
        # [batch_size, num_channels, num_hops, window_size]
        x = x.unfold(-1, self.window_size, self.hop_size)
        batch_size, num_channels, num_hops, window_size = x.shape

        # Pack dims into batch
        # [batch_size * num_channels * num_hops, window_size]
        x = x.reshape(-1, window_size)
        x = x[..., : -self.fft_hop_size]

        # [batch_size * num_channels * num_hops, n_freq, n_frames]
        power = (
            torch.stft(
                x,
                n_fft=self.num_fft,
                win_length=self.num_fft,
                hop_length=self.fft_hop_size,
                center=True,
                return_complex=True,
                window=self.fft_window,
                pad_mode="constant",
            )
            .abs()
            .pow(2)
        )
        # [batch_size * num_channels * num_hops, n_mels, n_frames]
        x = self.mel_fbank @ power
        x = (self._power_to_db(x) + 40.0) / 40.0
        # [batch_size * num_channels * num_hops, 1, n_frames, n_mels]
        x = x.transpose(-1, -2).unsqueeze(1)

        # Run net
        # [batch_size * num_channels * num_hops, 1]
        x: torch.Tensor = self.net(x)

        # Unpack batch dims
        # [batch_size, num_channels, num_hops, window_size]
        x = x.reshape(batch_size, num_channels, num_hops, 1)

        # DNSMOS of signal is just average of windows and channels
        return x.mean(dim=(1, 2))

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        if hz >= min_log_hz:
            return min_log_mel + math.log(hz / min_log_hz) / logstep
        return hz / f_sp

    @staticmethod
    def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        freqs = f_sp * mels
        log_t = mels >= min_log_mel
        return torch.where(
            log_t,
            min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
            freqs,
        )

    @staticmethod
    def _mel_filterbank(
        sample_rate: int,
        n_fft: int,
        n_mels: int,
    ) -> torch.Tensor:
        # Librosa-compatible Slaney banks using actual rfft bin frequencies.
        fftfreqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).to(
            torch.float32,
        )
        mel_f = DNSMOSp808._mel_to_hz(
            torch.linspace(
                DNSMOSp808._hz_to_mel(0.0),
                DNSMOSp808._hz_to_mel(sample_rate / 2.0),
                n_mels + 2,
            ),
        )
        fdiff = torch.diff(mel_f)
        ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
        lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
        upper = ramps[2:] / fdiff[1:].unsqueeze(1)
        weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        return weights * enorm.unsqueeze(1)

    @staticmethod
    def _power_to_db(
        power: torch.Tensor,
        *,
        amin: float = 1e-10,
        top_db: float = 80.0,
    ) -> torch.Tensor:
        log_spec = 10.0 * power.clamp_min(amin).log10()
        ref = power.amax(dim=(-2, -1), keepdim=True).clamp_min(amin)
        log_spec = log_spec - 10.0 * ref.log10()
        return torch.maximum(
            log_spec,
            log_spec.amax(dim=(-2, -1), keepdim=True) - top_db,
        )

    @classmethod
    def from_original_onnx(cls, onnx_model: "ModelProto") -> Self:
        """Load the DNSMOSp808 model from the original ONNX model."""
        state = _onnx_initializer_tensors(onnx_model)
        model = cls(DNSMOSp808Config())
        model.net.load_state_dict(
            {
                "0.0.weight": state["conv2d_5/kernel:0"],
                "0.0.bias": state["conv2d_5/bias:0"],
                "1.0.weight": state["conv2d_6/kernel:0"],
                "1.0.bias": state["conv2d_6/bias:0"],
                "2.0.weight": state["conv2d_7/kernel:0"],
                "2.0.bias": state["conv2d_7/bias:0"],
                "3.0.weight": state["conv2d_8/kernel:0"],
                "3.0.bias": state["conv2d_8/bias:0"],
                "4.0.weight": state["conv2d_9/kernel:0"],
                "4.0.bias": state["conv2d_9/bias:0"],
                "6.0.weight": _matmul_to_conv1x1(
                    state[
                        "mos_estimator_small_1/dense_3/MatMul/ReadVariableOp/resource:0"
                    ],
                ),
                "6.0.bias": state[
                    "mos_estimator_small_1/dense_3/BiasAdd/ReadVariableOp/resource:0"
                ],
                "7.0.weight": _matmul_to_conv1x1(
                    state[
                        "mos_estimator_small_1/dense_4/MatMul/ReadVariableOp/resource:0"
                    ],
                ),
                "7.0.bias": state[
                    "mos_estimator_small_1/dense_4/BiasAdd/ReadVariableOp/resource:0"
                ],
                "8.weight": _matmul_to_conv1x1(
                    state[
                        "mos_estimator_small_1/dense_5/MatMul/ReadVariableOp/resource:0"
                    ],
                ),
                "8.bias": state[
                    "mos_estimator_small_1/dense_5/BiasAdd/ReadVariableOp/resource:0"
                ],
            },
        )
        return model


@dataclass(frozen=True, kw_only=True)
class DNSMOSp835Config:
    """Configuration for DNSMOSp835 model.

    Args:
        sample_rate: int: sample rate expected by the model (don't change)
        window_size_seconds: float: window size in seconds
        hop_size_seconds: float: hop size in seconds
            (change to 9 for faster inference if you don't care about the spec)
        fft_num: int: FFT size (n_fft)
        fft_window_size_seconds: float: FFT window size in seconds
        fft_hop_size_seconds: float: FFT hop size in seconds
        personalized: bool: whether the model is personalized or not
        polyfit_regular: SIG, BAK, OVRL polynomial coeffs
            (highest-degree-first; quadratic padded with a leading 0)
        polyfit_personalized: SIG, BAK, OVRL polynomial coeffs
            for the personalized model (highest-degree-first)

    """

    sample_rate: int = 16_000
    window_size_seconds: float = 9.01
    hop_size_seconds: float = 1.0
    fft_num: int = 320
    fft_window_size_seconds: float = 0.02
    fft_hop_size_seconds: float = 0.01
    personalized: bool = False
    # SIG, BAK, OVRL; highest-degree-first.
    # Regular (quadratic) is padded with a leading 0.
    polyfit_regular: tuple[tuple[float, ...], ...] = (
        (0.0, -0.08397278, 1.22083953, 0.0052439),
        (0.0, -0.13166888, 1.60915514, -0.39604546),
        (0.0, -0.06766283, 1.11546468, 0.04602535),
    )
    polyfit_personalized: tuple[tuple[float, ...], ...] = (
        (-0.01019296, 0.02751166, 1.19576786, -0.24348726),
        (-0.04976499, 0.44276479, -0.1644611, 0.96883132),
        (-0.00533021, 0.005101, 1.18058466, -0.11236046),
    )


class DNSMOSp835(nn.Module, PyTorchModelHubMixin):
    """DNSMOSp835 model."""

    def __init__(self, config: DNSMOSp835Config) -> None:
        """Initialize the DNSMOSp835 model."""
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            ConvBlock(1, 128, 3, 1),
            ConvBlock(128, 64, 3, 1),
            ConvBlock(64, 64, 3, 1),
            ConvBlock(64, 32, 3, 1, pool=True),
            ConvBlock(32, 32, 3, 1, pool=True),
            ConvBlock(32, 32, 3, 1, pool=True),
            ConvBlock(32, 64, 3, 1),
            nn.AdaptiveMaxPool2d(1),
            ConvBlock(64, 128, 1, 0),
            ConvBlock(128, 64, 1, 0),
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
        )
        polyfit = (
            self.config.polyfit_personalized
            if config.personalized
            else self.config.polyfit_regular
        )
        self.polyfit = nn.Buffer(torch.tensor(polyfit, dtype=torch.float32))

        to_samples = _get_to_samples(self.config.sample_rate)
        self.window_size = to_samples(self.config.window_size_seconds)
        self.hop_size = to_samples(self.config.hop_size_seconds)
        self.fft_num = self.config.fft_num
        self.fft_window_size = to_samples(self.config.fft_window_size_seconds)
        self.fft_hop_size = to_samples(self.config.fft_hop_size_seconds)
        n_bins = self.fft_num // 2 + 1
        self.stft_real = nn.Conv1d(
            1,
            n_bins,
            kernel_size=self.fft_window_size,
            stride=self.fft_hop_size,
            bias=False,
        )
        self.stft_imag = nn.Conv1d(
            1,
            n_bins,
            kernel_size=self.fft_window_size,
            stride=self.fft_hop_size,
            bias=False,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        """Compute the DNSMOS of the input waveform.

        Args:
            waveform: [batch_size, num_channels == 1, input_num_samples]
            sample_rate: int

        Returns:
            [batch_size, 3]  (SIG, BAK, OVRL)

        """
        # Resample to target freq
        # [batch_size, num_channels, num_samples]
        x = resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=self.config.sample_rate,
        )

        # Pad so we have at least one window
        if (diff := self.window_size - x.shape[-1]) > 0:
            x = F.pad(x, (0, diff), mode="reflect")

        # Get overlapping windows of 9 seconds with 1 second hop
        # [batch_size, num_channels, num_hops, window_size]
        x = x.unfold(-1, self.window_size, self.hop_size)
        batch_size, num_channels, num_hops, window_size = x.shape

        # Pack dims into batch
        # [batch_size * num_channels * num_hops, window_size]
        x = x.reshape(-1, window_size)

        # Learned STFT (Keras-style Conv1d) then log-power spectrogram
        # [batch_size * num_channels * num_hops, 1, window_size]
        x = x.unsqueeze(1)
        real = self.stft_real(x)
        imag = self.stft_imag(x)
        x = (real.square() + imag.square()).clamp_min(1e-12).log10()
        # [batch_size * num_channels * num_hops, 1, num_fft_windows, n_bins]
        x = x.transpose(-1, -2).unsqueeze(1)

        # Run net
        # [batch_size * num_channels * num_hops, 3, 1, 1]
        x: torch.Tensor = self.net(x)

        # Unpack batch dims
        # [batch_size, num_channels, num_hops, 3]  (SIG, BAK, OVRL)
        x = x.reshape(batch_size, num_channels, num_hops, 3)

        # Polyfit per hop, then average hops and channels
        x = (
            self.polyfit[:, 0] * x.pow(3)
            + self.polyfit[:, 1] * x.pow(2)
            + self.polyfit[:, 2] * x
            + self.polyfit[:, 3]
        )
        return x.mean(dim=(1, 2))

    @classmethod
    def from_original_onnx(
        cls,
        onnx_model: "ModelProto",
        *,
        personalized: bool = False,
    ) -> Self:
        """Load the DNSMOSp835 model from the original ONNX model."""
        state = _onnx_initializer_tensors(onnx_model)
        config = DNSMOSp835Config(personalized=personalized)
        model = cls(config)
        with torch.no_grad():
            model.stft_real.weight.copy_(
                state["time2freq/stft-real/kernel:0"].permute(0, 2, 1),
            )
            model.stft_imag.weight.copy_(
                state["time2freq/stft-imag/kernel:0"].permute(0, 2, 1),
            )
        model.net.load_state_dict(
            {
                "0.0.weight": state["conv2d/kernel:0"],
                "0.0.bias": state["conv2d/bias:0"],
                "1.0.weight": state["conv2d_1/kernel:0"],
                "1.0.bias": state["conv2d_1/bias:0"],
                "2.0.weight": state["conv2d_2/kernel:0"],
                "2.0.bias": state["conv2d_2/bias:0"],
                "3.0.weight": state["conv2d_3/kernel:0"],
                "3.0.bias": state["conv2d_3/bias:0"],
                "4.0.weight": state["conv2d_4/kernel:0"],
                "4.0.bias": state["conv2d_4/bias:0"],
                "5.0.weight": state["conv2d_5/kernel:0"],
                "5.0.bias": state["conv2d_5/bias:0"],
                "6.0.weight": state["conv2d_6/kernel:0"],
                "6.0.bias": state["conv2d_6/bias:0"],
                "8.0.weight": _matmul_to_conv1x1(
                    state[
                        "mos_estimator_logpow/dense/MatMul/ReadVariableOp/resource:0"
                    ],
                ),
                "8.0.bias": state[
                    "mos_estimator_logpow/dense/BiasAdd/ReadVariableOp/resource:0"
                ],
                "9.0.weight": _matmul_to_conv1x1(
                    state[
                        "mos_estimator_logpow/dense_1/MatMul/ReadVariableOp/resource:0"
                    ],
                ),
                "9.0.bias": state[
                    "mos_estimator_logpow/dense_1/BiasAdd/ReadVariableOp/resource:0"
                ],
                "10.weight": _matmul_to_conv1x1(
                    state[
                        "mos_estimator_logpow/dense_3/MatMul/ReadVariableOp/resource:0"
                    ],
                ),
                "10.bias": state[
                    "mos_estimator_logpow/dense_3/BiasAdd/ReadVariableOp/resource:0"
                ],
            },
        )
        return model


class ConvBlock(nn.Sequential):
    """ConvBlock module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        *,
        pool: bool = False,
    ) -> None:
        """Initialize the ConvBlock module."""
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        super().__init__(*layers)
