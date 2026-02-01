import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from dnsmos_pytorch.dnsmos import (
    DNSMOSp808,
    DNSMOSp808Config,
    DNSMOSp835,
    DNSMOSp835Config,
)


def _jsonable_config(config: object) -> object:
    return json.loads(json.dumps(asdict(config)))


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (DNSMOSp808, DNSMOSp808Config()),
        (DNSMOSp835, DNSMOSp835Config()),
        (DNSMOSp835, DNSMOSp835Config(personalized=True)),
    ],
    ids=["p808", "p835", "p835-personalized"],
)
def test_save_and_load(tmp_path: Path, model_cls, config) -> None:
    model = model_cls(config)
    model.eval()
    model.save_pretrained(tmp_path)

    loaded = model_cls.from_pretrained(tmp_path, strict=True)
    loaded.eval()

    assert type(loaded.config) is type(model.config)
    assert _jsonable_config(loaded.config) == _jsonable_config(model.config)

    orig_state = model.state_dict()
    loaded_state = loaded.state_dict()
    assert orig_state.keys() == loaded_state.keys()
    for key, tensor in orig_state.items():
        torch.testing.assert_close(tensor, loaded_state[key], msg=key)

    waveform = torch.randn(1, 1, int(9.01 * 16_000))
    with torch.no_grad():
        torch.testing.assert_close(model(waveform), loaded(waveform))
