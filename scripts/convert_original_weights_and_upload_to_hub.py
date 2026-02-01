import argparse  # noqa: INP001
from pathlib import Path

import onnx
from huggingface_hub import HfApi

from dnsmos_pytorch.dnsmos import DNSMOSp808, DNSMOSp835

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GITHUB_URL = "https://github.com/rziga/dnsmos-pytorch"


def _hub_readme() -> str:
    """Return the repo README with a GitHub link prepended."""
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return f"{_GITHUB_URL}\n\n{readme}"


def _upload_readme(repo_id: str, *, readme: str, api: HfApi) -> None:
    """Upload README.md to a Hub model repo."""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload README.md",
    )


def main() -> None:
    """Convert original ONNX weights to PyTorch and upload them to the Hub."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert original ONNX weights to PyTorch "
            "and upload them to the Hub."
        ),
    )
    parser.add_argument(
        "--original-weights",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets"
        / "original_weights",
        help="Directory with original ONNX weights.",
    )
    parser.add_argument(
        "--hub-user",
        default="rziga",
        help="Hugging Face Hub username or org.",
    )
    args = parser.parse_args()
    api = HfApi()
    readme = _hub_readme()

    # Convert and upload DNSMOSp808.
    p808 = DNSMOSp808.from_original_onnx(
        onnx.load(args.original_weights / "DNSMOS" / "model_v8.onnx"),
    )
    p808.eval()
    p808_repo = f"{args.hub_user}/DNSMOSp808"
    p808.push_to_hub(p808_repo, config=p808.config)
    _upload_readme(p808_repo, readme=readme, api=api)

    # Convert and upload DNSMOSp835
    p835 = DNSMOSp835.from_original_onnx(
        onnx.load(args.original_weights / "DNSMOS" / "sig_bak_ovr.onnx"),
        personalized=False,
    )
    p835.eval()
    p835_repo = f"{args.hub_user}/DNSMOSp835"
    p835.push_to_hub(p835_repo, config=p835.config)
    _upload_readme(p835_repo, readme=readme, api=api)

    # Convert and upload personalized DNSMOSp835
    p835_personalized = DNSMOSp835.from_original_onnx(
        onnx.load(args.original_weights / "pDNSMOS" / "sig_bak_ovr.onnx"),
        personalized=True,
    )
    p835_personalized.eval()
    p835_personalized_repo = f"{args.hub_user}/DNSMOSp835-personalized"
    p835_personalized.push_to_hub(
        p835_personalized_repo,
        config=p835_personalized.config,
    )
    _upload_readme(p835_personalized_repo, readme=readme, api=api)


if __name__ == "__main__":
    main()
