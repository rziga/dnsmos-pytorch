#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_ROOT}/assets/original_weights"

mkdir -p "${DEST}/DNSMOS" "${DEST}/pDNSMOS"

# GitHub blob pages serve HTML; curl the raw files instead.
BASE="https://raw.githubusercontent.com/microsoft/DNS-Challenge/master"

curl -fL -o "${DEST}/DNSMOS/model_v8.onnx" \
  "${BASE}/DNSMOS/DNSMOS/model_v8.onnx"
curl -fL -o "${DEST}/DNSMOS/sig_bak_ovr.onnx" \
  "${BASE}/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
curl -fL -o "${DEST}/pDNSMOS/sig_bak_ovr.onnx" \
  "${BASE}/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
