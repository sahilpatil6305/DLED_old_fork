#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DLED_PRETRAINED_ROOT="${DLED_PRETRAINED_ROOT:-$ROOT/training/pretrained/clip}"
mkdir -p "$DLED_PRETRAINED_ROOT"
"${PYTHON_BIN:-$ROOT/.venv/bin/python}" - "$ROOT" "$DLED_PRETRAINED_ROOT" <<'PY'
import hashlib, os, sys
from pathlib import Path
import urllib.request
root, cache = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
from clip.clip import _MODELS, _download
name = "ViT-B/16"
url = _MODELS[name]
path = Path(_download(url, root=str(cache)))
expected = url.rstrip('/').split('/')[-2]
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
vocab = root / "clip" / "bpe_simple_vocab_16e6.txt.gz"
vocab_url = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
vocab_sha256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
if not vocab.is_file() or hashlib.sha256(vocab.read_bytes()).hexdigest() != vocab_sha256:
    vocab.parent.mkdir(parents=True, exist_ok=True)
    temporary = vocab.with_suffix(vocab.suffix + ".download")
    urllib.request.urlretrieve(vocab_url, temporary)
    actual_vocab_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual_vocab_hash != vocab_sha256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Tokenizer vocabulary SHA256 mismatch: expected {vocab_sha256}, got {actual_vocab_hash}")
    temporary.replace(vocab)
actual_vocab_hash = hashlib.sha256(vocab.read_bytes()).hexdigest()
if actual_vocab_hash != vocab_sha256:
    raise SystemExit(f"Tokenizer vocabulary SHA256 mismatch for {vocab}")
print(f"Checkpoint verified: {path} (sha256 {actual})")
print(f"Tokenizer vocabulary verified: {vocab} (sha256 {actual_vocab_hash})")
PY
