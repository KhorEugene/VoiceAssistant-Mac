# Copyright © 2023 Apple Inc.

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    try:
        full_model_path = str(model_path / "weights.npz")
        weights = mx.load(full_model_path)
    except Exception:
        full_model_path = str(model_path / "weights.safetensors")
        try:
            weights = mx.load(full_model_path)
        except Exception as e:
            print(f"My error: {e}")
            raise Exception()
    weights = tree_unflatten(list(weights.items()))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    model.update(weights)
    mx.eval(model.parameters())
    return model
