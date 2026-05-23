#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
One-time export tool: turn a pre-trained ResNet-20 (CIFAR-10) checkpoint
into the bf16 binaries that tt-foil's test_cifar10_resnet20 loads at
runtime.

What this script does:

  1. Defines the canonical CIFAR-10 ResNet-20 architecture (the
     akamaster variant — 3 stages × 3 BasicBlocks, option-A
     channel-doubling skip via zero-padded subsample).
  2. Loads the pretrained checkpoint from akamaster's GitHub release
     (~1 MB, downloaded once and cached under .cache/).
  3. Loads ONE CIFAR-10 test image (downloads the dataset once and
     caches it under .cache/cifar-10-batches-py/) — picks a fixed
     index so the test is reproducible.
  4. Folds each BatchNorm into the preceding Conv: the conv was
     originally bias-less, so the fold gives us (W', b') ready for
     the device's bias_relu_post stage.
  5. Runs a forward pass in float32 to get reference logits.
  6. Writes:
       data/cifar10_resnet20/weights.bin   (concatenated bf16 weights)
       data/cifar10_resnet20/image.bin     (one CIFAR-10 image, bf16)
       data/cifar10_resnet20/golden.bin    (10 reference logits, bf16)
       data/cifar10_resnet20/manifest.json (offsets, shapes, label)

The C++ test reads manifest.json to find each layer's offset/shape in
weights.bin; the binary layout itself stays opaque to C++.

Usage:
  python3 models/cifar10_resnet20/export.py

  --image-index N   pick CIFAR-10 test image N (default 0)
  --cache-dir DIR   where to keep downloads (default .cache/)
  --out-dir DIR     where to write binaries (default data/cifar10_resnet20/)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture (akamaster CIFAR-10 ResNet-20)
# ---------------------------------------------------------------------------

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # akamaster's "option A": zero-padded subsample skip.
            pad = (planes - in_planes) // 2
            self.shortcut = LambdaLayer(lambda x, pad=pad:
                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])  # global avg pool
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ---------------------------------------------------------------------------
# BatchNorm folding
#   y = gamma * (conv(x) - mu) / sqrt(var + eps) + beta
#     = (gamma / sqrt(var + eps)) * conv(x) + (beta - gamma * mu / sqrt(var + eps))
# So with the conv being bias-less:
#   W' = (gamma / sqrt(var + eps))[:, None, None, None] * W
#   b' = beta - gamma * mu / sqrt(var + eps)
# ---------------------------------------------------------------------------

def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mu = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w = conv.weight.detach() * scale[:, None, None, None]
    b = beta - mu * scale
    return w, b


# ---------------------------------------------------------------------------
# bf16 serialisation
# ---------------------------------------------------------------------------

def to_bf16_bytes(t: torch.Tensor) -> bytes:
    """Lossy-round a tensor to bf16 (round-to-nearest-even) and return its
    raw little-endian bytes — same wire format the device kernels see."""
    t_bf16 = t.to(torch.bfloat16).contiguous()
    return t_bf16.view(torch.uint16).cpu().numpy().tobytes()


# ---------------------------------------------------------------------------
# CIFAR-10 sample loader (downloads the python pickle once)
# ---------------------------------------------------------------------------

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar10_test_image(cache_dir: Path, index: int):
    """Load CIFAR-10 test image at `index`, normalised the way the
    akamaster checkpoint expects. Returns (tensor[3,32,32], label_int).
    """
    import pickle
    import tarfile

    cache_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = cache_dir / "cifar-10-batches-py" / "test_batch"
    if not pickle_path.exists():
        tar_path = cache_dir / "cifar-10-python.tar.gz"
        if not tar_path.exists():
            print(f"[export] downloading CIFAR-10 test batch from {CIFAR_URL}")
            urllib.request.urlretrieve(CIFAR_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(cache_dir)
    with open(pickle_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    # batch[b'data'] shape (N, 3072) uint8 -> (3, 32, 32) per image
    raw = batch[b"data"][index].reshape(3, 32, 32).astype("float32") / 255.0
    label = batch[b"labels"][index]
    t = torch.from_numpy(raw)
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t, label


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

CHECKPOINT_URL = (
    "https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/"
    "pretrained_models/resnet20-12fca82f.th"
)


def load_pretrained(model: nn.Module, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "resnet20-12fca82f.th"
    if not path.exists():
        print(f"[export] downloading checkpoint from {CHECKPOINT_URL}")
        urllib.request.urlretrieve(CHECKPOINT_URL, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    # Strip "module." prefix from DataParallel-wrapped state dict.
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def collect_folded_layers(model: ResNet20):
    """Walk the model in execution order and yield (name, kind, tensor)
    tuples for every weight the device side will need."""

    # Stem
    w, b = fold_bn(model.conv1, model.bn1)
    yield "stem.conv.w", "conv", w
    yield "stem.bn.b",   "bias", b

    for stage_idx, layer in enumerate([model.layer1, model.layer2, model.layer3], start=1):
        for blk_idx, blk in enumerate(layer):
            base = f"layer{stage_idx}.{blk_idx}"
            w1, b1 = fold_bn(blk.conv1, blk.bn1)
            w2, b2 = fold_bn(blk.conv2, blk.bn2)
            yield f"{base}.conv1.w", "conv", w1
            yield f"{base}.bn1.b",   "bias", b1
            yield f"{base}.conv2.w", "conv", w2
            yield f"{base}.bn2.b",   "bias", b2

    # FC (no BN to fold)
    yield "fc.w", "fc", model.linear.weight.detach()
    yield "fc.b", "fc_bias", model.linear.bias.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--out-dir", default="data/cifar10_resnet20")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model + load weights -----------------------------------
    model = ResNet20(num_classes=10)
    load_pretrained(model, cache_dir)
    model.eval()

    # ---- Pull a CIFAR-10 test image ----------------------------------
    img, label = load_cifar10_test_image(cache_dir, args.image_index)
    print(f"[export] image index={args.image_index} label={label} ({CIFAR_CLASSES[label]})")

    # ---- Reference forward pass (fp32) --------------------------------
    # Also export per-stage activations so the C++ test can debug numeric
    # drift layer-by-layer.
    stage_activations = {}
    def hook(name):
        def f(_m, _inp, out):
            stage_activations[name] = out.detach().clone().squeeze(0)
        return f
    handles = []
    handles.append(model.bn1.register_forward_hook(hook("post_stem")))
    for li, layer in enumerate([model.layer1, model.layer2, model.layer3], start=1):
        for bi, blk in enumerate(layer):
            handles.append(blk.register_forward_hook(hook(f"post_layer{li}.{bi}")))
    with torch.no_grad():
        # bn1 captures pre-ReLU; pre-relu activation = bn1 output, post-
        # ReLU happens in F.relu inside ResNet.forward. We capture both:
        # the stage_activations dict above holds bn outputs (pre-ReLU),
        # plus we re-run hooks on block outputs (post-ReLU since basic
        # block applies its final ReLU before returning).
        logits = model(img.unsqueeze(0)).squeeze(0)
    for h in handles: h.remove()

    # Apply ReLU to the stem activation we captured (the model does that
    # in forward right after bn1, so the natural "post-stem" value is
    # post-ReLU).
    stage_activations["post_stem"] = torch.relu(stage_activations["post_stem"])

    argmax = int(torch.argmax(logits).item())
    print(f"[export] reference argmax={argmax} ({CIFAR_CLASSES[argmax]}), "
          f"max_logit={logits.max().item():.4f}")

    # ---- Pack weights -------------------------------------------------
    weights_path = out_dir / "weights.bin"
    manifest = {"format_version": 1, "layers": []}
    offset = 0
    with open(weights_path, "wb") as f:
        for name, kind, tensor in collect_folded_layers(model):
            buf = to_bf16_bytes(tensor)
            f.write(buf)
            manifest["layers"].append({
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "offset": offset,
                "bytes": len(buf),
            })
            offset += len(buf)

    print(f"[export] wrote {weights_path} ({offset} bytes, "
          f"{len(manifest['layers'])} layers)")

    # ---- Image + label + golden logits --------------------------------
    image_bytes = to_bf16_bytes(img)
    (out_dir / "image.bin").write_bytes(image_bytes)
    print(f"[export] wrote {out_dir / 'image.bin'} ({len(image_bytes)} bytes)")

    golden_bytes = to_bf16_bytes(logits)
    (out_dir / "golden.bin").write_bytes(golden_bytes)
    print(f"[export] wrote {out_dir / 'golden.bin'} ({len(golden_bytes)} bytes)")

    # ---- Per-stage golden activations (post-ReLU CHW) -----------------
    stages_dir = out_dir / "stages"
    stages_dir.mkdir(exist_ok=True)
    for name, t in stage_activations.items():
        path = stages_dir / f"{name}.bin"
        path.write_bytes(to_bf16_bytes(t.contiguous()))
        print(f"[export] wrote {path} shape={tuple(t.shape)}")

    manifest["image_index"] = args.image_index
    manifest["label"]       = label
    manifest["class_name"]  = CIFAR_CLASSES[label]
    manifest["ref_argmax"]  = argmax
    manifest["ref_class"]   = CIFAR_CLASSES[argmax]
    manifest["ref_logits_fp32"] = [float(v) for v in logits]
    manifest["classes"]     = CIFAR_CLASSES

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[export] wrote {manifest_path}")

    # Sibling text manifest, easier for the C++ test to parse than JSON
    # (no third-party deps). One line per layer.
    txt_path = out_dir / "manifest.txt"
    with open(txt_path, "w") as f:
        f.write("# Auto-generated. Format: <num_layers>, then "
                "<name offset_bytes total_bytes ndim s0 s1 s2 s3> per layer.\n")
        f.write(f"{len(manifest['layers'])}\n")
        for layer in manifest["layers"]:
            shape = list(layer["shape"]) + [0] * (4 - len(layer["shape"]))
            f.write(f"{layer['name']} {layer['offset']} {layer['bytes']} "
                    f"{len(layer['shape'])} {shape[0]} {shape[1]} {shape[2]} {shape[3]}\n")
        f.write(f"# ref_argmax {manifest['ref_argmax']}  "
                f"class {manifest['ref_class']}\n")
    print(f"[export] wrote {txt_path}")


if __name__ == "__main__":
    main()
