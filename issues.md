# SAM-Audio Docker — Issues and Fixes

## Issue 1: `decord` has no ARM64 wheels — pip resolver blocks entire install

**Problem:** `perception-models` (a SAM-Audio dependency) requires `decord`, which has no `aarch64` wheels on PyPI. pip's dependency resolver treats this as a hard failure and refuses to install the entire dependency tree — even a simple `pip install sam-audio` fails immediately.

**Fix:** Install `perception-models` with `--no-deps` to bypass the decord requirement. `decord` is only used for video decoding in perception-models, which is irrelevant for SAM-Audio's audio separation pipeline.

---

## Issue 2: `torchaudio` not bundled in NGC ARM64 image

**Problem:** The NGC `nvcr.io/nvidia/pytorch:25.03-py3` image for ARM64 does **not** include `torchaudio` (unlike the x86 variant). This was discovered when the Dockerfile's diagnostic step (`import torchaudio`) failed at build time — the initial plan assumed NGC bundled it on all architectures.

**Fix:** Install `torchaudio==2.7.0` from PyPI with `--no-deps`. The `--no-deps` flag is needed because torchaudio 2.7.0 declares `torch==2.7.0` as a dependency, but NGC's torch reports as `2.7.0a0+7c8ec84dab.nv25.03` — pip considers this a version mismatch even though the ABI is compatible.

---

## Issue 3: `torchaudio` version mismatch — CUDA 13 vs CUDA 12

**Problem:** Without a version pin, `pip install torchaudio` pulls the latest version (2.11.0), which is compiled against CUDA 13 and requires `libcudart.so.13`. The NGC image ships CUDA 12.8, so importing torchaudio 2.11.0 crashes with `OSError: libcudart.so.13: cannot open shared object file`.

**Fix:** Pin `torchaudio==2.7.0`, which matches the NGC PyTorch 2.7 ABI and is linked against CUDA 12. This was not obvious — the first attempt used an unpinned install, and the error only appeared at import time, not during `pip install`.

---

## Issue 4: `audiobox_aesthetics` silently overwrites `torchaudio` pin

**Problem:** Even after pinning torchaudio to 2.7.0, installing `audiobox_aesthetics` with normal dependency resolution pulls in `torchaudio==2.11.0` as a transitive dependency, silently replacing the pinned version. The CUDA 13 error then returns at import time.

**Fix:** Install `audiobox_aesthetics` with `--no-deps`, and add a final `pip install --no-deps --force-reinstall torchaudio==2.7.0` as the last pip step in the Dockerfile to guarantee nothing has overwritten it.

---

## Issue 5: `descript-audiotools` requires `protobuf<3.20`

**Problem:** `descript-audiotools` (a dependency of `dacvae`, which is a dependency of SAM-Audio) pins `protobuf<3.20,>=3.9.2`. The NGC image ships `protobuf==4.24.4` (used by TensorBoard and other NGC-bundled tools). pip's resolver cannot satisfy both constraints and refuses to install.

**Fix:** Install `descript-audiotools` with `--no-deps`. protobuf 4.x maintains backwards compatibility with the 3.x API for the features audiotools uses, so the pin is overly conservative. However, this means all of descript-audiotools' own dependencies (`flatten-dict`, `julius`, `ffmpy`, `markdown2`, `pyloudnorm`, `randomname`, `pystoi`, `torch-stoi`, `importlib-resources`) must be installed explicitly in a separate pip call.

---

## Issue 6: `xformers` unavailable on ARM64

**Problem:** `perception-models` hard-imports `xformers.ops` (specifically `AttentionBias` and `fmha.memory_efficient_attention`) in its `core/transformer.py`. This import runs as part of `from sam_audio import SAMAudio`. xformers has no ARM64 wheels on PyPI, and building from source requires CUDA GPU kernel compilation at build time — which fails on a CPU-only build host. The xformers maintainers have stated ARM64 support is not planned (GitHub issue facebookresearch/xformers#1071).

**Import chain:** `sam_audio` → `sam_audio.model.model` → `core.audio_visual_encoder` → `core.audio_visual_encoder.pe` → `core.audio_visual_encoder.transformer` → `core.transformer` → `xformers.ops`

**Fix:** Create a minimal Python stub package at the xformers import path:
- `xformers/__init__.py` — empty
- `xformers/ops/__init__.py` — provides `AttentionBias` as an empty class, re-exports `fmha`
- `xformers/ops/fmha.py` — provides `memory_efficient_attention` that raises `NotImplementedError`
- `xformers/profiler/__init__.py` — stub for `MemSnapshotsProfiler`, `PyTorchProfiler`, `profile()`

SAM-Audio uses `flex_attention` (from PyTorch itself), not xformers' `fmha`, so the stub is never called on the actual inference path.

---

## Issue 7: `torchcodec` 0.11.0 links CUDA 13

**Problem:** `torchcodec` is imported by perception-models' `core/audio_visual_encoder/transforms.py` for `AudioDecoder` and `VideoDecoder`. The only ARM64 wheel available on PyPI (version 0.11.0) is compiled against CUDA 13, producing `OSError: libnvrtc.so.13: cannot open shared object file` at import time. There are no older ARM64-compatible versions — PyPI only offers `0.0.0.dev0` and `0.11.0` for aarch64.

**Import chain:** `sam_audio` → `core.audio_visual_encoder` → `core.audio_visual_encoder.transforms` → `torchcodec.decoders` → `torchcodec._core.ops` → tries to load `libtorchcodec*.so` → fails on missing `libnvrtc.so.13`.

**Fix:** Create a stub: `torchcodec/__init__.py` (empty) and `torchcodec/decoders/__init__.py` providing `AudioDecoder` and `VideoDecoder` as empty classes. SAM-Audio uses torchaudio for audio I/O, not torchcodec.

---

## Issue 8: NGC base image Python version constraint

**Problem:** SAM-Audio requires `Python>=3.11`. NGC PyTorch images before 24.12 ship Python 3.10, which is incompatible. This was identified during the research phase, not as a build failure.

**Fix:** Use `nvcr.io/nvidia/pytorch:25.03-py3` which ships Python 3.12.3. The ARM64 availability was confirmed via `docker manifest inspect`.

---

## Issue 9: Disk space for NGC image pull

**Problem:** The NGC PyTorch ARM64 image is ~15 GB compressed. The default 30 GB boot disk on Google T2A instances only had 11 GB free, causing the image pull to fail with `no space left on device` during layer extraction (specifically when writing `libtriton.so`).

**Fix:** Resized the GCP persistent disk to 80 GB, then expanded the partition and filesystem:
```
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```
This gave 73 GB free, more than enough for the image and build layers.

---

## Issue 10: Missing transitive deps from `descript-audiotools` (whack-a-mole)

**Problem:** Because `descript-audiotools` was installed with `--no-deps` (due to the protobuf conflict), none of its own dependencies were installed. The build-time import check revealed missing packages one at a time: first `flatten_dict`, then `julius`, then `ffmpy`. Each missing package only surfaced after the previous one was added and the image rebuilt.

**Fix:** Retrieved the full dependency list of descript-audiotools (`pip show descript-audiotools | grep Requires`) and added all of them to the transitive deps install step in one batch: `flatten-dict`, `julius`, `ffmpy`, `markdown2`, `pyloudnorm`, `randomname`, `pystoi`, `torch-stoi`, `importlib-resources`.

**Lesson:** When using `--no-deps`, always audit the full dependency list upfront rather than discovering them one by one through build failures.

---

## Issue 11: Over-engineering the Dockerfile — too many moving parts

**Problem:** The initial Dockerfile attempts tried to solve every possible dependency issue at once: manually managing ~30 transitive deps for perception-models, building decord from source on ARM64, installing xformers from source, handling video-related packages, and trying to make the full production inference stack work. Each additional layer introduced new conflicts, making the build increasingly fragile. Multiple iterations failed because the complexity created cascading dependency conflicts.

**Fix:** Scrapped the over-engineered Dockerfile entirely and started from a minimal approach:
1. Install only the 7 packages that truly need `--no-deps` (the ones with actual unresolvable conflicts)
2. Create stubs for only the 2 packages that are in the import chain but genuinely cannot be installed on ARM64 (xformers, torchcodec)
3. Let pip resolve everything else normally in a single `pip install` call
4. Add a build-time import check to catch any remaining missing deps

This reduced the Dockerfile from ~100 lines across many fragile layers to ~57 lines with clear separation between conflicting and non-conflicting packages.

---

## Issue 12: pip still resolves `decord` transitively even after pre-installing perception-models

**Problem:** One early approach was to pre-install `perception-models` with `--no-deps`, then install `sam-audio` normally, hoping pip would skip already-installed packages. This did not work — pip's resolver re-evaluates the full dependency tree of perception-models (including `decord`) when processing sam-audio's dependency on it, regardless of whether perception-models is already installed.

**Fix:** Install `sam-audio` itself with `--no-deps` as well, then install all resolvable transitive deps in a separate pip call that doesn't reference any of the problematic packages.
