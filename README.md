# SAM-Audio Docker (Linux/ARM64 + CUDA)

A Docker image for [SAM-Audio](https://github.com/facebookresearch/sam-audio) on Linux/ARM64 with CUDA support, including a standalone script from the Basic Text Prompting example.

## Requirements

- ARM64 host
- Docker
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Build and Run

```bash
docker build -t sam-audio .
docker run --gpus all sam-audio
```

## x86_64 (RunPod / generic CUDA)

A separate Dockerfile, [`Dockerfile.x86_64`](Dockerfile.x86_64), builds the same stack for Linux/x86_64 + CUDA hosts (e.g. RunPod RTX 4090). It uses the same NGC PyTorch base image (multi-arch), so the dependency mitigations from `issues.md` carry over.

```bash
docker build -f Dockerfile.x86_64 -t sam-audio-x86 .
docker run --rm --gpus all sam-audio-x86
```

The default `CMD` only runs an import + CUDA sanity check (prints torch / torchaudio / torchcodec versions, `cuda_available`, GPU name). It does not download Hugging Face weights and does not require `HF_TOKEN`.

## Notes

See [`issues.md`](issues.md) for the dependency and compatibility issues encountered during development, together with the fixes applied.
