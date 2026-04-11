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

## Notes

See [`issues.md`](issues.md) for the dependency and compatibility issues encountered during development, together with the fixes applied.
