FROM nvcr.io/nvidia/pytorch:25.03-py3

# System packages: git (pip git+https deps), ffmpeg (pydub), libsndfile (soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# torchaudio: not bundled in NGC ARM64 image.
# --no-deps because NGC torch is a custom 2.7.0a0+nv build.
RUN pip install --no-cache-dir --no-deps torchaudio==2.7.0

# Packages that need --no-deps due to unresolvable conflicts:
#   sam-audio + 4 git deps : depend on decord (no ARM64 wheels)
#   descript-audiotools    : requires protobuf<3.20 (NGC has 4.x)
#   audiobox_aesthetics    : pulls torchaudio 2.11 (needs CUDA 13)
RUN pip install --no-cache-dir --no-deps \
        "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git" \
        "dacvae @ git+https://github.com/facebookresearch/dacvae.git" \
        "imagebind @ git+https://github.com/facebookresearch/ImageBind.git" \
        "laion-clap @ git+https://github.com/lematt1991/CLAP.git" \
        "perception-models @ git+https://github.com/facebookresearch/perception_models@unpin-deps" \
        "descript-audiotools>=0.7.2" \
        audiobox_aesthetics

# Stubs for packages that cannot be installed on ARM64 but are
# in the sam_audio import chain:
#   xformers  : no ARM64 wheels, needs GPU to build from source
#   torchcodec: only wheel (0.11) links CUDA 13 (libnvrtc.so.13)
RUN SITE=/usr/local/lib/python3.12/dist-packages && \
    mkdir -p $SITE/xformers/ops $SITE/xformers/profiler && \
    touch $SITE/xformers/__init__.py && \
    printf 'class AttentionBias: pass\nfrom . import fmha\n' > $SITE/xformers/ops/__init__.py && \
    printf 'def memory_efficient_attention(*a, **kw):\n    raise NotImplementedError("xformers unavailable on ARM64")\n' > $SITE/xformers/ops/fmha.py && \
    printf 'class MemSnapshotsProfiler: pass\nclass PyTorchProfiler: pass\ndef profile(*a, **kw):\n    import contextlib; return contextlib.nullcontext()\n' > $SITE/xformers/profiler/__init__.py && \
    mkdir -p $SITE/torchcodec/decoders && \
    touch $SITE/torchcodec/__init__.py && \
    printf 'class AudioDecoder: pass\nclass VideoDecoder: pass\n' > $SITE/torchcodec/decoders/__init__.py

# All remaining transitive deps — pip resolves these normally.
RUN pip install --no-cache-dir \
        einops "transformers>=4.54.0" timm huggingface-hub \
        pydub torchdiffeq librosa soundfile \
        "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f" \
        ftfy regex iopath types-regex argbind \
        "numba>=0.5.7" omegaconf scipy \
        flatten-dict julius ffmpy markdown2 pyloudnorm randomname \
        pystoi torch-stoi importlib-resources

# Re-pin torchaudio in case any dep above pulled in 2.11
RUN pip install --no-cache-dir --no-deps --force-reinstall torchaudio==2.7.0

# Verify the import chain works.
RUN python -c "from sam_audio import SAMAudio, SAMAudioProcessor; print('sam_audio: OK')"

WORKDIR /workspace
COPY basic_text_prompting.py /workspace/basic_text_prompting.py
CMD ["python", "basic_text_prompting.py", "--verify-only"]
