#!/usr/bin/env python3
"""
basic_text_prompting.py - SAM-Audio text-prompted sound separation.

Standalone script based on the "Basic Text Prompting" example from the
SAM-Audio README (https://github.com/facebookresearch/sam-audio).

Verifies that all sam-audio dependencies resolve correctly on ARM64+CUDA,
then optionally runs inference if model weights are available.

Usage:
    # Verify imports only (no weights needed):
    python basic_text_prompting.py --verify-only

    # Run full inference (requires HF auth + model access):
    python basic_text_prompting.py --audio input.wav --description "a dog barking"
"""

import argparse
import sys


def verify_imports():
    """Verify all critical imports resolve without error."""
    checks = {}

    # Core PyTorch stack
    import torch
    checks["torch"] = torch.__version__
    checks["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        checks["cuda_device"] = torch.cuda.get_device_name(0)

    import torchaudio
    checks["torchaudio"] = torchaudio.__version__

    import torchvision
    checks["torchvision"] = torchvision.__version__

    # SAM-Audio core
    from sam_audio import SAMAudio, SAMAudioProcessor
    checks["sam_audio"] = "OK"

    # Key transitive dependencies
    import einops
    checks["einops"] = "OK"

    import transformers
    checks["transformers"] = transformers.__version__

    import torchcodec
    checks["torchcodec"] = "OK"

    import torchdiffeq
    checks["torchdiffeq"] = "OK"

    import pydub
    checks["pydub"] = "OK"

    # flex_attention check (known issue on older PyTorch)
    try:
        from torch.nn.attention.flex_attention import flex_attention
        checks["flex_attention"] = "OK"
    except (ImportError, ModuleNotFoundError):
        checks["flex_attention"] = "MISSING (may cause runtime errors)"

    return checks


def run_inference(audio_path, description):
    """Run SAM-Audio text-prompted separation (from README example)."""
    from sam_audio import SAMAudio, SAMAudioProcessor
    import torchaudio
    import torch

    print(f"Loading model: facebook/sam-audio-large")
    model = SAMAudio.from_pretrained("facebook/sam-audio-large")
    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
    model = model.eval().cuda()

    batch = processor(
        audios=[audio_path],
        descriptions=[description],
    ).to("cuda")

    with torch.inference_mode():
        result = model.separate(batch, predict_spans=False, reranking_candidates=1)

    sample_rate = processor.audio_sampling_rate
    torchaudio.save("target.wav", result.target.cpu(), sample_rate)
    torchaudio.save("residual.wav", result.residual.cpu(), sample_rate)
    print("Saved: target.wav, residual.wav")


def main():
    parser = argparse.ArgumentParser(description="SAM-Audio text-prompted separation")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify imports, do not load model",
    )
    parser.add_argument("--audio", type=str, help="Path to input audio file")
    parser.add_argument(
        "--description", type=str, help="Text description of sound to isolate"
    )
    args = parser.parse_args()

    # Always run verification first
    print("=" * 60)
    print("SAM-Audio Dependency Verification")
    print("=" * 60)

    try:
        checks = verify_imports()
        for k, v in checks.items():
            # cuda_available is informational — not a failure on CPU-only hosts
            if k == "cuda_available":
                status = "INFO"
            else:
                status = "PASS" if v not in (False, None) else "FAIL"
            print(f"  [{status}] {k}: {v}")
        # Exclude cuda_available from pass/fail — it depends on hardware, not deps
        all_ok = all(
            v not in (False, None)
            for k, v in checks.items()
            if k != "cuda_available"
        )
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        sys.exit(1)

    if not all_ok:
        print("\nSome checks failed. See above.")
        sys.exit(1)

    print("\nAll imports verified successfully.")

    if args.verify_only:
        sys.exit(0)

    # Inference mode
    if not args.audio or not args.description:
        print(
            "\nProvide --audio and --description for inference, or use --verify-only."
        )
        sys.exit(1)

    try:
        run_inference(args.audio, args.description)
    except Exception as e:
        print(f"\nInference failed: {e}")
        if "gated" in str(e).lower() or "401" in str(e) or "auth" in str(e).lower():
            print("This model requires HuggingFace authentication.")
            print("Run: huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    main()
