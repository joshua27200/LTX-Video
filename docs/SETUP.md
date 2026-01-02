## Local Setup (Step-by-Step)

Follow these steps to get LTX-Video running locally with the smaller 2B distilled checkpoint.

### 1. Prerequisites
- Python 3.10+ and a working CUDA 12.2+ GPU (or Apple MPS). CPU-only is possible but slow.
- `git lfs` installed (required for large model files).
- A Hugging Face token with access to the LTX-Video repository. Create one at https://huggingface.co/settings/tokens and keep it handy.

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[inference]
```

### 3. Authenticate with Hugging Face (to download weights)
```bash
huggingface-cli login  # paste your token when prompted
# Optional: speed up downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### 4. Download the 2B distilled checkpoint and upscaler
Place the weights in the repository root so they match the defaults in
`configs/ltxv-2b-0.9.8-distilled.yaml`:
```bash
mkdir -p weights
huggingface-cli download Lightricks/LTX-Video ltxv-2b-0.9.8-distilled.safetensors \
  --local-dir weights --local-dir-use-symlinks False
huggingface-cli download Lightricks/LTX-Video ltxv-spatial-upscaler-0.9.8.safetensors \
  --local-dir weights --local-dir-use-symlinks False

# Move them next to the repo root (or update the config to point into weights/)
cp weights/ltxv-2b-0.9.8-distilled.safetensors .
cp weights/ltxv-spatial-upscaler-0.9.8.safetensors .
```
*(If you prefer to keep weights in `weights/`, edit `checkpoint_path` and
`spatial_upscaler_model_path` in the YAML to point to those files.)*

### 5. Run a quick smoke test
Use a small resolution and frame count to validate the setup:
```bash
python inference.py \
  --prompt "A cinematic shot of waves rolling onto a beach at sunset" \
  --height 256 --width 320 --num_frames 49 --frame_rate 8 --seed 42 \
  --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml \
  --output_path outputs/setup-smoketest
```
The script will write the video to `outputs/setup-smoketest`. If the weights are
present locally, no additional downloads are needed; otherwise they are fetched
via the Hugging Face Hub using your credentials.

### 6. Troubleshooting
- **Hugging Face download errors**: ensure `huggingface-cli login` succeeded and
  that your token has access to `Lightricks/LTX-Video`.
- **No GPU detected**: the pipeline will fall back to CPU but will be very slow.
  Verify `nvidia-smi` (CUDA) or `torch.backends.mps.is_available()` (macOS).
- **ffmpeg missing**: install `ffmpeg` from your package manager so
  `imageio[ffmpeg]` can encode the output video.
