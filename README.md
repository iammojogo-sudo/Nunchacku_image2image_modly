# Qwen-Image-Edit (Nunchaku) Image Edit — Modly Extension

Higher-quality **and** faster instruction-based image editing with Qwen-Image-Edit, quantized via **Nunchaku SVDQuant 4-bit**. SVDQuant keeps quality much closer to the full model than plain GGUF 4-bit, runs faster (optimized INT4/FP4 kernels, no on-the-fly dequant), and offloads down to ~3-4GB VRAM. Apache-2.0 and ungated — no login.

Same flow as the other editors: one image + one text instruction in, edited image(s) out. Great for novel viewpoints (rotate an object to show its back), relighting, object/background edits.

---

## Requirements (read this first)

- **NVIDIA GPU, Ampere or newer** (RTX 30-series / 40-series / 50-series). Nunchaku's INT4 kernels don't support older cards (GTX / RTX 20-series) — use the GGUF edit extension on those.
- **Recent NVIDIA driver** (CUDA 12.8-capable). Update it if you hit CUDA errors.
- **Windows:** the latest **Visual C++ Redistributable** must be installed, or Nunchaku fails to load its DLL.
- **VRAM:** runs on ~6GB up. 12GB is comfortable. Lower VRAM = lower `Blocks on GPU` = slower (see below).

---

## Installation

1. Open Modly → **Extensions** → **Install from GitHub** and paste this repo URL.
2. Setup installs PyTorch 2.8 + diffusers, then **auto-installs the matching Nunchaku wheel**. If that step fails (wheel/Python mismatch), see *Manual Nunchaku install* below.
3. Click **Download** on the Edit Image node. It pulls the base components (text encoder + VAE, skipping the 40GB original transformer) and the Nunchaku 4-bit transformer (~one file).

### Manual Nunchaku install (if auto-install failed)

Nunchaku ships as prebuilt wheels pinned to an exact **torch + Python + OS** combo. This extension pins **torch 2.8**. Find your Python tag, then install the matching wheel into the extension's venv:

```
# Python tag (cp311 / cp312 / cp313):
"<ext>/venv/Scripts/python.exe" -c "import sys;print('cp%d%d'%sys.version_info[:2])"

# install the wheel for torch2.8 + that cp tag + your OS:
"<ext>/venv/Scripts/python.exe" -m pip install <wheel-url>
```

Wheels: <https://github.com/nunchaku-tech/nunchaku/releases> or <https://huggingface.co/nunchaku-tech/nunchaku/tree/main>. Pick the file named like `nunchaku-<ver>+torch2.8-cp312-cp312-win_amd64.whl`. On Linux use `linux_x86_64`. After installing, verify with `... -c "import nunchaku"`.

---

## Usage (Workflows tab)

1. **Image** node → Edit Image node's image input.
2. **Text** node (your instruction, e.g. `rotate the camera to show the back of this object`) → Edit Image node's text input.
3. Edit Image output → **Preview Image**.
4. Run.

Number of Outputs = 1 emits a single image; 2–4 emits multiple variations (different seeds) as a list.

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| Quality (rank) | r128 | r128 = best quality; r32 = smaller/slightly faster |
| Blocks on GPU | Auto | Speed/VRAM knob for the offload (see below) |
| Steps | 30 | 20–30 is a good range; Nunchaku is fast enough to go higher |
| CFG Scale | 4.0 | Prompt adherence (`true_cfg_scale`); ~2.5–4 works well |
| Number of Outputs | 1 | 1–4 variations per run |
| Seed | 0 | 0 = random each run; fixed = reproducible |

### Blocks on GPU

The transformer offloads itself block-by-block; this sets how many blocks stay resident. **More = faster, but more VRAM.** Auto picks by detected VRAM (≈20 for 12GB, dropping to 1 for ~6GB, and whole-model offload above ~18GB). Lower it if you hit out-of-memory; raise it if you have VRAM to spare. Changing it takes effect on the next model load.

---

## Notes

- First run downloads weights and compiles/loads; later runs in the same session reuse the loaded model. Changing **rank** triggers a reload.
- Want even more speed? Nunchaku also ships **Lightning** (4/8-step) Qwen-Image-Edit variants — quality dips a little but it's several times faster. Ask and I'll wire up a Lightning toggle.
- This is the higher-quality, faster sibling of the GGUF edit extension. Keep the GGUF one as a fallback for non-Ampere / older GPUs where Nunchaku won't run.

---

## License

Extension code is **MIT** — see [LICENSE](LICENSE). The Qwen-Image-Edit weights (Alibaba) and the Nunchaku library + quantized weights (nunchaku-tech) are **Apache 2.0** (commercial use permitted). Independent community extension, not affiliated with Alibaba, Nunchaku, Hugging Face, or Modly; each user is responsible for the model license and for whatever they generate. Provided "as is", without warranty.
