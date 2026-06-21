import os
import random
import sys
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

from services.generators.base import BaseGenerator, smooth_progress

# keep stdout clean for the runner protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

# Enable HF's fast multi-connection downloader if available — the big single
# files here (text encoder shards, nunchaku transformer) are much faster with
# it on. Silently no-ops if hf_transfer isn't installed.
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

# Base pipeline components (text encoder / vae / tokenizer / scheduler). Public,
# Apache-2.0, ungated. The original 40GB transformer in this repo is skipped on
# download — Nunchaku supplies the (much smaller) quantized transformer instead.
BASE_REPO = "Qwen/Qwen-Image-Edit"

# Nunchaku SVDQuant 4-bit transformer (single safetensors). Public, Apache-2.0.
NUNCHAKU_REPO = "nunchaku-tech/nunchaku-qwen-image-edit"
# filename pattern: svdq-{int4|fp4}_r{32|128}-qwen-image-edit.safetensors
# int4 = most GPUs, fp4 = Blackwell / RTX 50-series (auto-detected)


def _int(val, default):
    try:
        return int(val)
    except:
        return default


def _float(val, default):
    try:
        return float(val)
    except:
        return default


class QwenNunchakuEditGenerator(BaseGenerator):
    MODEL_ID     = "qwen_image_edit_nunchaku"
    DISPLAY_NAME = "Qwen-Image-Edit (Nunchaku) Image Edit"
    VRAM_GB      = 6

    # ----------------------------------------------------------- file resolution
    def _precision(self):
        p = getattr(self, "_prec", None)
        if p:
            return p
        try:
            from nunchaku.utils import get_precision
            p = get_precision()  # "int4" (most GPUs) or "fp4" (Blackwell)
        except Exception:
            p = "int4"
        self._prec = p
        return p

    def _nunchaku_filename(self):
        rank = getattr(self, "_rank", None) or 128
        return "svdq-%s_r%d-qwen-image-edit.safetensors" % (self._precision(), rank)

    def is_downloaded(self):
        base_ok = (self.model_dir / "model_index.json").exists()
        tok_ok = (self.model_dir / "tokenizer" / "merges.txt").exists()
        nun_ok = (self.model_dir / self._nunchaku_filename()).exists()
        return base_ok and tok_ok and nun_ok

    # ------------------------------------------------------------------ loading
    def load(self):
        rank = getattr(self, "_rank", None) or 128
        state = (rank, self._precision())

        if self._model is not None and getattr(self, "_loaded_state", None) == state:
            return
        if self._model is not None:
            self.unload()

        if not self.is_downloaded():
            self._download_weights()

        import torch
        from diffusers import QwenImageEditPipeline
        try:
            from nunchaku import NunchakuQwenImageTransformer2DModel
            from nunchaku.utils import get_gpu_memory
        except Exception as e:
            raise RuntimeError(
                "Nunchaku isn't installed in this extension's venv (%s). If setup didn't "
                "auto-install it, see the 'Manual Nunchaku install' section in the README." % e
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device != "cuda":
            raise RuntimeError("Nunchaku requires an NVIDIA CUDA GPU (Ampere / RTX 30-series or newer); none detected.")

        fn = self._nunchaku_filename()
        print("[Qwen-NK] loading transformer %s" % fn)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(str(self.model_dir / fn))

        pipe = QwenImageEditPipeline.from_pretrained(
            str(self.model_dir),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        try:
            vram = float(get_gpu_memory())
        except Exception:
            vram = 0.0

        nb = self._num_blocks(vram)
        if vram > 18 and nb <= 0:
            pipe.enable_model_cpu_offload()
            print("[Qwen-NK] model CPU offload (vram ~%.0f GB)" % vram)
        else:
            blocks = max(1, nb)
            # Nunchaku does its own async per-block offload for the transformer, so the
            # transformer must be EXCLUDED from diffusers' sequential offload (which only
            # streams the text encoder + vae here).
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=blocks)
            try:
                pipe._exclude_from_cpu_offload.append("transformer")
            except Exception:
                pass
            pipe.enable_sequential_cpu_offload()
            print("[Qwen-NK] per-block offload, num_blocks_on_gpu=%d (vram ~%.0f GB)" % (blocks, vram))

        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._model = pipe
        self._transformer = transformer
        self._loaded_state = state
        print("[Qwen-NK] ready on %s" % self._device)

    def _num_blocks(self, vram):
        nb = getattr(self, "_blocks_param", None)
        if nb not in (None, "", "auto"):
            try:
                return int(nb)
            except:
                pass
        # auto by VRAM. 0 = signal to use whole-model offload (big GPUs).
        if vram >= 18:
            return 0
        if vram >= 12:
            return 20
        if vram >= 8:
            return 8
        if vram >= 6:
            return 3
        return 1

    def unload(self):
        self._model = None
        self._transformer = None
        self._device = None
        self._loaded_state = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------------- generation
    def _to_pil(self, image_in):
        from PIL import Image
        if image_in is None:
            raise ValueError("no input image — connect an Image node to the Edit node")
        if isinstance(image_in, Image.Image):
            return image_in.convert("RGB")
        if isinstance(image_in, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(image_in))).convert("RGB")
        if hasattr(image_in, "__fspath__") or isinstance(image_in, str):
            return Image.open(image_in).convert("RGB")
        if hasattr(image_in, "read"):
            return Image.open(image_in).convert("RGB")
        raise ValueError("unrecognized image input type: %r" % type(image_in))

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}
        self._rank = _int(params.get("rank"), 128)
        self._blocks_param = params.get("num_blocks_on_gpu") or "auto"

        state = (self._rank, self._precision())
        if self._model is None or getattr(self, "_loaded_state", None) != state:
            self.load()

        prompt = ""
        for _k in ("prompt", "text", "instruction", "negative_prompt"):
            _v = params.get(_k)
            if _v and str(_v).strip():
                prompt = str(_v).strip()
                break
        if not prompt:
            print("[Qwen-NK] no prompt found; params keys received: %r" % (sorted(params.keys()),))
            raise ValueError("no edit instruction — connect a Text node to the Edit node's text input")

        image = self._to_pil(image_bytes)

        steps     = _int(params.get("steps"), 30)
        cfg       = _float(params.get("true_cfg_scale"), 4.0)
        n_images  = max(1, min(_int(params.get("num_images"), 1), 4))
        base_seed = _int(params.get("seed"), 0)

        self._report(progress_cb, 5, "starting up")
        self._check_cancelled(cancel_event)

        if self.outputs_dir:
            out_dir = self.outputs_dir
        else:
            out_dir = self.model_dir.parent.parent.parent / "outputs" / self.MODEL_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i in range(n_images):
            self._check_cancelled(cancel_event)

            if base_seed == 0:
                seed = random.randint(1, 2**31 - 1)
            else:
                seed = base_seed + i
            gen = torch.Generator(device="cpu").manual_seed(seed)

            lo = 10 + int(85 * (i / float(n_images)))
            hi = 10 + int(85 * ((i + 1) / float(n_images)))
            label = "editing" if n_images == 1 else "editing %d/%d" % (i + 1, n_images)
            self._report(progress_cb, lo, label)

            stop = threading.Event()
            ticker = None
            if progress_cb:
                ticker = threading.Thread(
                    target=smooth_progress,
                    args=(progress_cb, lo, hi, label, stop),
                    daemon=True,
                )
                ticker.start()

            try:
                with torch.inference_mode():
                    result = self._model(
                        image=image,
                        prompt=prompt,
                        negative_prompt=" ",
                        true_cfg_scale=cfg,
                        num_inference_steps=steps,
                        generator=gen,
                    )
                out_img = result.images[0]
            finally:
                stop.set()
                if ticker:
                    ticker.join(timeout=1.0)

            filename = "qwen_nk_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
            out_path = out_dir / filename
            out_img.save(str(out_path), format="PNG")
            paths.append(str(out_path))
            print("[Qwen-NK] saved %s (seed %d)" % (out_path, seed))

        self._report(progress_cb, 100, "done")

        # single output -> path string (same contract as the t2i node, so the existing
        # Preview node works unchanged); multiple -> list of paths.
        if len(paths) == 1:
            return paths[0]
        return paths

    # ----------------------------------------------------------------- download
    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download, hf_hub_download

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # base components, minus the original 40GB transformer (Nunchaku replaces it)
        print("[Qwen-NK] downloading base components from %s" % BASE_REPO)
        snapshot_download(
            repo_id=BASE_REPO,
            local_dir=str(self.model_dir),
            ignore_patterns=["transformer/*"],
        )

        # nunchaku quantized transformer (single safetensors)
        fn = self._nunchaku_filename()
        print("[Qwen-NK] downloading %s from %s" % (fn, NUNCHAKU_REPO))
        hf_hub_download(
            repo_id=NUNCHAKU_REPO,
            filename=fn,
            local_dir=str(self.model_dir),
        )

        print("[Qwen-NK] download complete")
