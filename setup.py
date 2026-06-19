import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

IS_WIN = platform.system() == "Windows"
PLAT = "win_amd64" if IS_WIN else "linux_x86_64"

# We pin torch 2.8 so the Nunchaku wheel's torch tag matches exactly.
TORCH_TAG = "torch2.8"
TORCH_PKGS = ["torch==2.8.0", "torchvision==0.23.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Fallback Nunchaku versions to try (GitHub keeps old release assets, so these
# URLs stay valid even after newer releases). The latest is looked up first.
FALLBACK_VERSIONS = ["1.3.0", "1.0.1"]


def venv_python(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def pip(venv, *args, check=True):
    return subprocess.run([str(venv_python(venv)), "-m", "pip"] + list(args), check=check)


def py_tag(venv):
    out = subprocess.check_output(
        [str(venv_python(venv)), "-c", "import sys;print('cp%d%d' % sys.version_info[:2])"]
    )
    return out.decode().strip()


def nunchaku_candidates(py):
    suffix = "+%s-%s-%s-%s.whl" % (TORCH_TAG, py, py, PLAT)
    urls = []
    # 1) newest matching wheel from the GitHub releases API
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/nunchaku-tech/nunchaku/releases?per_page=15",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "modly-setup"},
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            for rel in json.load(r):
                for a in rel.get("assets", []):
                    n = a.get("name", "")
                    if n.startswith("nunchaku-") and n.endswith(suffix):
                        urls.append(a["browser_download_url"])
    except Exception as e:
        print("[nunchaku] release lookup failed (%s); using fallback URLs" % e)
    # 2) hardcoded fallbacks (GitHub + HuggingFace mirrors)
    for ver in FALLBACK_VERSIONS:
        urls.append("https://github.com/nunchaku-tech/nunchaku/releases/download/v%s/nunchaku-%s%s" % (ver, ver, suffix))
        urls.append("https://huggingface.co/nunchaku-tech/nunchaku/resolve/main/nunchaku-%s%s" % (ver, suffix))
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def install_nunchaku(venv):
    py = py_tag(venv)
    for url in nunchaku_candidates(py):
        print("[nunchaku] trying %s" % url)
        if pip(venv, "install", url, check=False).returncode == 0:
            ok = subprocess.run([str(venv_python(venv)), "-c", "import nunchaku"], check=False)
            if ok.returncode == 0:
                print("[nunchaku] installed OK")
                return True, py
    return False, py


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"
    if not venv.exists():
        print("creating venv...")
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("venv exists, skipping creation")

    pip(venv, "install", "--upgrade", "pip", "wheel", "setuptools")

    print("installing torch 2.8 (cu128)...")
    pip(venv, "install", *TORCH_PKGS, "--index-url", TORCH_INDEX)

    print("installing dependencies...")
    pip(venv, "install",
        "diffusers>=0.36.0",
        "transformers>=4.53.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.24.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "numpy",
    )

    print("installing nunchaku wheel...")
    ok, py = install_nunchaku(venv)
    if not ok:
        print("=" * 64)
        print("[nunchaku] AUTO-INSTALL FAILED — install it manually into this venv:")
        print("  \"%s\" -m pip install <wheel-url>" % venv_python(venv))
        print("Pick the wheel matching %s + Python %s + %s from:" % (TORCH_TAG, py, PLAT))
        print("  https://github.com/nunchaku-tech/nunchaku/releases")
        print("  https://huggingface.co/nunchaku-tech/nunchaku/tree/main")
        print("(See the README 'Manual Nunchaku install' section.)")
        print("=" * 64)

    print("done")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2:
        a = json.loads(sys.argv[1])
        setup(Path(a["python_exe"]), Path(a["ext_dir"]), int(a["gpu_sm"]))
    else:
        print("usage: setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
