#!/usr/bin/env python3
"""
Quick environment verification for the kspace-sampling-prototype.
Checks for required Python packages and prints installation hints.
"""
from __future__ import annotations

import importlib
import sys

RECOMMENDED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("nibabel", "nibabel"),
    ("skimage", "scikit-image"),
]

OPTIONAL = [
    ("mrinufft", "mri-nufft[finufft]  # requires finufft native library"),
]


def check(pkgs):
    ok = True
    for mod, pkg in pkgs:
        try:
            importlib.import_module(mod)
            print(f"OK: {mod}")
        except Exception:
            print(f"MISSING: {mod}  -> pip install {pkg}")
            ok = False
    return ok


def main():
    print("Checking required packages...")
    req_ok = check(RECOMMENDED)

    print("\nChecking optional packages for NUFFT support...")
    opt_ok = check(OPTIONAL)

    if req_ok and opt_ok:
        print("\nEnvironment looks good for full functionality.")
        sys.exit(0)
    elif req_ok and not opt_ok:
        print("\nCore dependencies are present, but optional NUFFT support is missing.")
        print("If you only want to run summary scripts that read CSV/PNG, you may proceed.")
        sys.exit(0)
    else:
        print("\nSome required packages are missing. Install them with:")
        print("  python -m pip install -r requirements.txt")
        sys.exit(2)


if __name__ == '__main__':
    main()
