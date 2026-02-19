"""
Auto-detect and configure build tools (ninja, MSVC) for gsplat CUDA JIT compilation.

Import this module before using gsplat to ensure the build environment is ready.
"""

import glob
import os
import shutil
import sys


def setup():
    """Add ninja and MSVC cl.exe to PATH if not already available."""
    # 1. Ninja
    try:
        import ninja
        ninja_dir = ninja.BIN_DIR
        if ninja_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ninja_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

    # 2. MSVC cl.exe (Windows only)
    if sys.platform == "win32" and not shutil.which("cl"):
        cl_path = _find_msvc_cl()
        if cl_path:
            cl_dir = os.path.dirname(cl_path)
            os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")


def _find_msvc_cl():
    """Find cl.exe from Visual Studio installations."""
    vs_roots = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Microsoft Visual Studio"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Microsoft Visual Studio"),
    ]
    for vs_root in vs_roots:
        # Search across VS versions (2022, 2019, etc.) and editions
        pattern = os.path.join(vs_root, "*", "*", "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")
        matches = sorted(glob.glob(pattern), reverse=True)  # newest first
        if matches:
            return matches[0]
    return None


setup()
