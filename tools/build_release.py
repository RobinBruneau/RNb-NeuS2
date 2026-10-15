#!/usr/bin/env python3
"""Build the testbed and package the Meshroom plugin as a release archive.

One script for both Linux and Windows. Run by the linux / windows jobs in
.github/workflows/plugin-release.yml with these environment variables:

    VERSION     release version                    e.g. 2.0.0
    CUDA        CUDA version installed on runner   e.g. 12.4.1
    CUDA_ARCH   SM list (';' / ',' / space)        e.g. 7.5;8.0;8.6;8.9
    OUTPUT_DIR  directory to leave the finished archive in
    PYTHON      Python version of the bundled venv/  e.g. 3.11

Steps: build a headless testbed (no GUI, no OptiX), stage a ready-to-use plugin
folder, bundle the CUDA runtime libraries next to the binary, check that the
binary resolves them, create the plugin venv/ with uv, and archive the folder
into ``OUTPUT_DIR``.

Pass ``--skip-build`` to package an already built ``build/testbed``.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
PLUGIN_NAME = "RNb-NeuS2"
IS_WINDOWS = os.name == "nt"
TESTBED_NAME = "testbed.exe" if IS_WINDOWS else "testbed"

# The testbed looks for configs/ and utils/ next to its parent directory, and
# the Meshroom node imports rnb_neus2 from two levels above itself, so the
# repository layout is kept as is.
PLUGIN_DIRS = ["configs", "utils", "rnb_neus2", "meshroom"]
PLUGIN_FILES = ["run_pipeline.py", "pyproject.toml", "setup.py", "README.md", "LICENSE.txt"]

# Shared libraries copied next to the testbed on Linux (matched on the SONAME).
LINUX_BUNDLED_PREFIXES = ("libcublas", "libcudart", "libgomp")
# Provided by the NVIDIA driver, never shipped and absent on GPU-less runners.
LINUX_DRIVER_LIBS = {"libcuda.so.1"}


def run(cmd, **kwargs):
    """subprocess.run with the command echoed to the log first, and check=True."""
    print("+ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def require_env(name):
    """Fetch an env var or fail fast with a clear message (vs. a KeyError later)."""
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"missing required env var {name}")
    return value


def build(cuda_arch):
    """Configure and build the headless testbed into build/."""
    arches = [a for a in re.split(r"[;,\s]+", cuda_arch.strip()) if a]
    build_env = dict(os.environ)
    # Runners have no GPU to detect, so the target architectures must be given.
    build_env["TCNN_CUDA_ARCHITECTURES"] = ";".join(a.replace(".", "") for a in arches)

    configure = [
        "cmake", "-S", REPO_ROOT, "-B", BUILD_DIR,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DNGP_BUILD_WITH_GUI=OFF",
        "-DNGP_BUILD_WITH_OPTIX=OFF",
        "-DNGP_DEPLOY=ON",
    ]
    if IS_WINDOWS:
        # Ninja + the MSVC environment exported by the workflow: no need for
        # the CUDA MSBuild integration, and cl is not shadowed by MinGW.
        configure += ["-G", "Ninja", "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
    else:
        # DT_RPATH (not DT_RUNPATH) so that the bundled libcublas also finds
        # the bundled libcublasLt next to the testbed.
        configure += [
            "-DCMAKE_BUILD_RPATH=$ORIGIN",
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags",
        ]

    subprocess.run(["nvcc", "--version"], env=build_env, check=False)
    run(configure, env=build_env)
    run(["cmake", "--build", BUILD_DIR, "--config", "Release",
         "--parallel", os.cpu_count() or 2], env=build_env)


def stage_plugin(stage_root):
    """Copy the plugin files and the testbed into a fresh plugin folder."""
    testbed = BUILD_DIR / TESTBED_NAME
    if not testbed.is_file():
        sys.exit(f"testbed not found: {testbed}")

    plugin_dir = stage_root / PLUGIN_NAME
    for name in PLUGIN_DIRS:
        shutil.copytree(REPO_ROOT / name, plugin_dir / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in PLUGIN_FILES:
        shutil.copy2(REPO_ROOT / name, plugin_dir / name)

    (plugin_dir / "build").mkdir()
    shutil.copy2(testbed, plugin_dir / "build" / TESTBED_NAME)
    return plugin_dir


def ldd(binary, extra_lib_dirs=()):
    """Return {soname: resolved path or None} for a Linux binary."""
    env = dict(os.environ)
    if extra_lib_dirs:
        paths = [str(d) for d in extra_lib_dirs] + [env.get("LD_LIBRARY_PATH", "")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in paths if p)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    out = run(["ldd", binary], env=env, capture_output=True, text=True).stdout
    print(out, flush=True)

    libs = {}
    for line in out.splitlines():
        match = re.match(r"\s*(\S+) => (not found|\S+)", line)
        if match:
            soname, path = match.groups()
            libs[soname] = None if path == "not found" else Path(path)
    return libs


def cuda_root():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        sys.exit("nvcc not found on PATH")
    return Path(nvcc).resolve().parent.parent


def bundle_linux(bin_dir):
    """Copy the CUDA/OpenMP runtime next to the testbed and check it resolves."""
    testbed = bin_dir / TESTBED_NAME
    cuda_lib_dir = cuda_root() / "lib64"

    bundled = []
    for soname, path in ldd(BUILD_DIR / TESTBED_NAME, [cuda_lib_dir]).items():
        if not soname.startswith(LINUX_BUNDLED_PREFIXES):
            continue
        if path is None:
            sys.exit(f"cannot bundle {soname}: not found")
        # Copy the real file under its SONAME, which is what the loader asks for.
        shutil.copy2(path.resolve(), bin_dir / soname)
        bundled.append(soname)
    if not any(name.startswith("libcublas") for name in bundled):
        sys.exit("testbed does not link to cuBLAS, nothing was bundled")

    # Without LD_LIBRARY_PATH, everything but the driver library must resolve,
    # and the bundled libraries must be the ones picked up.
    resolved = ldd(testbed)
    missing = {name for name, path in resolved.items() if path is None}
    if missing - LINUX_DRIVER_LIBS:
        sys.exit(f"unresolved libraries: {sorted(missing - LINUX_DRIVER_LIBS)}")
    for soname in bundled:
        if resolved.get(soname) is None or resolved[soname].resolve().parent != bin_dir.resolve():
            sys.exit(f"{soname} is not resolved from {bin_dir}: {resolved.get(soname)}")

    # Smoke test. The toolkit only ships an unversioned libcuda.so stub, so
    # expose it under the SONAME the driver would provide.
    with tempfile.TemporaryDirectory() as stub_dir:
        os.symlink(cuda_lib_dir / "stubs" / "libcuda.so", Path(stub_dir) / "libcuda.so.1")
        run([testbed, "--version"], env=dict(os.environ, LD_LIBRARY_PATH=stub_dir))


def bundle_windows(bin_dir):
    """Copy the cuBLAS and MSVC runtime DLLs next to the testbed."""
    cuda_bin = cuda_root() / "bin"
    dlls = []
    # CUDA 13 moved the DLLs to bin/x64.
    for directory in (cuda_bin, cuda_bin / "x64"):
        for pattern in ("cublas64_*.dll", "cublasLt64_*.dll"):
            dlls += sorted(directory.glob(pattern))
    if not dlls:
        sys.exit(f"no cuBLAS DLLs found under {cuda_bin}")

    redist_dir = os.environ.get("VCToolsRedistDir", "").strip()
    if not redist_dir:
        sys.exit("VCToolsRedistDir is not set, run from an MSVC environment")
    redist_dlls = []
    for component in ("CRT", "OpenMP"):
        redist_dlls += sorted(Path(redist_dir).glob(f"x64/Microsoft.VC*.{component}/*.dll"))
    if not redist_dlls:
        sys.exit(f"no MSVC runtime DLLs found under {redist_dir}")

    for dll in dlls + redist_dlls:
        print(f"bundling {dll}", flush=True)
        shutil.copy2(dll, bin_dir / dll.name)

    # The testbed cannot be run here (nvcuda.dll comes with the NVIDIA driver),
    # so only log what it depends on.
    subprocess.run(["dumpbin", "/dependents", str(bin_dir / TESTBED_NAME)], check=False)


def build_venv(plugin_dir, python):
    """Create the plugin venv/ with the Python dependencies of rnb_neus2."""
    venv = plugin_dir / "venv"
    # A uv-managed (standalone) Python, so any version can be picked on any runner.
    run(["uv", "venv", "--relocatable", "--python-preference", "only-managed",
         "--python", python, venv])
    # Only the dependencies of pyproject.toml: the node imports rnb_neus2 from the plugin root.
    run(["uv", "pip", "install", "--python", venv, "--link-mode", "copy",
         "-r", REPO_ROOT / "pyproject.toml"])

    venv_python = venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    run([venv_python, "-c", "import numpy, cv2, trimesh, scipy, embreex"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true",
                        help="Package the existing build/ instead of building.")
    args = parser.parse_args()

    version = require_env("VERSION")
    cuda = require_env("CUDA")
    output_dir = Path(require_env("OUTPUT_DIR")).resolve()
    python = require_env("PYTHON")

    if not args.skip_build:
        build(require_env("CUDA_ARCH"))

    cuda_flat = "cu" + "".join(cuda.split(".")[:2])  # 12.4.1 -> cu124
    python_flat = "py" + "".join(python.split(".")[:2])  # 3.11 -> py311
    platform_tag = "windows-x86_64" if IS_WINDOWS else "linux-x86_64"
    archive_base = output_dir / f"{PLUGIN_NAME}-{version}-{platform_tag}-{cuda_flat}-{python_flat}"

    with tempfile.TemporaryDirectory() as stage_root:
        plugin_dir = stage_plugin(Path(stage_root).resolve())
        if IS_WINDOWS:
            bundle_windows(plugin_dir / "build")
        else:
            bundle_linux(plugin_dir / "build")
        build_venv(plugin_dir, python)

        output_dir.mkdir(parents=True, exist_ok=True)
        archive = shutil.make_archive(str(archive_base), "zip" if IS_WINDOWS else "gztar",
                                      root_dir=stage_root, base_dir=PLUGIN_NAME)

    print(f"::notice::built {Path(archive).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
