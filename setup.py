# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import platform

try:
    from skbuild import setup
except ImportError:
    from setuptools import setup

setup_requires = [
    "numpy",
    "pybind11",
    "cmake > 3.20",
    "scikit-build > 0.16.0"
]

if any(arg in sys.argv for arg in ("pytest", "test")):
    setup_requires.append("pytest-runner")

os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

cmake_args = []

if platform.system() == "Darwin":
    cmake_args.append("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
    
    # Handle macOS universal2 builds
    cibw_archs = os.environ.get("CIBW_ARCHS", "")
    cibw_archs_macos = os.environ.get("CIBW_ARCHS_MACOS", "")
    archflags = os.environ.get("ARCHFLAGS", "")
    cibuildwheel = os.environ.get("CIBUILDWHEEL", "")
    _arch = os.environ.get("_PYTHON_HOST_PLATFORM", "")
    
    # Debug print
    print(f"CIBW_ARCHS: {cibw_archs}")
    print(f"CIBW_ARCHS_MACOS: {cibw_archs_macos}")
    print(f"ARCHFLAGS: {archflags}")
    print(f"CIBUILDWHEEL: {cibuildwheel}")
    print(f"_PYTHON_HOST_PLATFORM: {_arch}")
    
    # Check if we're in a cibuildwheel build
    if cibuildwheel == "1":
        # Check both CIBW_ARCHS and CIBW_ARCHS_MACOS
        if "universal2" in cibw_archs or "universal2" in cibw_archs_macos or "universal2" in _arch:
            # Universal2 build: both x86_64 and arm64
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
            print("Setting CMAKE_OSX_ARCHITECTURES=arm64;x86_64 for universal2 build")
        elif archflags:
            # Parse ARCHFLAGS to determine architectures
            archs = []
            if "-arch x86_64" in archflags:
                archs.append("x86_64")
            if "-arch arm64" in archflags:
                archs.append("arm64")
            if archs:
                cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}")
                print(f"Setting CMAKE_OSX_ARCHITECTURES={';'.join(archs)} from ARCHFLAGS")
        else:
            # Fallback: if ARCHFLAGS is set, we should be building universal2
            if archflags == "-arch x86_64 -arch arm64":
                cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
                print("Setting CMAKE_OSX_ARCHITECTURES=arm64;x86_64 from ARCHFLAGS (fallback)")

# Set CMAKE_ARGS environment variable
if cmake_args:
    existing_cmake_args = os.environ.get("CMAKE_ARGS", "")
    if existing_cmake_args:
        os.environ["CMAKE_ARGS"] = f"{existing_cmake_args} {' '.join(cmake_args)}"
    else:
        os.environ["CMAKE_ARGS"] = " ".join(cmake_args)
    print(f"CMAKE_ARGS set to: {os.environ['CMAKE_ARGS']}")

# Pass cmake_args to setup if we're using skbuild
setup_kwargs = {
    "setup_requires": setup_requires,
    "packages": [
        "openjij",
        "openjij.model",
        "openjij.sampler",
        "openjij.utils",
    ],
    "cmake_install_dir": "openjij",
    "include_package_data": False,
    "zip_safe": False,
}

# If using skbuild and we have cmake_args, pass them
if 'skbuild' in sys.modules and cmake_args:
    setup_kwargs["cmake_args"] = cmake_args

setup(**setup_kwargs)
