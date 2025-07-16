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

if platform.system() == "Darwin":
    cmake_args = ["-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
    
    # Handle macOS universal2 builds
    cibw_archs = os.environ.get("CIBW_ARCHS", "")
    archflags = os.environ.get("ARCHFLAGS", "")
    
    if cibw_archs == "universal2" or "universal2" in cibw_archs:
        # Universal2 build: both x86_64 and arm64
        cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
    elif archflags:
        # Parse ARCHFLAGS to determine architectures
        archs = []
        if "-arch x86_64" in archflags:
            archs.append("x86_64")
        if "-arch arm64" in archflags:
            archs.append("arm64")
        if archs:
            cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}")
    
    # Set CMAKE_ARGS environment variable
    existing_cmake_args = os.environ.get("CMAKE_ARGS", "")
    if existing_cmake_args:
        os.environ["CMAKE_ARGS"] = f"{existing_cmake_args} {' '.join(cmake_args)}"
    else:
        os.environ["CMAKE_ARGS"] = " ".join(cmake_args)

setup(
    setup_requires=setup_requires,
    packages=[
        "openjij",
        "openjij.model",
        "openjij.sampler",
        "openjij.utils",
    ],
    cmake_install_dir="openjij",
    include_package_data=False,
    zip_safe=False,
)
