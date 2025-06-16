# OpenJij : Framework for the Ising model and QUBO.

[![PyPI version shields.io](https://img.shields.io/pypi/v/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI implementation](https://img.shields.io/pypi/implementation/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI format](https://img.shields.io/pypi/format/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI license](https://img.shields.io/pypi/l/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI download month](https://img.shields.io/pypi/dm/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![Downloads](https://static.pepy.tech/badge/openjij)](https://pepy.tech/project/openjij)

[![CPP Test](https://github.com/OpenJij/OpenJij/actions/workflows/ci-test-cpp.yml/badge.svg)](https://github.com/OpenJij/OpenJij/actions/workflows/ci-test-cpp.yml)
[![Python Test](https://github.com/OpenJij/OpenJij/actions/workflows/ci-test-python.yaml/badge.svg)](https://github.com/OpenJij/OpenJij/actions/workflows/ci-test-python.yaml)
[![Build Documentation](https://github.com/OpenJij/OpenJij/actions/workflows/buid-doc.yml/badge.svg)](https://github.com/OpenJij/OpenJij/actions/workflows/buid-doc.yml)
[![CodeQL](https://github.com/OpenJij/OpenJij/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/OpenJij/OpenJij/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0204475dc07d48ffa851480d03db759e)](https://www.codacy.com/gh/OpenJij/OpenJij/dashboard?utm_source=github.com&utm_medium=referral&utm_content=OpenJij/OpenJij&utm_campaign=Badge_Grade)
[![Maintainability](https://api.codeclimate.com/v1/badges/3b2f43f3e601ae74c497/maintainability)](https://codeclimate.com/github/OpenJij/OpenJij/maintainability)
[![codecov](https://codecov.io/gh/OpenJij/OpenJij/branch/main/graph/badge.svg?token=WMSK3GS8E5)](https://codecov.io/gh/OpenJij/OpenJij)

## Coverage Graph

| **Sunburst**                                                                                                                                                         | **Grid**                                                                                                                                                         | **Icicle**                                                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <a href="https://codecov.io/gh/OpenJij/OpenJij"><img src="https://codecov.io/gh/OpenJij/OpenJij/branch/main/graphs/sunburst.svg?token=WMSK3GS8E5" width="100%"/></a> | <a href="https://codecov.io/gh/OpenJij/OpenJij"><img src="https://codecov.io/gh/OpenJij/OpenJij/branch/main/graphs/tree.svg?token=WMSK3GS8E5" width="100%"/></a> | <a href="https://codecov.io/gh/OpenJij/OpenJij"><img src="https://codecov.io/gh/OpenJij/OpenJij/branch/main/graphs/icicle.svg?token=WMSK3GS8E5" width="100%"/></a> |

- python >= 3.8
- (optional) gcc >= 7.0.0
- (optional) cmake >= 3.22
- (optional) Ninja

[OpenJij Website](https://www.openjij.org/)

### Change **IMPORT**

- OpenJij >= v0.5.0

  ```python
  import openjij.cxxjij
  ```

- OpenJij <= v0.4.9

  ```python
  import cxxjij
  ```

- [Documents](https://jij-inc.github.io/OpenJij/)

- [C++ Docs](https://openjij.github.io/OpenJij-Reference-Page/index.html)

## install

### install via pip

> Note: (2023/08/09) GPGPU algorithms will no longer be supported.

```
# Binary
$ pip install openjij 
# From Source
$ pip install --no-binary=openjij openjij
```

### install via pip from source codes

To install OpenJij from source codes, please install CMake first then install OpenJij.

#### cmake setup

If you want to use setup.py instead of PIP, You will need to install CMake>=3.22.\
We are Highly recommended install CMake via PYPI.

```
$ pip install -U cmake
```

Make sure the enviroment path for CMake is set correctly.

#### install OpenJij

```
$ pip install --no-binary=openjij openjij
```

### install from github repository

```
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd openjij
$ python -m pip install -vvv .
```

### Development Installation (Editable Mode)

For developers who want to make changes to the OpenJij source code, you can install OpenJij in editable mode. This allows you to modify the source code and see the changes without reinstalling the package.

#### Method 1: Direct editable install (Recommended)

```bash
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd OpenJij
$ pip install -e . -v
```

This will:
1. Build the C++ extensions using CMake automatically
2. Install the package in editable mode
3. Allow you to modify Python code and see changes immediately
4. Require rebuilding only when C++ code is modified

#### Method 2: Manual C++ build followed by editable install

If you need more control over the build process or want to build C++ components separately:

```bash
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd OpenJij

# First, build C++ components manually
$ mkdir build
$ cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
$ cmake --build build --parallel

# Then install in editable mode
$ pip install -e . -v --no-build-isolation
```

#### Method 3: Development build with debug symbols

For debugging purposes, you can build with debug symbols:

```bash
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd OpenJij

# Set environment variable for debug build
$ export CMAKE_BUILD_TYPE=Debug

# Install in editable mode with debug build
$ pip install -e . -v
```

#### Rebuilding after C++ changes

When you modify C++ source files, you need to rebuild the extensions:

```bash
# Option 1: Reinstall (rebuilds everything)
$ pip install -e . -v --force-reinstall --no-deps

# Option 2: Manual rebuild and reinstall
$ rm -rf build/
$ pip install -e . -v --no-build-isolation
```

**Note**: Python-only changes (in `openjij/*.py` files) don't require rebuilding, but C++ changes (in `include/`, `openjij/main.cpp`, etc.) require rebuilding the extensions.

## For Contributor

### Development Setup

For the best development experience, use editable installation mode:

```bash
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd OpenJij
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e . -v
```

### Pre-commit Setup

Use `pre-commit` for auto check before git commit.
`.pre-commit-config.yaml`

```
# pipx install pre-commit 
# or 
# pip install pre-commit
pre-commit install
```

### Development Workflow

1. Make changes to Python code - changes are immediately available (no rebuild needed)
2. Make changes to C++ code - rebuild required:
   ```bash
   $ pip install -e . -v --force-reinstall --no-deps
   ```
3. Run tests to verify changes:
   ```bash
   $ python -m pytest tests/
   ```

## Test

### Python

#### Using editable install (Recommended for development)

```sh
$ python -m venv .venv
$ . .venv/bin/activate
$ pip install pip-tools 
$ pip-compile setup.cfg
$ pip-compile dev-requirements.in
$ pip-sync requirements.txt dev-requirements.txt

# Install in editable mode with debug build for testing
$ export CMAKE_BUILD_TYPE=Debug
$ pip install -e . -v

# Run tests
$ python -m pytest tests/
$ python -m coverage html
```

#### Using setup.py (Legacy method)

```sh
$ python -m venv .venv
$ . .venv/bin/activate
$ pip install pip-tools 
$ pip-compile setup.cfg
$ pip-compile dev-requirements.in
$ pip-sync requirements.txt dev-requirements.txt
$ source .venv/bin/activate
$ export CMAKE_BUILD_TYPE=Debug
$ python setup.py --force-cmake install --build-type Debug -G Ninja
$ python setup.py --build-type Debug test 
$ python -m coverage html
```

### C++

```sh
$ mkdir build 
$ cmake -DCMAKE_BUILD_TYPE=Debug -S . -B build
$ cmake --build build --parallel
$ cd build
$ ./tests/cxxjij_test
# Alternatively  Use CTest 
$ ctest --extra-verbose --parallel --schedule-random
```

Needs: CMake > 3.22, C++17

- Format

```sh
$ pip-compile format-requirements.in
$ pip-sync format-requirements.txt
```

```sh
$ python -m isort 
$ python -m black 
```

- Aggressive Format

```sh
$ python -m isort --force-single-line-imports --verbose ./openjij
$ python -m autoflake --in-place --recursive --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables ./openjij
$ python -m autopep8 --in-place --aggressive --aggressive  --recursive ./openjij
$ python -m isort ./openjij
$ python -m black ./openjij
```

- Lint

```sh
$ pip-compile setup.cfg
$ pip-compile dev-requirements.in
$ pip-compile lint-requirements.in
$ pip-sync requirements.txt dev-requirements.txt lint-requirements.txt
```

```sh
$ python -m flake8
$ python -m mypy
$ python -m pyright
```

## Python Documentation 
Use Juyter Book for build documentation.   
With KaTeX    
Need: Graphviz

``` sh
$ pip-compile setup.cfg
$ pip-compile build-requirements.in
$ pip-compile doc-requirements.in
$ pip-sync requirements.txt build-requirements.txt doc-requirements.txt
```

Please place your document to `docs/tutorial`either markdown or jupyter notebook style.

```sh
$ pip install -vvv .
```

```sh 
$ jupyter-book build docs --all
```


## How to use

### Python example

```python
import openjij as oj
sampler = oj.SASampler()
response = sampler.sample_ising(h={0: -1}, J={(0,1): -1})
response.states
# [[1,1]]

# with indices
response = sampler.sample_ising(h={'a': -1}, J={('a','b'): 1})
[{index: s for index, s in zip(response.indices, state)} for state in response.states]
# [{'b': -1, 'a': 1}]
```

## Community

- [OpenJij Discord Community](https://discord.gg/Km5dKF9JjG)

## About us

This product is maintained by Jij Inc.

**Please visit our website for more information!**
https://www.j-ij.com/

### Licences

Copyright 2023 Jij Inc.

Licensed under the Apache License, Version 2.0 (the "License");\
you may not use this file except in compliance with the License.\
You may obtain a copy of the License at

```
 http://www.apache.org/licenses/LICENSE-2.0  
```

Unless required by applicable law or agreed to in writing, software\
distributed under the License is distributed on an "AS IS" BASIS,\
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\
See the License for the specific language governing permissions and\
limitations under the License.
