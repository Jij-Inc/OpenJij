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
[![codecov](https://codecov.io/gh/OpenJij/OpenJij/branch/main/graph/badge.svg?token=WMSK3GS8E5)](https://codecov.io/gh/OpenJij/OpenJij)

[![DOI](https://zenodo.org/badge/164117633.svg)](https://zenodo.org/badge/latestdoi/164117633)

- python >= 3.9
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

For development installation, you will need to install CMake>=3.22.\
We highly recommend installing CMake via PYPI.

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

### Development Install (Recommended for Contributors)

For faster development iteration, use the editable install approach:

```sh
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd OpenJij
$ python -m venv .venv
$ source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
$ pip install -e .[dev] --no-build-isolation
```

This setup allows you to:
- Edit Python code and see changes immediately (no reinstall needed)
- Only rebuild C++ when you modify C++ source files
- Maintain fast development iterations

When you modify C++ code, rebuild with:
```sh
$ pip install -e . --no-build-isolation
```

### Troubleshooting Development Setup

If you encounter issues:

1. **Import errors after editable install**: Make sure C++ extension is built:
   ```sh
   $ pip install -e . --no-build-isolation
   ```

2. **CMake errors**: Ensure you have CMake > 3.22:
   ```sh
   $ pip install -U cmake
   ```

3. **Clean rebuild**: Remove build artifacts and rebuild:
   ```sh
   $ rm -rf _skbuild/ openjij/*.so openjij/include/ openjij/share/
   $ pip install -e . --no-build-isolation
   ```

4. **Check installation**: Verify the setup is working:
   ```sh
   $ python -c "import openjij; print('Success:', dir(openjij))"
   ```

## For Contributor

Please follow the existing code style and run tests before submitting a pull request.

## Test

### Python

```sh
$ python -m venv .venv
$ . .venv/bin/activate
$ pip install -e .[dev] --no-build-isolation
$ export CMAKE_BUILD_TYPE=Debug
$ python -m pytest tests/ -v --tb=short 
$ python -m coverage html
```

### Development with Editable Install

For development, you can build only the C++ extension once and install Python code in editable mode:

```sh
# Install Python code and build C++ extension in editable mode
$ pip install -e . --no-build-isolation

# Now you can edit Python code and changes will be reflected immediately
# To rebuild C++ extension after making C++ changes:
$ pip install -e . --no-build-isolation
```

This approach allows you to:
- Modify Python code without reinstalling
- Only rebuild C++ when necessary
- Faster development iteration

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

- Code Quality Check (Lint & Format)

```sh
# Dependencies are managed in pyproject.toml
$ pip install -e .[dev]
```

```sh
# Unified linting and formatting with ruff
$ python -m ruff check .                     # Lint check
$ python -m ruff format .                    # Format code
$ python -m ruff check . --fix               # Auto-fix issues
$ python -m ruff format . --check --diff     # Check format without changing
```

## Python Documentation 
Use Juyter Book for build documentation.   
With KaTeX    
Need: Graphviz

``` sh
$ pip install -e .[dev]
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
