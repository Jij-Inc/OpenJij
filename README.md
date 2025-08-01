# OpenJij : Framework for the Ising model and QUBO.

[![PyPI version shields.io](https://img.shields.io/pypi/v/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI implementation](https://img.shields.io/pypi/implementation/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI format](https://img.shields.io/pypi/format/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI license](https://img.shields.io/pypi/l/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![PyPI download month](https://img.shields.io/pypi/dm/openjij.svg)](https://pypi.python.org/pypi/openjij/)
[![Downloads](https://static.pepy.tech/badge/openjij)](https://pepy.tech/project/openjij)

[![CPP Test](https://github.com/Jij-Inc/OpenJij/actions/workflows/ci-test-cpp.yml/badge.svg)](https://github.com/Jij-Inc/OpenJij/actions/workflows/ci-test-cpp.yml)
[![Python Test](https://github.com/Jij-Inc/OpenJij/actions/workflows/ci-test-python.yaml/badge.svg)](https://github.com/Jij-Inc/OpenJij/actions/workflows/ci-test-python.yaml)

[![DOI](https://zenodo.org/badge/164117633.svg)](https://zenodo.org/badge/latestdoi/164117633)

- python >= 3.9
- (optional) gcc >= 7.0.0
- (optional) cmake >= 3.22
- (optional) Ninja

- [OpenJij Website](https://www.openjij.org/)

- [Documents](https://jij-inc.github.io/OpenJij/)

- [C++ Docs](https://jij-inc.github.io/OpenJij-Reference-Page/)

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

OpenJij uses [uv](https://docs.astral.sh/uv/) for efficient dependency management and reproducible development environments.

```sh
# Clone repository
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd OpenJij

# Install uv (choose one method)
$ pip install uv                                 # Recommended
# or: curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: brew install uv                           # Homebrew

# Set up development environment with exact dependency versions
$ uv sync --locked --group dev

# Verify installation (includes C++ extension build)
$ uv run python -c "import openjij; import openjij.cxxjij; print('OpenJij setup complete')"
$ uv run pytest tests/ -v --tb=short
```

### Dependency Groups

OpenJij uses [PEP 735](https://peps.python.org/pep-0735/) dependency groups for efficient quantum computing development:

| Group | Purpose | Command | Use Case |
|-------|---------|---------|----------|
| **dev** | Full development environment | `uv sync --group dev` | Complete setup with all tools |
| **test** | Testing and coverage tools | `uv sync --group test` | CI/CD and automated testing |
| **format** | Code quality and formatting | `uv sync --group format` | Linting and style checks |
| **build** | Build and packaging tools | `uv sync --group build` | Package creation and distribution |

**Dependency management**:
- **Reproducible builds** (CI/CD, team collaboration): `uv sync --locked --group dev`
- **Latest versions** (local development): `uv sync --group dev`
- **Update dependencies**: `uv lock` or `uv lock --upgrade`

All dependencies are locked in `uv.lock` for consistent environments across different systems.

### C++ Extension Integration

OpenJij's C++ extensions are built automatically during installation for optimal quantum computing performance:

```sh
# Development with C++ code modifications
$ uv sync --group dev                           # Initial setup
# ... modify C++ source files ...
$ uv run pip install .  # Rebuild C++ extension
$ uv run python -c "import openjij.cxxjij; print('C++ extension updated')"
```

## Test

### Python Tests

```sh
# Install test dependencies with exact versions
$ uv sync --locked --group test

# Run comprehensive test suite
$ uv run pytest tests/ -v --tb=short
$ uv run pytest tests/ -v --cov=openjij --cov-report=html
$ uv run python -m coverage html
```

### C++ Tests

```sh
# Build and run C++ tests (independent of Python environment)
$ mkdir build 
$ cmake -DCMAKE_BUILD_TYPE=Debug -S . -B build
$ cmake --build build --parallel
$ cd build
$ ./tests/cxxjij_test

# Alternative: Use CTest for comprehensive testing
$ ctest --extra-verbose --parallel --schedule-random
```

**Requirements**: CMake ≥ 3.22, C++17 compatible compiler

## Code Quality

```sh
# Install formatting tools with exact versions
$ uv sync --locked --group format

# Unified ruff-based quality checks
$ uv run ruff check .                        # Lint check
$ uv run ruff format .                       # Format code
$ uv run ruff check . --fix                  # Auto-fix issues

# All-in-one quality verification
$ uv run ruff check . && uv run ruff format --check .
```

## For Contributors

Contributors are welcome! Please follow these guidelines:

1. **Set up development environment**: Use `uv sync --locked --group dev` for consistent setup
2. **Follow code standards**: Run quality checks before submitting
3. **Test thoroughly**: Ensure both Python and C++ tests pass
4. **Document changes**: Update relevant documentation

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
