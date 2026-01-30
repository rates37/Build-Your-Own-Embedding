# Build-Your-Own-Embedding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BuildYourOwnEmbedding is a Python library designed for generating synthetic neural responses and analysing their resultant embeddings. This library provides tools for modelling a variety of neural response functions, generating sets of synthetic responses, adding noise to the responses, and advanced analysis and evaluation techniques like PCA and RDMs.

## Installation

### Using `uv` (recommended):

[uv](https://docs.astral.sh/uv/) is a fast Python package manager, install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install this library:

```bash
git clone https://github.com/rates37/Build-Your-Own-Embedding.git
cd Build-Your-Own-Embedding
uv sync
```

To include development dependencies (pytest, ruff, mypy, etc.):

```bash
uv sync --extra dev
```

To include documentation dependencies (Sphinx):

```bash
uv sync --extra docs
```

### Using pip

You can also install with pip:

```bash
pip install BuildYourOwnEmbedding
```

Alternatively, clone the repository and install the package:

```bash
git clone https://github.com/rates37/Build-Your-Own-Embedding.git
cd BuildYourOwnEmbedding
pip install .
```

## Getting Started

### Generating Neural Responses

The following example demonstrates how to create a custom Gaussian response function:

```py
import numpy as np
from BuildYourOwnEmbedding import responses, parameters

# Define input stimulus
x = np.linspace(0, 1, 100)

# Define response parameters
params = {
    "mean": parameters.ConstantParameter(0.5),
    "std": parameters.ConstantParameter(0.1)
}

# Create a Gaussian response manager
responseManager = responses.ResponseManager(responses.GaussianResponse, **params)

# Generate a neural response with no noise
neural_response = responseManager.generate_responses(x, noiseLevel=0)
```

## Documentation

Full documentation is available [here](https://rates37.github.io/build_your_own_embedding/), including examples.

### Building documentation locally:

```bash
uv sync --extra docs
cd docs
uv run make html
```

Or with pip:

```bash
cd docs
make html
```

## Development:

### Running Tests:

```bash
uv run pytest tests/ -v
```

### Type checking:

```bash
uv run ruff check .
uv run mypy BuildYourOwnEmbedding/
```

## Contributing

This project welcomes contributions! To contribute to this project:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Added Feature X'`).
4. Push your branch (`git push origin feature-branch`).
5. Create a pull request.

Please ensure that all new features are covered appropriately with tests and documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
