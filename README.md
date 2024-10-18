# Build-Your-Own-Embedding

BuildYourOwnEmbedding is a Python library designed for generating synthetic neural responses and analysing their resultant embeddings. This library provides tools for modelling a variety of neural response functions, generating sets of synthetic responses, adding noise to the responses, and advanced analysis and evaluation techniques like PCA and RDMs. 


## Installation
To install the library, simply use `pip`:

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

Full documentation is available [here](https://rates37.github.io/build_your_own_embedding/), including examples. You can generate the documentation locally using Sphinx:

```bash
cd docs
make html
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
