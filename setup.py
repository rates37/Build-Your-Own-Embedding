from setuptools import setup, find_packages

"""
Run `python setup.py sdist bdist_wheel` to build the package.
"""

README_FILE = "README.md"


def main() -> None:
    with open(README_FILE, "r") as f:
        readme = f.read()

    setup(
        name="BuildYourOwnEmbedding",
        version="1.1",
        packages=find_packages(),
        install_requires=[
            "numpy>=1.24.0",
            "matplotlib>=3.9.1",
            "matplotlib-inline>=0.1.6",
            "typing_extensions>=4.4.0",
            "mplcursors>=0.5.3",
            "scikit-learn>=1.3.0 ",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
    )


if __name__ == "__main__":
    main()
