from setuptools import setup, find_packages

setup(
    name="igv",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "networkx>=3.0",
    ],
    python_requires=">=3.10",
)
