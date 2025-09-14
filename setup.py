from setuptools import setup, find_packages

setup(
    name="supply-chain-delay",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
    ],
)
