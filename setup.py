from setuptools import setup, find_packages

setup(
    name="pywarpfactory",
    version="1.0.0",
    description="pyWarpFactory: a Python-first toolkit for analyzing warp drive spacetimes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nelson",
    url="https://github.com/xys004/pyWarpFactory",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "google-cloud-storage>=2.16.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)
