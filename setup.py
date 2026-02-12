from setuptools import setup, find_packages

setup(
    name="pywarpfactory",
    version="1.0.0",
    description="WarpFactory: A Python toolkit for analyzing warp drive spacetimes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nelson (Ported from Original MATLAB Team)",
    url="https://github.com/NerdsWithAttitudes/WarpFactory", # link to original repo for now
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)
