# Static Warp Bubbles (2025)

This repository contains the implementation of the **Static Spherically-Symmetric Warp Bubble** metric described by Bolívar, Abellán, and Vasilev (2025).

It is built as an extension to [pyWarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory), using it as a core engine for metric handling and Einstein tensor solving.

## Installation

1.  **Install pyWarpFactory**:
    Ensure you have the main `warpfactory` package installed. If you have the source locally:
    ```bash
    cd ../WarpFactory-main
    pip install -e .
    ```

2.  **Install StaticWarpBubbles**:
    ```bash
    cd StaticWarpBubbles
    pip install -e .
    ```

## Structure

-   `static_bubbles/`: Core Python package.
    -   `generator.py`: Generates metric from energy density $\rho(r)$.
    -   `analyzer.py`: Checks Energy Conditions analytically.
    -   `utils.py`: (Optional utility functions).

-   `examples/`: Demo scripts.
    -   `demo.py`: Generates plots of Density, Shift, and ECs.

-   `tests/`: Verification scripts.
    -   `validate.py`: Validates numerical solver against analytic types.

## Usage

Run the demo:
```bash
python examples/demo.py
```
