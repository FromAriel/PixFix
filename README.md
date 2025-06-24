# PixFix

PixFix is a collection of utilities for detecting and correcting the pixel scale of artwork.
It consists of a small Rust library exposed to Python and several helper scripts.

## Components

- **rust/** – `pixfix_rs` Rust crate exposing optimized image analysis
  functions to Python via `pyo3`.
- **scripts/** – command line and Tk GUI tools written in Python.
- **images/** – sample pixel art used for testing.
- **logs/** – output from running the scale detector.

## Building the Rust Extension

The Rust crate can be built with [`maturin`](https://github.com/PyO3/maturin):

```bash
maturin build --release -C PixFix/rust
pip install PixFix/rust/target/wheels/*.whl
```

This will produce a Python wheel that exposes the `pixfix_rs` module.

## CLI Usage

After installing the wheel you can run the command line helper:

```bash
python PixFix/scripts/pixfix_cli.py <image> --threshold 42
```

The script prints the detected grid size and writes a resized image with
`_pix` appended to the filename.

## GUI Tool

`PixFix.py` in the `scripts/` directory launches a simple Tk window for manual
experimentation. It can also be run in headless mode:

```bash
python PixFix/scripts/PixFix.py --headless --file <image>
```

## Detecting Scale in Bulk

`detect_scale.py` scans a directory of images, saves resized versions and logs
results to `logs/pixscale_results.json`.

```bash
python PixFix/scripts/detect_scale.py path/to/images
```

## Changelog

See `pixscale_changelog.txt` for a short history of updates.
