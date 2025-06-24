import os
import argparse
from PIL import Image
import numpy as np
import pixfix_rs


def smart_blur_flatten(img, threshold=42.0, passes=1):
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    res = pixfix_rs.smart_blur_flatten(arr, float(threshold), int(passes))
    return Image.fromarray(np.array(res, dtype=np.uint8))


def find_stairstep_grid(img, max_period=40, edge_thresh=20.0):
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return pixfix_rs.find_stairstep_grid(arr, int(max_period), float(edge_thresh))


def verify_grid(img, grid, regions=3, controls=2, thresh=0.001):
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    ok, pts = pixfix_rs.verify_grid(arr, tuple(grid), int(regions), int(controls), float(thresh))
    return ok, [tuple(p) for p in pts]


def process_image(path, threshold=42.0):
    print("Loading image...")
    img = Image.open(path).convert("RGBA")
    print("Blurring for analysis...")
    flat = smart_blur_flatten(img, threshold=threshold, passes=1)
    print("Detecting grid...")
    grid = find_stairstep_grid(flat)
    if grid:
        print(f"Grid candidate: {grid}")
        ok, _ = verify_grid(img, grid, thresh=0.001)
        scale_x = img.width / grid[0]
        scale_y = img.height / grid[1]
        scale = (scale_x + scale_y) / 2.0
        target = (round(img.width / scale), round(img.height / scale))
        print(f"Resizing to {target} using scale {scale:.2f}...")
        small = img.resize(target, Image.NEAREST)
        out = os.path.splitext(path)[0] + "_pix.png"
        small.save(out)
        print(f"Verification: {'ok' if ok else 'low confidence'}")
        print(f"Saved resized image to {out}")
    else:
        print("Grid detection failed with threshold", threshold)


def main():
    parser = argparse.ArgumentParser(description="Pixel art scale detection")
    parser.add_argument("file", help="Image file to process")
    parser.add_argument("--threshold", type=float, default=42.0, help="Averaging threshold")
    args = parser.parse_args()
    process_image(args.file, args.threshold)


if __name__ == "__main__":
    main()
