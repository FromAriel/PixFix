import os
import json
import argparse
from PIL import Image
import numpy as np
try:
    import pixfix_rs
    HAVE_RUST = True
except Exception:
    pixfix_rs = None
    HAVE_RUST = False

USE_RUST_FIND = False

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")

def _estimate_period(signal, max_p):
    signal = signal - signal.mean()
    f = np.fft.fft(signal, n=2 * len(signal))
    ac = np.fft.ifft(f * np.conjugate(f))[:len(signal)].real
    ac[0] = 0
    return int(np.argmax(ac[1:max_p]) + 1)

def _find_stairstep_grid_py(img, max_period=40, edge_thresh=20):
    gray = np.array(img.convert("L"), dtype=float)
    edges_x = np.abs(np.diff(gray, axis=1)) > edge_thresh
    edges_y = np.abs(np.diff(gray, axis=0)) > edge_thresh
    px = _estimate_period(edges_x.mean(axis=0), max_period)
    py = _estimate_period(edges_y.mean(axis=1), max_period)
    if px <= 0 or py <= 0:
        return None
    grid_w = round(img.width / px)
    grid_h = round(img.height / py)
    if grid_w <= 0 or grid_h <= 0:
        return None
    return (grid_w, grid_h)

def _edge_map(img):
    gray = np.array(img.convert("L"), dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    dx = np.pad(dx, ((0, 0), (1, 0)), mode="constant")
    dy = np.abs(np.diff(gray, axis=0))
    dy = np.pad(dy, ((1, 0), (0, 0)), mode="constant")
    edges = dx + dy
    if edges.max() > 0:
        edges /= edges.max()
    return edges

def _coherence_regions(img, grid, top_n=3):
    edges = _edge_map(img)
    h, w = edges.shape
    grid_w, grid_h = grid
    filt = np.zeros_like(edges)
    filt[::grid_h, :] = 1.0
    filt[:, ::grid_w] += 1.0
    F_edges = np.fft.fft2(edges)
    F_filt = np.fft.fft2(filt)
    coherence = np.fft.ifft2(F_edges * np.conj(F_filt)).real
    coords = []
    temp = coherence.copy()
    for _ in range(top_n):
        idx = np.argmax(temp)
        y, x = np.unravel_index(idx, temp.shape)
        coords.append((x, y))
        y0 = max(0, y - grid_h)
        x0 = max(0, x - grid_w)
        y1 = min(h, y + grid_h)
        x1 = min(w, x + grid_w)
        temp[y0:y1, x0:x1] = -np.inf
    return coords

def _local_step_score(img, pos, grid):
    edges = _edge_map(img)
    x, y = pos
    grid_w, grid_h = grid
    x0 = max(0, x - grid_w)
    y0 = max(0, y - grid_h)
    x1 = min(edges.shape[1], x + grid_w)
    y1 = min(edges.shape[0], y + grid_h)
    patch = edges[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    v = patch[:, grid_w-1::grid_w]
    h_line = patch[grid_h-1::grid_h, :]
    if v.size == 0 or h_line.size == 0:
        return 0.0
    step_strength = (v.mean() + h_line.mean()) / 2.0
    baseline = patch.mean()
    return step_strength - baseline

def _verify_grid_py(img, grid, regions=3, controls=2, thresh=0.001):
    pts = _coherence_regions(img, grid, top_n=regions)
    scores = [_local_step_score(img, p, grid) for p in pts]
    rand_pts = [(np.random.randint(0, img.width), np.random.randint(0, img.height)) for _ in range(controls)]
    rand_scores = [_local_step_score(img, p, grid) for p in rand_pts]
    avg_score = np.mean(scores) if scores else 0.0
    avg_rand = np.mean(rand_scores) if rand_scores else 0.0
    confidence = avg_score - avg_rand
    return confidence > thresh, pts

def smart_blur_flatten(img, threshold=42.0, passes=1):
    if HAVE_RUST:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        res = pixfix_rs.smart_blur_flatten(arr, float(threshold), int(passes))
        return Image.fromarray(np.array(res, dtype=np.uint8))
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    for _ in range(passes):
        new_arr = arr.copy()
        for y in range(1, h-1):
            for x in range(1, w-1):
                center = arr[y, x]
                neighbors = np.array([
                    arr[y-1, x], arr[y+1, x], arr[y, x-1], arr[y, x+1]
                ])
                if np.all(np.linalg.norm(neighbors - center, axis=1) < threshold):
                    avg = np.mean(np.vstack((center, neighbors)), axis=0)
                    new_arr[y, x] = avg
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        new_arr[y+dy, x+dx] = avg
        arr = new_arr
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def detect_image_scale(img, blur_threshold=42.0):
    flat = smart_blur_flatten(img, threshold=blur_threshold, passes=1)
    if HAVE_RUST and USE_RUST_FIND:
        grid = pixfix_rs.find_stairstep_grid(np.array(flat.convert("RGB"), dtype=np.uint8), 40, 20.0)
    else:
        grid = _find_stairstep_grid_py(flat)
    if not grid:
        return None
    if HAVE_RUST:
        ok, _ = pixfix_rs.verify_grid(np.array(img.convert("RGB"), dtype=np.uint8), tuple(grid), 3, 2, 0.001)
    else:
        ok, _ = _verify_grid_py(img, grid)
    scale_x = img.width / grid[0]
    scale_y = img.height / grid[1]
    scale = (scale_x + scale_y) / 2.0
    return int(round(scale)), ok

def process_directory(directory, blur_threshold=42.0, log_path="logs/pixscale_results.json"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    prev = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            prev = json.load(f)
    results = {}
    improved = 0
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(SUPPORTED_FORMATS):
            continue
        path = os.path.join(directory, name)
        try:
            img = Image.open(path).convert("RGBA")
        except Exception:
            results[name] = {"scale": None}
            continue
        scale = detect_image_scale(img, blur_threshold)
        if scale:
            s, ok = scale
            results[name] = {"scale": s, "verified": bool(ok)}
            if s > 1:
                out_small = os.path.splitext(path)[0] + f"_{s}x.png"
                target = (img.width // s, img.height // s)
                Image.Image.resize(img, target, Image.NEAREST).save(out_small)
            if name not in prev or not prev[name].get("scale"):
                improved += 1
        else:
            results[name] = {"scale": None}
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Newly detected images this run: {improved}")
    for name, info in results.items():
        print(f"{name}: {info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect pixel art scale in images")
    parser.add_argument("directory", nargs="?", default=".codex/temp", help="Directory to scan")
    parser.add_argument("--threshold", type=float, default=42.0, help="Blur threshold")
    args = parser.parse_args()
    process_directory(args.directory, args.threshold)
