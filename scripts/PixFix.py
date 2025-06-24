# USAGE
# PixFix.py --headless --file .codex/temp/Zpix.png --threshold 42
# PixFix.py --headless --file J:/Zpix.png --threshold 42
import os
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
try:
    import pixfix_rs
    HAVE_RUST = True
except Exception:
    HAVE_RUST = False

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")

def is_image_file(path):
    return os.path.isfile(path) and path.lower().endswith(SUPPORTED_FORMATS)

def smart_blur_flatten(img, threshold=42, passes=2):
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

def color_contrast_mask(img, intensity_thresh=80, color_thresh=90):
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114])
    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=bool)
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        shifted = np.roll(arr, shift=(dy, dx), axis=(0,1))
        shifted_gray = np.dot(shifted[...,:3], [0.299, 0.587, 0.114])
        mask |= (np.abs(gray - shifted_gray) > intensity_thresh)
        dist = np.sqrt(np.sum((arr - shifted) ** 2, axis=2))
        mask |= (dist > color_thresh)
    mask[:1,:] = mask[-1:,:] = mask[:,:1] = mask[:,-1:] = False
    return mask

def find_stairstep_grid(img, max_period=40, edge_thresh=20):
    if HAVE_RUST:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return pixfix_rs.find_stairstep_grid(arr, int(max_period), float(edge_thresh))
    gray = np.array(img.convert("L"), dtype=float)
    edges_x = np.abs(np.diff(gray, axis=1)) > edge_thresh
    edges_y = np.abs(np.diff(gray, axis=0)) > edge_thresh

    def estimate_period(signal, max_p):
        signal = signal - signal.mean()
        f = np.fft.fft(signal, n=2 * len(signal))
        ac = np.fft.ifft(f * np.conjugate(f))[:len(signal)].real
        ac[0] = 0
        return int(np.argmax(ac[1:max_p]) + 1)

    px = estimate_period(edges_x.mean(axis=0), max_period)
    py = estimate_period(edges_y.mean(axis=1), max_period)
    if px <= 0 or py <= 0:
        return None

    grid_w = round(img.width / px)
    grid_h = round(img.height / py)
    if grid_w <= 0 or grid_h <= 0:
        return None
    return (grid_w, grid_h)

def rescale_to_pixel_grid(image, grid_size):
    if grid_size is None:
        return image
    small = image.resize(grid_size, Image.NEAREST)
    return small.resize(image.size, Image.NEAREST)

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

def verify_grid(img, grid, regions=3, controls=2, thresh=0.001):
    if HAVE_RUST:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        ok, pts = pixfix_rs.verify_grid(arr, tuple(grid), int(regions), int(controls), float(thresh))
        return ok, [tuple(p) for p in pts]
    pts = _coherence_regions(img, grid, top_n=regions)
    scores = [_local_step_score(img, p, grid) for p in pts]
    rand_pts = [(np.random.randint(0, img.width), np.random.randint(0, img.height)) for _ in range(controls)]
    rand_scores = [_local_step_score(img, p, grid) for p in rand_pts]
    avg_score = np.mean(scores) if scores else 0.0
    avg_rand = np.mean(rand_scores) if rand_scores else 0.0
    confidence = avg_score - avg_rand
    return confidence > thresh, pts

class PixelArtSmartBlurTool:
    def __init__(self, master):
        self.master = master
        master.title("Pixel Art Smart Blur & Grid Scale Finder")
        master.geometry("960x480")
        master.resizable(True, True)

        self.img_orig = None
        self.img_flat = None
        self.img_pix = None
        self.img_pix_small = None
        self.grid_guess = None
        self.checked_regions = []
        self.manual_scale = False
        self.setting_scale = False
        self.scale_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.DoubleVar(value=42.0)
        self.scale_update_job = None
        self.blur_update_job = None

        self.status_var = tk.StringVar(value="Ready")

        ttk.Button(master, text="Open Image", command=self.open_image).grid(row=0, column=0)
        ttk.Button(master, text="Save Pix Image", command=self.save_image).grid(row=0, column=1)

        self.canvas1 = tk.Canvas(master, width=220, height=220, bg="#2a2a2a")
        self.canvas1.grid(row=1, column=0, padx=8, pady=4)
        self.canvas2 = tk.Canvas(master, width=220, height=220, bg="#2a2a2a")
        self.canvas2.grid(row=1, column=1, padx=8, pady=4)
        self.canvas3 = tk.Canvas(master, width=220, height=220, bg="#2a2a2a")
        self.canvas3.grid(row=1, column=2, padx=8, pady=4)
        self.canvas4 = tk.Canvas(master, width=110, height=110, bg="#333333")
        self.canvas4.grid(row=1, column=3, padx=8, pady=4)

        ttk.Label(master, text="Manual scale override:").grid(row=2, column=0, sticky="e")
        self.scale_slider = ttk.Scale(
            master, from_=0.5, to=5.0, variable=self.scale_var,
            orient="horizontal", command=self.schedule_scale_update
        )
        self.scale_slider.grid(row=2, column=1, sticky="we")

        ttk.Label(master, text="Averaging threshold:").grid(row=3, column=0, sticky="e")
        self.blur_slider = ttk.Scale(
            master, from_=4, to=128, variable=self.blur_var,
            orient="horizontal", command=self.schedule_blur_update
        )
        self.blur_slider.grid(row=3, column=1, sticky="we")
        ttk.Label(master, text="(lower = more aggressive)").grid(row=3, column=2, sticky="w")

        ttk.Label(master, text="Original").grid(row=4, column=0)
        ttk.Label(master, text="Smart Blur").grid(row=4, column=1)
        ttk.Label(master, text="Pixel Grid Preview").grid(row=4, column=2)
        ttk.Label(master, text="Grid Size").grid(row=4, column=3)

        self.grid_label = ttk.Label(master, text="")
        self.grid_label.grid(row=5, column=0, columnspan=4)
        self.status_bar = ttk.Label(master, textvariable=self.status_var, anchor="w")
        self.status_bar.grid(row=6, column=0, columnspan=4, sticky="we")
        master.grid_columnconfigure(1, weight=1)

    def set_status(self, text):
        self.status_var.set(text)
        self.master.update_idletasks()

    def _set_scale(self, value):
        self.setting_scale = True
        self.scale_var.set(value)
        self.setting_scale = False

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff")]
        )
        if not path:
            return
        try:
            self.set_status("Loading image...")
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load image: {e}")
            return
        self.img_orig = img
        self.update_blur_and_grid()
    def update_blur_and_grid(self):
        if self.img_orig is None:
            return
        self.set_status("Analyzing...")
        threshold = self.blur_var.get()
        self.img_flat = smart_blur_flatten(self.img_orig, threshold=threshold, passes=2)
        self.grid_guess = find_stairstep_grid(self.img_flat)
        self.checked_regions = []
        if self.grid_guess:
            ok, pts = verify_grid(self.img_orig, self.grid_guess)
            if ok:
                self.checked_regions = pts
            scale_guess = self.img_orig.width / self.grid_guess[0]
            if 0.5 <= scale_guess <= 5.0:
                self.manual_scale = False
                self._set_scale(scale_guess)
        self.update_previews()
        self.set_status("Ready")

    def update_previews(self):
        if self.img_orig is None:
            return
        # Original
        prev1 = self.img_orig.copy().resize((220,220), Image.NEAREST)
        prev1_tk = ImageTk.PhotoImage(prev1)
        self.canvas1.delete("all")
        self.canvas1.create_image(0,0,anchor="nw",image=prev1_tk)
        self.canvas1.image = prev1_tk
        if self.checked_regions:
            sx = 220 / self.img_orig.width
            sy = 220 / self.img_orig.height
            gw, gh = self.grid_guess
            for x,y in self.checked_regions:
                self.canvas1.create_rectangle(
                    x*sx, y*sy, (x+gw)*sx, (y+gh)*sy, outline="red"
                )
        # Smart blur
        prev2 = self.img_flat.copy().resize((220,220), Image.NEAREST)
        prev2_tk = ImageTk.PhotoImage(prev2)
        self.canvas2.create_image(0,0,anchor="nw",image=prev2_tk)
        self.canvas2.image = prev2_tk
        # Pixel grid preview
        if not self.manual_scale and self.grid_guess:
            scale_x = self.img_orig.width / self.grid_guess[0]
            scale_y = self.img_orig.height / self.grid_guess[1]
            scale = (scale_x + scale_y) / 2.0
            gw = max(1, int(round(self.img_orig.width / scale)))
            gh = max(1, int(round(self.img_orig.height / scale)))
        else:
            gw = max(1, int(round(self.img_orig.width / self.scale_var.get())))
            gh = max(1, int(round(self.img_orig.height / self.scale_var.get())))
        pix = self.img_orig.resize((gw,gh), Image.NEAREST)
        pix_big = pix.resize(self.img_orig.size, Image.NEAREST)
        prev3 = pix_big.resize((220,220), Image.NEAREST)
        prev3_tk = ImageTk.PhotoImage(prev3)
        self.canvas3.create_image(0,0,anchor="nw",image=prev3_tk)
        self.canvas3.image = prev3_tk
        prev4 = pix.resize((110,110), Image.NEAREST)
        prev4_tk = ImageTk.PhotoImage(prev4)
        self.canvas4.create_image(0,0,anchor="nw",image=prev4_tk)
        self.canvas4.image = prev4_tk

        self.img_pix = pix_big
        self.img_pix_small = pix
        status = "confirmed" if self.checked_regions else "uncertain"
        self.grid_label.config(text=(
            f"Detected grid: {gw} × {gh} | Scale: {self.scale_var.get():.2f} | "
            f"Blur threshold: {self.blur_var.get():.1f} | {status}"
        ))

    def schedule_scale_update(self, *_):
        if self.setting_scale:
            return
        self.manual_scale = True
        if self.scale_update_job:
            self.master.after_cancel(self.scale_update_job)
        self.scale_update_job = self.master.after(300, self.update_previews)

    def schedule_blur_update(self, *_):
        if self.blur_update_job:
            self.master.after_cancel(self.blur_update_job)
        self.blur_update_job = self.master.after(300, self.update_blur_and_grid)

    def save_image(self):
        if not (self.img_pix and self.img_pix_small):
            messagebox.showwarning("No Image", "No image to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPG","*.jpg"),("BMP","*.bmp"),("All","*.*")]
        )
        if not path:
            return
        self.set_status("Saving...")
        base, ext = os.path.splitext(path)
        self.img_pix_small.save(f"{base}_pix{ext}")
        self.img_pix.save(f"{base}_pix_big{ext}")
        messagebox.showinfo("Saved",
            f"Images saved as:\n{base}_pix{ext}\n{base}_pix_big{ext}"
        )
        self.set_status("Ready")

def run_headless(path, threshold):
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
    parser = argparse.ArgumentParser(description="Pixel art smart blur tool")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--file", type=str, help="Image to process in headless mode")
    parser.add_argument("--threshold", type=float, default=42.0, help="Initial averaging threshold")
    args = parser.parse_args()

    if args.headless and args.file:
        run_headless(args.file, args.threshold)
    else:
        root = tk.Tk()
        app = PixelArtSmartBlurTool(root)
        root.mainloop()

if __name__ == "__main__":
    main()
