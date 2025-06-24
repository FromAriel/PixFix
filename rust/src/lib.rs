use pyo3::prelude::*;
use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, IntoPyArray, PyUntypedArrayMethods};
use rand::Rng;

fn edge_map(arr: &Vec<u8>, w: usize, h: usize) -> Vec<f32> {
    let mut edges = vec![0f32; w*h];
    for y in 0..h {
        for x in 0..w {
            let idx = (y*w + x)*3;
            let gray = 0.299*(arr[idx] as f32) + 0.587*(arr[idx+1] as f32) + 0.114*(arr[idx+2] as f32);
            let gx = if x+1 < w {
                let idx2 = (y*w + x+1)*3;
                let g2 = 0.299*(arr[idx2] as f32) + 0.587*(arr[idx2+1] as f32) + 0.114*(arr[idx2+2] as f32);
                (gray - g2).abs()
            } else {0.0};
            let gy = if y+1 < h {
                let idx2 = ((y+1)*w + x)*3;
                let g2 = 0.299*(arr[idx2] as f32) + 0.587*(arr[idx2+1] as f32) + 0.114*(arr[idx2+2] as f32);
                (gray - g2).abs()
            } else {0.0};
            edges[y*w + x] = gx + gy;
        }
    }
    let max_v = edges.iter().cloned().fold(0./0., f32::max);
    if max_v > 0.0 {
        for e in edges.iter_mut() { *e /= max_v; }
    }
    edges
}

#[pyfunction]
fn smart_blur_flatten<'py>(py: Python<'py>, img: PyReadonlyArray3<u8>, threshold: f32, passes: usize) -> PyResult<&'py PyArray3<u8>> {
    let dims = img.shape();
    let (h, w, c) = (dims[0], dims[1], dims[2]);
    if c < 3 { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("image must have 3 channels")); }
    let mut arr: Vec<f32> = img.as_slice()?.iter().map(|&v| v as f32).collect();
    let mut temp = arr.clone();
    for _ in 0..passes {
        temp.copy_from_slice(&arr);
        for y in 1..h-1 {
            for x in 1..w-1 {
                let idx = (y*w + x)*c;
                let center = [arr[idx], arr[idx+1], arr[idx+2]];
                let mut ok = true;
                for (dy, dx) in [(-1i32,0i32),(1,0),(0,-1),(0,1)] {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    let nidx = (ny*w + nx)*c;
                    let neighbor = [arr[nidx], arr[nidx+1], arr[nidx+2]];
                    let dist = ((neighbor[0]-center[0]).powi(2)+(neighbor[1]-center[1]).powi(2)+(neighbor[2]-center[2]).powi(2)).sqrt();
                    if dist >= threshold { ok = false; break; }
                }
                if ok {
                    let mut sum = center;
                    for (dy, dx) in [(-1i32,0i32),(1,0),(0,-1),(0,1)] {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let nidx = (ny*w + nx)*c;
                        sum[0] += arr[nidx]; sum[1] += arr[nidx+1]; sum[2] += arr[nidx+2];
                    }
                    let avg = [sum[0]/5.0,sum[1]/5.0,sum[2]/5.0];
                    let idxs = [idx, idx+1, idx+2];
                    for i in 0..3 { temp[idxs[i]] = avg[i]; }
                    for (dy, dx) in [(-1i32,0i32),(1,0),(0,-1),(0,1)] {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let nidx = (ny*w + nx)*c;
                        for i in 0..3 { temp[nidx+i] = avg[i]; }
                    }
                }
            }
        }
        arr.copy_from_slice(&temp);
    }
    let res: Vec<u8> = arr.iter().map(|&v| v.clamp(0.0,255.0) as u8).collect();
    let arr3 = Array3::from_shape_vec((h, w, c), res).unwrap();
    Ok(arr3.into_pyarray(py))
}

fn from_image_array(img: PyReadonlyArray3<u8>) -> (Vec<u8>, usize, usize, usize) {
    let dims = img.shape();
    let h = dims[0]; let w = dims[1]; let c = dims[2];
    let v = img.as_slice().unwrap().to_vec();
    (v, w, h, c)
}

fn estimate_period(signal: &[f32], max_p: usize) -> usize {
    let len = signal.len();
    let mean = signal.iter().sum::<f32>() / len as f32;
    let centered: Vec<f32> = signal.iter().map(|v| *v - mean).collect();
    let mut best = 0usize;
    let mut best_val = f32::MIN;
    for p in 1..=max_p {
        let mut s = 0f32;
        for i in 0..len - p {
            s += centered[i] * centered[i+p];
        }
        if s > best_val { best_val = s; best = p; }
    }
    best
}

#[pyfunction]
fn find_stairstep_grid(img: PyReadonlyArray3<u8>, max_period: usize, edge_thresh: f32) -> PyResult<Option<(usize, usize)>> {
    let (data, w, h, c) = from_image_array(img);
    let mut gray = vec![0f32; w*h];
    for y in 0..h {
        for x in 0..w {
            let idx = (y*w + x)*c;
            let g = 0.299*(data[idx] as f32) + 0.587*(data[idx+1] as f32) + 0.114*(data[idx+2] as f32);
            gray[y*w + x] = g;
        }
    }
    let mut edges_x = vec![0f32; w*h];
    let mut edges_y = vec![0f32; w*h];
    for y in 0..h {
        for x in 0..w-1 {
            let a = gray[y*w + x];
            let b = gray[y*w + x +1];
            edges_x[y*w + x] = if (a-b).abs() > edge_thresh {1.0} else {0.0};
        }
    }
    for y in 0..h-1 {
        for x in 0..w {
            let a = gray[y*w + x];
            let b = gray[(y+1)*w + x];
            edges_y[y*w + x] = if (a-b).abs() > edge_thresh {1.0} else {0.0};
        }
    }
    let mut mean_x = vec![0f32; w];
    for x in 0..w {
        let mut s=0f32; for y in 0..h { s += edges_x[y*w + x]; }
        mean_x[x] = s / h as f32;
    }
    let mut mean_y = vec![0f32; h];
    for y in 0..h {
        let mut s=0f32; for x in 0..w { s += edges_y[y*w + x]; }
        mean_y[y] = s / w as f32;
    }
    let px = estimate_period(&mean_x, max_period);
    let py = estimate_period(&mean_y, max_period);
    if px == 0 || py == 0 { return Ok(None); }
    let grid_w = ((w as f32 / px as f32).round()) as usize;
    let grid_h = ((h as f32 / py as f32).round()) as usize;
    if grid_w == 0 || grid_h == 0 { return Ok(None); }
    Ok(Some((grid_w, grid_h)))
}

fn local_step_score(edges: &[f32], w: usize, h: usize, pos: (usize, usize), grid: (usize, usize)) -> f32 {
    let (x, y) = pos; let (gw, gh) = grid;
    let x0 = x.saturating_sub(gw); let y0 = y.saturating_sub(gh);
    let x1 = (x+gw).min(w-1); let y1 = (y+gh).min(h-1);
    if x0>=x1 || y0>=y1 { return 0.0; }
    let mut sum_patch = 0f32; let mut count = 0f32;
    let mut sum_lines = 0f32; let mut count_lines = 0f32;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let e = edges[yy*w + xx];
            sum_patch += e; count +=1.0;
            if ((xx+1) % gw == 0) || ((yy+1) % gh == 0) {
                sum_lines += e; count_lines +=1.0;
            }
        }
    }
    if count_lines==0.0 { return 0.0; }
    let step_strength = sum_lines / count_lines;
    let baseline = sum_patch / count;
    step_strength - baseline
}

fn coherence_regions(edges: &[f32], w: usize, h: usize, grid: (usize, usize), top_n: usize) -> Vec<(usize, usize)> {
    let (gw, gh) = grid;
    let mut scores = vec![0f32; w*h];
    for y in 0..h {
        for x in 0..w {
            if x % gw == 0 || y % gh == 0 {
                scores[y*w + x] += edges[y*w + x];
            }
        }
    }
    let mut coords = Vec::new();
    let mut temp = scores.clone();
    for _ in 0..top_n {
        if let Some((idx, _)) = temp.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()) {
            let y = idx / w; let x = idx % w;
            coords.push((x,y));
            let y0 = y.saturating_sub(gh); let x0 = x.saturating_sub(gw);
            let y1 = (y+gh).min(h-1); let x1 = (x+gw).min(w-1);
            for yy in y0..=y1 { for xx in x0..=x1 { temp[yy*w+xx] = f32::NEG_INFINITY; } }
        }
    }
    coords
}

#[pyfunction]
fn verify_grid(img: PyReadonlyArray3<u8>, grid: (usize, usize), regions: usize, controls: usize, thresh: f32) -> PyResult<(bool, Vec<(usize, usize)>)> {
    let (data, w, h, _c) = from_image_array(img);
    let edges = edge_map(&data, w, h);
    let pts = coherence_regions(&edges, w, h, grid, regions);
    let mut rng = rand::thread_rng();
    let mut scores = Vec::new();
    for &(x,y) in &pts { scores.push(local_step_score(&edges, w, h, (x,y), grid)); }
    let mut rand_scores = Vec::new();
    for _ in 0..controls {
        let x = rng.gen_range(0..w);
        let y = rng.gen_range(0..h);
        rand_scores.push(local_step_score(&edges, w, h, (x,y), grid));
    }
    let avg_score = if scores.len()>0 { scores.iter().sum::<f32>()/scores.len() as f32 } else {0.0};
    let avg_rand = if rand_scores.len()>0 { rand_scores.iter().sum::<f32>()/rand_scores.len() as f32 } else {0.0};
    let confidence = avg_score - avg_rand;
    Ok((confidence>thresh, pts))
}

#[pymodule]
fn pixfix_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smart_blur_flatten, m)?)?;
    m.add_function(wrap_pyfunction!(find_stairstep_grid, m)?)?;
    m.add_function(wrap_pyfunction!(verify_grid, m)?)?;
    Ok(())
}

