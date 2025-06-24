Review the following plan and lay out a set of tasks or sub tasks to attempt to implement every possible feature below we can manage to do.



---

### PIX-FIX 2.0 – Master-Plan Preamble

Welcome, Codex.  Our goal is to evolve **pix-fix** into a *tiered, self-tuning pixel-art scale-detection engine* capable of processing thousands of mixed-quality images with minimum manual oversight.  You’ll be asked to turn a long catalogue of candidate techniques (ultra-light pre-checks, fast heuristics, mid-tier refinements, heavy brute-force verifiers, exotic plugins, GPU paths, meta-learning, etc.) into a coherent, maintainable code-base.

#### What you are building

* A **pipeline runner** that executes multiple detection passes (ULTRA-LIGHT → PASS-0 → PASS-1 → PASS-2) using cached artefacts and vote fusion.
* A **plugin architecture** so new detectors (FFT comb filter, Popescu-Farid, CNN, Hough grid…) can be dropped in without touching the core loop.
* **Adaptive orchestration**: early exits on high-confidence wins, fall-backs when ambiguity remains, and parameter auto-search when confidence stalls.
* **Rich logging & metrics**: per-image JSON, aggregated CSV, optional debug overlays for visual inspection, plus unit / regression tests for every detector.

#### Over-arching constraints

* **Modularity first**: each detector lives in its own module with a common `DetectResult` interface (`scale`, `confidence`, `extras`).
* **Stateless core, cached artefacts**: expensive intermediates (edge maps, spectra, windows, ROIs) are stored once and reused by later passes.
* **Config-driven**: all thresholds, window sizes, and enable/disable flags live in a single `pixfix.toml` (or YAML) so CI can sweep them.
* **Graceful degradation**: if a Rust/GPU helper or CNN model is missing, fallback to the pure-Python path and flag slower mode in logs.
* **Determinism & reproducibility**: random crops / LHC sampling seeded by image hash + run ID; identical inputs yield identical outputs.

#### Deliverables & success metrics

* Code, docs, and unit tests land under `pixfix/` with type hints and docstrings.
* A CLI: `pixfix scan <folder> [--heavy --gpu]`.
* **Baseline accuracy**: pass `Zpix.png` and ≥ 80 % of the provided AI set with correct integer scale.
* **Performance budget** (CPU-only reference machine):

  * Ultra-light pre-pass ≤ 1 ms @ 4K
  * Fast pass ≤ 40 ms @ 4K
  * Mid-tier ≤ 300 ms @ 4K (when invoked)
  * Heavy pass capped at 2 s **per undecided image** (ROI-based).

#### How you should respond next

After this preamble you will receive a *hierarchical task list* (not included here).
Break that list down, implement, test, and iterate until the pipeline meets the above goals.

Good luck – let’s make pixel-scale detection bullet-proof!

ULTRA-LIGHT PRE-PASS  (≈ 1 ms on 4 K)
──────────────────────────────────────
• Palette cardinality check
    - Count unique colours in a 64×64 down-sample
    - If   |palette|  <  80   ⇒   very likely crisp pixel-art → skip heavy passes
    - If   |palette|  > 256   ⇒   probably painted / AI ⇒ go through normal pipeline

• EXIF / PNG chunk sniff
    - Look for iTXt “sourceXres”, “pixel_density”, or a Game-Boy-style “TILESIZE” tag
    - Usually absent, but when present it’s a free win → hard-override scale.



First-Pass Checklist  ────────────────────────────────────────────────────────
  INPUT PREP
    [ ]  Load frame → uint8 RGB or RGBA
    [ ]  Optional ½-scale shrink (bilinear) just for analysis
    [ ]  Global contrast boost
          • fast gamma ≈ 1.8  OR
          • CLAHE (8×8 tiles, clip=2.0)
    [ ]  smart_blur_flatten (1 pass, thr=40) to merge near-identical pixels
    [ ]  Convert to gray 8-bit; keep RGB copy for later

  EDGE & HISTOGRAM FEATURES
    [ ]  Binary edge map X (|dI/dx| > 20)         →  1-D histogram Hx
    [ ]  Binary edge map Y (|dI/dy| > 20)         →  1-D histogram Hy
    [ ]  Edge-density (edges / total)  → quick “is there enough texture?” gate
    [ ]  Orientation histogram (0°,90° buckets from Sobel) – spike check

  PERIOD / SCALE CANDIDATES
    [ ]  Autocorrelation of Hx  → top-3 peaks Px
    [ ]  Autocorrelation of Hy  → top-3 peaks Py
    [ ]  Fast FFT of Hx & Hy (real-to-complex, size 1024) – confirm peaks
    [ ]  Candidate set  P  =  union(Px,Py)  clipped to 2 … 16
    [ ]  Harmonic collapse  (if 4 & 8 present, keep 4, etc.)
    [ ]  Initial vote table  {scale → score_x+score_y}

  COARSE TILE SWEEP (LEVEL-0)
    [ ]  Split frame into 4 quads (½×½)
    [ ]  For each quad:
          • repeat edge → autocorr → top-1 scale + per-tile confidence
          • update vote table
    [ ]  Early-exit test:
          if  (top-vote / total votes ≥ 0.70)  AND
              (mean confidence ≥ τ0)               → accept scale & stop

  CACHED DATA FOR LATER PASSES
    [ ]  Save:  edge maps, Hx/Hy spectra, orientation hist, vote table
    [ ]  Save:  list of “ambiguous” quads (confidence < τ0 or disagreeing)
    [ ]  Emit:  quick JSON blob  
          {
            "fast_scale_guess": N or null,
            "vote_table": {N1:v1, N2:v2, …},
            "edge_density": e,
            "needs_refine": true/false,
            "tiles": [ {bbox, scale, conf}, … ]
          }

Mid-Tier Checklist  ──────────────────────────────────────────────────────────
  LOAD FAST-PASS CACHE
    [ ]  edge maps, spectra, vote_table, ambiguous_tiles ← fast_pass.json
    [ ]  If   needs_refine == false          → return cached scale immediately
    [ ]  Restrict candidate scale set  S  to top-K votes (K ≤ 4)

  SUB-TILE SLIDING WINDOW (LEVEL-1)
    [ ]  Window size  w = 128×128  (clamp to ≥3×max(S))
    [ ]  Stride       s = w // 4   (75 % overlap)
    [ ]  For each window:
          • reuse global edge maps (slice)
          • 1-D autocorr → local scale guess
          • keep windows whose top guess ∈ S and conf ≥ τ1
    [ ]  Build heat-map   H(x,y) = count of matches per scale
    [ ]  If any scale has  >M windows (M≈8) AND coverage >5 % of frame
          → accept that scale  (store offset of densest window)

  FREQUENCY PEAK REFINEMENT
    [ ]  2-D FFT of **full** edge magnitude (≃ 35 ms @4 K with numpy-FFT)
    [ ]  Radial slice analysis:
          • detect comb-like peaks every N pixels along kx=0 & ky=0 axes
          • candidate set  S_freq
    [ ]  Intersect S_freq with S ; boost vote_table[scale] += w_freq
    [ ]  If vote_table top-scale now ≥ 0.8 * total votes → accept

  POPESCU-FARID RESIDUE TEST
    [ ]  Compute 1st-order linear prediction error image  R
    [ ]  2-D FFT on R; locate periodic peaks
    [ ]  Peaks that align with 1/N grids  → add +w_pf to vote_table
    [ ]  Update acceptance criterion as above

  HARMONIC DISAMBIGUATION / GCD
    [ ]  If axes disagree (e.g. 4 vs 6)   → scale = gcd(axes) (verify below)
    [ ]  If 2×N appears stronger than N   → attempt to down-vote harmonic

  CROP-AND-VERIFY LOOP
    [ ]  For top-2 remaining scales:
          • crop 3–5 windows with best heat-map response
          • _verify_grid_py (or Rust) on each crop
          • accumulate pass count & average confidence
    [ ]  Choose scale with   pass_rate ≥ 66 %   AND   mean_conf ≥ τ2
    [ ]  If none pass  → flag for Pass 2 (heavy)

  OUTPUT / CACHE FOR NEXT PASS
    [ ]  dump  mid_pass.json:
          {
            "mid_scale_guess": N or null,
            "vote_table": {...},
            "heatmap_summary": {scale:[win_cnt,coverage]},
            "needs_heavy": true/false,
            "debug": {fft_peaks, pf_peaks}
          }

Heavy-Pass Checklist  ────────────────────────────────────────────────────────
  INPUT
    [ ]  mid_pass.json  →  candidate_scales   S  (≤8)       
    [ ]  roi_list       →  {x,y,w,h,scale_votes}

  1. BRUTE CYCLE-CONSISTENCY SWEEP
     For each ROI and each     s ∈ S :
       • I_down  = nearest_downsample( ROI , factor = s )
       • I_up    = nearest_upscale  ( I_down , factor = s )
       • score   = PSNR( ROI , I_up , region_mask=edge_band(2 px) )
     Keep  best_s per ROI ;   accumulate global histogram   H_cycle
     Note: PSNR band-mask focuses on stair edges → more discriminative.

  2. FULL TEMPLATE X-CORRELATION (STEERABLE)
     For each ROI, each s :
       • build ideal  s×s  “+” and “L” step kernels  (binary, anti-aliased)
       • conv2d   ROI_edges ⨂ kernel    (FFT-based, so O(N log N))
       • take max response → template_score[s]
     Boost vote_table[ s ] += w_template if response > τ_template

  3. SPECTRAL COMB FILTER (“frequency-of-frequency”)
     Per ROI:
       • G = |FFT( ROI_grad_mag )|
       • Gx0 = column 0 of G   ;   Gy0 = row 0
       • P  = |FFT( Gx0 )| + |FFT( Gy0 )|
       • Detect highest peak → period  p   ;   s_freq = round(p)
     If  s_freq ∈ S    → vote_table[ s_freq ] += w_comb

  4. MORPHO-SKELETON BOX-COUNTING
     Per ROI:
       • Binarise ROI via Otsu on luminance
       • skel = morphological_skeleton( B )
       • Run connected-component on axis-aligned fragments → length_list
       • Dominant length mode L  ≈ scale factor
     Adds weak votes; good when AI anti-aliases edges into faint lines.

  5. CNN   “PatchScale-Net”  (Optional, CPU-heavy)
     • Input: 64×64 RGB patch  →  logits over {1…16}
     • Averaged over 128 random crops inside ROI
     • Add to vote_table with weight proportional to softmax prob.
     Side-note: on CPU ~45 ms per run, so we keep crop count low;
                on GPU (torch-cuda) this step is virtually free.

  6. HOUGH GRID REFINEMENT
     • For top-2 scales after votes:
         – Run probabilistic Hough on ROI_edges
         – Cluster lines into vertical/horizontal groups
         – If mean spacing ≈ s±1 px → confidence += w_hough
     • Reject if spacing inconsistent (alias / perspective warp).

  7. FINAL AGGREGATION & CONFIDENCE
     For each scale s:
       total_vote[s] =   α·H_cycle[s]
                      +  β·template_votes[s]
                      +  γ·spectral_votes[s]
                      +  δ·cnn_votes[s]
                      +  ε·hough_conf[s]
     Choose   s*  = argmax total_vote
     Accept if          total_vote[s*]  ≥ κ
               AND      total_vote[s*]  ≥ 1.2 × next_best

  8. OUTPUT
     heavy_pass.json  ← {
       "final_scale"    : s*      or null if still undecided,
       "vote_breakdown" : {cycle:…, template:…, comb:…, cnn:…, hough:…},
       "roi_winners"    : [{roi_id,scale,psnr,template,maxvote}, …],
       "debug"          : {peak_periods, rejected_harmonics},
       "needs_manual"   : true/false
     }

ULTRA-LIGHT PRE-PASS  (≈ 1 ms on 4 K)
──────────────────────────────────────
• Palette cardinality check
    - Count unique colours in a 64×64 down-sample
    - If   |palette|  <  80   ⇒   very likely crisp pixel-art → skip heavy passes
    - If   |palette|  > 256   ⇒   probably painted / AI ⇒ go through normal pipeline

• EXIF / PNG chunk sniff
    - Look for iTXt “sourceXres”, “pixel_density”, or a Game-Boy-style “TILESIZE” tag
    - Usually absent, but when present it’s a free win → hard-override scale.


TEMPORAL / NEIGHBOUR HINTING (for batch or video)
─────────────────────────────────────────────────
• Keep a  LRU cache  {sha1(image_block) → scale}
  - If a new frame’s centre tile hash matches an earlier frame, reuse its scale.
  - Fantastic for sprite-sheets or GIF frames; zero extra compute.

• Motion-based ROI reuse
  - For successive video frames, propagate ROIs along optical-flow vectors
    and skip Pass-0/1 if the patch hasn’t changed.


AUTO-PARAM SEARCH (“Self-Tune” Daemon)
──────────────────────────────────────
• When fast-pass fails, fire a background thread:
    - Latin-Hypercube sample  (edge_thresh, blur_thr, CLAHE_clip, …)
    - Evaluate on a 128×128 seed tile for 1–2 seconds
    - Store param set that yields the sharpest autocorr peak ↣ reuse next run
  Trade-off:  100 % offline; bounded cost; slowly adapts to new asset style.


ROTATION-AWARE GRID  (rare but nasty)
──────────────────────────────────────
• Hough-angle sweep  (θ = ­5°…+5°, step 0.5°)
    - Rotate ROI by −θ
    - Run quick 1-D autocorr
    - If one angle yields  *strong*  periodicity, record (scale, θ)
  Use when user drops SNES screenshots captured with slight keystone warp.



COMPRESSIBILITY HEURISTIC
─────────────────────────
• For each candidate scale  s :
    − D  =  ROI  –  nearest_up( nearest_down( ROI, s ), s )
    − gzip_len(D)   ~ information left *after* down↔up cycle
  Pick  s  that minimises  gzip_len.  
  Slower than PSNR but robust to colour-grading and noise (entropy is king).


STATISTICAL ENSEMBLE / META-LEARNER
───────────────────────────────────
Input = feature vector per image:
    [fast_vote_ratio,
     edge_density,
     peak_strength_x,
     peak_strength_y,
     best_psnr_cycle,
     gzip_entropy_gain,
     cnn_confidence,…]

Train a tiny Gradient-Boosted Tree (XGBoost-depth-3, ~5 kB model).
   → outputs   P(scale=k | features)

You end up with a single, calibrated probability that can drive:
   if  P ≥ 0.92  → auto-accept
   elif 0.7 ≤ P < 0.92 → push to Heavy pass
   else → mark “needs manual”.



GPU / SIMD ACCEL PLUG-INS
─────────────────────────
• cupy.fft  drop-in for numpy FFTs (1-line swap) – 10-20× speed-up
• Numba @vectorize for edge-threshold kernels
• VulkanFFT via PyTorch + torch.fft  if CUDA isn’t available
How they slot in

             ┌──────────┐     cache-hit   ┌─────────────┐
Input  ───►  │ UL pre   │──yes──────────►│  RETURN     │
             │          │                └─────────────┘
             └─────no───┘
                    │
                    ▼
            ( your Pass-0 / Pass-1 / Pass-2 … )
                    │
          still ambiguous?
                    ▼
         ┌──────────────────┐
         │ Exotic add-ons   │  (Rotation-aware, Compressibility,
         │  + Meta-learner) │   Auto-param, etc.)
         └──────────────────┘
Why bother?
These plug-ins either give you “free wins” on corners cases (metadata, hash
reuse) or provide a cheap guard rail that prevents the heavy artillery
from firing when it’s clearly pointless.

When to skip?
For a one-off CLI tool you can ignore most of them.
For bulk asset QA or nightly CI on thousands of sprites, they pay for
themselves in wall-clock time saved.

Feel free to cherry-pick; none of these are mandatory, but they’re proven
tricks when a detector has to survive in the wild.
