# 3D Gaussian Splatting Implementation Plan

## Project Overview
A from-scratch implementation of 3D Gaussian Splatting for novel view synthesis, following the original paper's methodology.

---

## ✅ Currently Implemented Features

### Core Infrastructure
- [x] **Gaussian Representation** ([core/gaussian.py](core/gaussian.py))
  - nn.Module-based implementation with learnable parameters
  - Positions, scales (log-space), quaternion rotations, opacity logits, RGB colors
  - Covariance matrix computation from scales and rotations
  - Proper activation functions (exp for scales, sigmoid for opacities)

- [x] **Camera Model** ([core/camera.py](core/camera.py))
  - Intrinsic parameters (fx, fy, cx, cy)
  - Extrinsic parameters (c2w, w2c matrices)
  - NeRF-format dataset loading support
  - Device-aware buffer management

- [x] **Scene Management** ([core/scene.py](core/scene.py))
  - Scene container with cameras and Gaussians
  - Random Gaussian initialization
  - Dataset loading from NeRF format

### Rendering Pipeline
- [x] **3D to 2D Projection** ([rendering/projection.py](rendering/projection.py))
  - World-to-camera transformation
  - Perspective projection with intrinsic matrix
  - Covariance projection using Jacobian
  - View frustum culling (z < -0.01, in-bounds checking)

- [x] **Tile-Based Rasterization** ([rendering/rasterizer.py](rendering/rasterizer.py))
  - 16×16 tile subdivision for memory efficiency
  - Gaussian-to-tile assignment via bounding boxes
  - Batch processing (100 tiles per batch)
  - Vectorized pixel rendering within tiles
  - Front-to-back alpha compositing
  - Empty tile handling

- [x] **Optimizations**
  - Vectorized tile assignment (no nested Python loops)
  - Chunked eigenvalue computation (10k Gaussians per chunk to avoid CUDA limits)
  - Padding-based batching for variable Gaussian counts per tile

### Training Infrastructure
- [x] **Loss Functions** ([training/loss.py](training/loss.py))
  - L1 photometric loss
  - D-SSIM loss (using pytorch-msssim)
  - Combined loss: `L = 0.8 × L1 + 0.2 × (1 - SSIM)`

- [x] **Optimizer** ([training/optimizer.py](training/optimizer.py))
  - Adam optimizer with per-parameter learning rates:
    - Positions: 0.00016
    - Quaternions: 0.001
    - Scales: 0.005
    - Opacities: 0.05
    - Colors: 0.0025

- [x] **Learning Rate Scheduling** ([training/scheduler.py](training/scheduler.py))
  - Exponential decay for positions: `lr = initial_lr × 0.01^(t/max_steps)`
  - Constant learning rates for other parameters

- [x] **Training Loop** ([training/trainer.py](training/trainer.py))
  - Full forward-backward pass
  - Gradient clipping (max_norm=1.0)
  - Progress tracking with tqdm
  - Loss and metric logging

- [x] **Adaptive Densification** ([training/densification.py](training/densification.py))
  - Gradient tracking for position parameters
  - Splitting: high gradient + large scale → 2 smaller Gaussians (1.6× scale reduction)
  - Cloning: high gradient + small scale → duplicate Gaussian
  - Runs every 100 iterations from iter 500 to 15,000
  - Optimizer state management (Adam momentum extension/pruning)

- [x] **Pruning** ([training/pruning.py](training/pruning.py))
  - Remove Gaussians with opacity < 0.005
  - Runs every 100 iterations starting from iter 3,000
  - Optimizer state cleanup

### Data Loading
- [x] **NeRF Format Dataset** ([core/scene.py](core/scene.py))
  - Load transforms.json
  - Parse camera parameters (FOV, transform matrices)
  - Load ground truth images
  - Image preprocessing (to tensor, device transfer)

---

## ❌ Missing Features (Prioritized)

### Priority 1: Essential Quality Features (CRITICAL for paper-matching results)

#### 1.1 Opacity Reset (30 minutes)
**Status:** Not implemented
**Why:** Prevents opacity saturation during long training runs
**Implementation:** In [training/trainer.py](training/trainer.py), add:
```python
OPACITY_RESET_INTERVAL = 3000

if iteration % OPACITY_RESET_INTERVAL == 0 and iteration > 0:
    with torch.no_grad():
        scene.gaussian_splats._opacity_logits.data.fill_(-2.0)  # sigmoid(-2) ≈ 0.12
    print(f"Opacity reset at iteration {iteration}")
```

**Schedule:** Every 3,000 iterations

---

#### 1.2 Evaluation Metrics (2 hours)
**Status:** Not implemented
**Why:** Need PSNR/SSIM metrics to validate training quality
**Files to Create:**
- `training/metrics.py` (~50 lines)
  - `compute_psnr(img1, img2)` → PSNR in dB
  - `compute_ssim(img1, img2)` → SSIM value
  - `evaluate_scene(rasterizer, scene, cameras)` → average metrics

**Integration:** In [training/trainer.py](training/trainer.py):
```python
if iteration % 1000 == 0:
    metrics = evaluate_scene(rasterizer, scene, scene.cameras[:5], device)
    print(f"Eval at iter {iteration}: PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
```

---

### Priority 2: Performance Optimizations (Reduce training time from 15s/iter to 2-5s/iter)

#### 2.1 Move Covariance Computation After Filtering (30 minutes)
**Status:** Not implemented
**Current Issue:** Computing covariances for ALL 100k Gaussians, then filtering to ~10k valid ones (90% waste)
**Expected Speedup:** ~20% reduction in forward pass time

**File to Modify:** [rendering/rasterizer.py](rendering/rasterizer.py), `render()` method (lines 278-292)

**Change:**
```python
# CURRENT (wasteful):
covs_2d = project_cov_to_2d(gaussian_splats.get_covariances_3d(), positions_cam, camera)
valid_covs_2d = covs_2d[valid_mask]  # Only use ~10% of computed covariances

# NEW (efficient):
valid_covs_3d = gaussian_splats.get_covariances_3d()[valid_mask]  # Filter first
valid_covs_2d = project_cov_to_2d(valid_covs_3d, valid_positions_cam, camera)  # Then compute
```

---

#### 2.2 Mixed Precision Training (1 hour)
**Status:** Not implemented
**Expected Speedup:** 1.5-2× faster
**File to Modify:** [training/trainer.py](training/trainer.py)

**Implementation:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():  # Use FP16 for forward pass
    rendered_image = rasterizer.render(scene.gaussian_splats, camera)
    losses = compute_loss(rendered_image, gt_image)
    total_loss = losses["total"]

optimizer.zero_grad()
scaler.scale(total_loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(scene.gaussian_splats.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

---

#### 2.3 Reduce Initial Gaussian Count (5 minutes)
**Status:** Not implemented
**Current:** Starting with 100k Gaussians (too many, wastes training time)
**Target:** Start with 10k-30k, let densification grow it to 500k-2M
**Expected Speedup:** 10× faster initially, gradually slows as densification adds Gaussians

**File to Modify:** [core/scene.py](core/scene.py)
```python
# Change from:
self.gaussian_splats = GaussianSplats(num_splats=100_000)

# To:
self.gaussian_splats = GaussianSplats(num_splats=10_000)
```

---

#### 2.4 Lower Resolution Training (5 minutes)
**Status:** Not implemented
**Expected Speedup:** 4× fewer pixels = ~4× faster per iteration

**File to Modify:** [core/scene.py](core/scene.py) or dataset loading
```python
import torch.nn.functional as F

# Resize images from 800×800 to 400×400:
gt_image = F.interpolate(
    image.unsqueeze(0).permute(0,3,1,2),
    size=(400, 400),
    mode='bilinear'
).squeeze(0).permute(1,2,0)
```

**Note:** Can progressively increase resolution during training (e.g., 200×200 → 400×400 → 800×800)

---

#### 2.5 Remove Debug Print Statements (30 minutes)
**Status:** Print statements scattered throughout code
**Impact:** Minor performance overhead, cluttered output

**Files to Clean:**
- [rendering/rasterizer.py](rendering/rasterizer.py): Lines 108-109, 149-151
- [training/trainer.py](training/trainer.py): Debug prints

---

### Priority 3: Advanced Features (Nice-to-have for production quality)

#### 3.1 Spherical Harmonics for View-Dependent Colors (4-6 hours)
**Status:** Not implemented
**Current:** Using flat RGB colors (view-independent)
**Target:** SH degree 3 for realistic view-dependent effects

**Files to Create:**
- `core/sh_coefficients.py` - SH basis functions and evaluation
- Modify [core/gaussian.py](core/gaussian.py) to support `_features_rest` parameter
- Modify [rendering/rasterizer.py](rendering/rasterizer.py) to compute viewing directions and evaluate SH

**Benefits:**
- Better visual quality with view-dependent specular effects
- Matches original paper implementation

---

#### 3.2 COLMAP Sparse Point Cloud Initialization (3-4 hours)
**Status:** Not implemented (currently using random initialization)
**Target:** Initialize Gaussians from COLMAP sparse reconstruction

**Files to Create:**
- `data/colmap_loader.py` - Read COLMAP binary/text formats (cameras.bin, images.bin, points3D.bin)

**Benefits:**
- Faster convergence (5-10× fewer iterations to achieve same quality)
- Better initialization for real-world scenes

---

#### 3.3 Interactive Viewer (1-2 weeks)
**Status:** Not implemented

**Features:**
- Real-time rendering at 30+ FPS
- Camera controls (orbit, pan, zoom, WASD navigation)
- Training visualization (loss curves, Gaussian count)
- Export (video, images, point cloud)

**Technologies:**
- pyglet or similar for OpenGL context
- imgui-python for UI

---

#### 3.4 CUDA Optimization (2-4 weeks)
**Status:** Not implemented (currently using PyTorch)

**Target:**
- Custom CUDA kernels for tile-based rasterization
- Fast radix sort for depth ordering
- Expected speedup: 10-50× faster rendering

**Benefits:**
- Production-ready performance
- Real-time viewer at high resolutions

---

## Performance Metrics

### Current State:
- **Gaussians:** 100,000 (random init)
- **Iteration Time:** ~15 seconds
- **Full Training (30k iters):** ~125 hours (5.2 days)

### After Priority 1 (Quality):
- **Initial Gaussians:** 10,000
- **Peak Gaussians:** 500k-1M (via densification)
- **Quality:** Paper-matching with D-SSIM loss + densification + pruning

### After Priority 2 (Performance):
- **Iteration Time (early):** ~0.25-0.5 seconds (10k Gaussians, 400×400 resolution, AMP)
- **Iteration Time (peak):** ~2-5 seconds (500k-1M Gaussians)
- **Full Training (30k iters):** ~2-4 hours
- **Overall Speedup:** 40-60× faster

### After Priority 3 (Advanced):
- **With SH:** Better visual quality, view-dependent effects
- **With COLMAP init:** 5-10× fewer iterations needed
- **With CUDA:** 10-50× faster rendering (real-time viewer)

---

## Training Parameters (from Original Paper)

### Schedule:
- **Total iterations:** 30,000
- **Densification:** Every 100 iters, from iter 500 to 15,000
- **Pruning:** Every 100 iters, starting from iter 3,000
- **Opacity reset:** Every 3,000 iterations

### Thresholds:
- **Gradient threshold (densification):** 0.0002 (average positional gradient magnitude)
- **Scale threshold (split vs clone):** 0.01 world units
- **Opacity threshold (pruning):** 0.005
- **Scale reduction on split:** 1.6× smaller

### Gaussian Count:
- **Initial (random):** 10,000-30,000
- **Initial (COLMAP):** Use sparse points (10k-100k)
- **Peak:** 500,000-2,000,000 after densification

### Loss Function:
- **L = 0.8 × L1 + 0.2 × (1 - SSIM)**

---

## Implementation Checklist

### Week 1: Essential Quality
- [ ] Opacity reset (30 min)
- [ ] Evaluation metrics (2 hours)
- [ ] Test full 30k iteration training run
- [ ] Verify densification/pruning working correctly

### Week 2: Performance
- [ ] Move covariance after filtering (30 min)
- [ ] Mixed precision training (1 hour)
- [ ] Reduce initial Gaussian count (5 min)
- [ ] Lower resolution training (5 min)
- [ ] Clean up debug prints (30 min)
- [ ] Benchmark: measure speedup

### Week 3: Validation
- [ ] Test on NeRF synthetic datasets (Lego, Ship, etc.)
- [ ] Target: PSNR > 30 dB, SSIM > 0.95
- [ ] Verify training completes in 2-4 hours

### Week 4+: Advanced Features
- [ ] Spherical harmonics (4-6 hours)
- [ ] COLMAP point cloud initialization (3-4 hours)
- [ ] Interactive viewer (1-2 weeks)
- [ ] CUDA optimization (2-4 weeks)

---

## Quick Start for Contributors

### Setup:
```bash
# Install dependencies
uv sync

# Prepare dataset (NeRF format)
# Place in data/ directory with transforms.json
```

### Training:
```bash
python main.py
```

### Key Files:
- [main.py](main.py) - Entry point
- [core/gaussian.py](core/gaussian.py) - Gaussian representation
- [rendering/rasterizer.py](rendering/rasterizer.py) - Tile-based rendering
- [training/trainer.py](training/trainer.py) - Training loop
- [training/densification.py](training/densification.py) - Adaptive densification
- [training/pruning.py](training/pruning.py) - Gaussian pruning

### Current Issues:
- Training slow (15s/iter) → Priority 2 optimizations will fix
- No evaluation metrics → Priority 1.2
- No opacity reset → Priority 1.1

---

## References

- **Original Paper:** "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- **Project Page:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **GitHub:** https://github.com/graphdeco-inria/gaussian-splatting