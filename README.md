# 📦 Latent-Based Continual Learning with Dual-Layered Distillation and a Streamlined U- Net for Efficient Text-to-Image Generation

[![License](https://img.shields.io/badge/license-MIT-blue)]()

A PyTorch-based framework for **Latent-Based Continual Learning with
Dual-Layered Distillation and a Streamlined U- Net for Efficient Text-to-Image Generation** — a compact, efficient diffusion model that reuses weights across timesteps, distilled for faster sampling without quality loss.

---

## 📌 Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training](#training)  
  - [Sampling](#sampling)  
- [Configuration](#configuration)  
- [Examples](#examples)  
- [Benchmark & Results](#benchmark--results)  
- [Citation](#citation)  
- [License](#license)  

---

## 📈 Overview

WSDD aims to accelerate diffusion sampling by:

- **Sharing** neural network weights across multiple timesteps.
- **Distilling** a full-scale teacher model into a lightweight student.
- Striking a balance between **sampling speed** and **image fidelity**.

The result is a model that runs significantly faster (e.g., 4–8× speedup) with perceptually similar output to Full Diffusion.

---

## ✨ Key Features

- 🔁 **Weight Sharing** across distillation steps  
- 🎯 **Step Reduction** via progressive distillation  
- 🛠️ Fully compatible with Hugging Face `diffusers` pipelines  
- 🔋 Supports CUDA/FP16 inference  
- 🧠 Extensible modular architecture  

---

## 🛠️ Installation

```bash
git clone https://github.com/naimurborno/WSDD-Weight-Shared-Distilled-Diffusion.git
cd WSDD-Weight-Shared-Distilled-Diffusion

# Install framework and dependencies
pip install -r requirements.txt
python train.py \
  --teacher_model "sd-full" \
  --student_scales 32 64 96 128 \
  --sigmas 1.0 0.9 0.8 0.6 0.0 \
  --batch_size 32 \
  --steps_per_epoch 1000 \
  --epochs 10 \
  --output_dir "checkpoints/wsdd"
from wsdd import WSDDPipeline

pipe = WSDDPipeline.from_pretrained("checkpoints/wsdd/latest.ckpt", device="cuda")
img = pipe(
    prompt="a serene mountain lake at sunrise",
    scales=[32,64,96,128],
    sigmas=[1.0,0.9,0.8,0.6,0.0],
    guidance_scale=7.5
).images[0]

img.save("output.png")
