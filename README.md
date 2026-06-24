# HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video

[![arXiv](https://img.shields.io/badge/arXiv-PAPER_ID-b31b1b)](ARXIV_URL)
[![Project Page](https://img.shields.io/badge/Project-Page-orange)](https://iridescentjiang.github.io/HiReFF/)

Official PyTorch implementation for the paper:

> **[HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video](ARXIV_URL) [ECCV 2026]**

[Yiming Jiang](https://scholar.google.com.hk/citations?user=gqaK3igAAAAJ&hl=zh-CN)<sup>&#x2606;</sup>,
[Hanzhang Tu](https://scholar.google.com.hk/citations?user=0S0lNhUAAAAJ&hl=zh-CN&oi=ao),
[Wenfeng Song](https://scholar.google.com.hk/citations?user=BDfZbbEAAAAJ&hl=zh-CN),
[Siyou Lin](https://scholar.google.com.hk/citations?user=XBzr0pkAAAAJ&hl=zh-CN&oi=ao),
[Liang An](https://scholar.google.com.hk/citations?user=s0T1w0gAAAAJ&hl=zh-CN&oi=sra),
[Shuai Li](https://scholar.google.com.hk/citations?user=hn0KFx8AAAAJ&hl=zh-CN),
[Aimin Hao](https://research.buaa.edu.cn/en/persons/aimin-hao/)<sup>&#x2709;</sup>,
[Yebin Liu](https://scholar.google.com.hk/citations?user=ogXIdlYAAAAJ&hl=zh-CN)

> <sup>&#x2606;</sup> Work done during an internship at Tsinghua University. &nbsp; <sup>&#x2709;</sup> Corresponding author. Email: jiangyimingjym@buaa.edu.cn ham@buaa.edu.cn liuyebin@tsinghua.edu.cn

![Teaser](static/images/teaser.png)

> **HiReFF** is a feed-forward method for 2K-resolution 360° human video reconstruction from uncalibrated sparse-view videos. Taking only four views separated by 90° as input, it reconstructs temporally consistent 3D Gaussians in a streaming fashion at 3.01 FPS on a single RTX 4090 GPU, and achieves 2K resolution with only 34% additional VRAM during training compared to 0.5K.

## The Pipeline of Our Method

![Pipeline](static/images/pipeline.png)

> **HiReFF** decomposes 4D human reconstruction into two key tasks: foreground 3D Gaussian reconstruction from uncalibrated sparse-view videos and computationally efficient high-resolution synthesis. It employs Scale-synchronized Camera Calibration to resolve metric scale ambiguity, Gaussian-wise Foreground Masking to reconstruct clean foregrounds, and High-resolution Side-tuning for efficient 2K rendering.

## Code

- [ ] Release training code
- [ ] Release inference code
- [ ] Release pretrained models

## Citation

```bibtex
@inproceedings{jiang2026hireff,
  title     = {HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video},
  author    = {Yiming Jiang and Hanzhang Tu and Wenfeng Song and Siyou Lin and Liang An and Shuai Li and Aimin Hao and Yebin Liu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## Acknowledgement

We gratefully acknowledge the authors of [VGGT](https://github.com/facebookresearch/vggt) and [AnySplat](https://github.com/AnySplat/AnySplat) for making their code publicly available. Any third-party packages are owned by their respective authors and must be used under their respective licenses.
