# HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video

Official PyTorch implementation for the paper:

> **[HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video](ARXIV_URL) [ECCV 2026]**
>
> Yiming Jiang<sup>&#x2606;</sup>, Hanzhang Tu, Wenfeng Song, Siyou Lin, Liang An, Shuai Li, Aimin Hao<sup>&#x2709;</sup>, Yebin Liu
>
> <sup>&#x2606;</sup> Work done during an internship at Tsinghua University. &nbsp; <sup>&#x2709;</sup> Corresponding author. Email: \_\_\_@\_\_\_

![Teaser](static/images/teaser.png)

> **HiReFF** is a feed-forward method for 2K-resolution 360° human video reconstruction from uncalibrated sparse-view videos. Taking only four views separated by 90° as input, it reconstructs temporally consistent 3D Gaussians in a streaming fashion at 3.01 FPS on a single RTX 4090 GPU, and achieves 2K resolution with only 34% additional VRAM during training compared to 0.5K.

## The Pipeline of Our Method

![Pipeline](static/images/pipeline.png)

> HiReFF decomposes 4D human reconstruction into two key tasks: foreground 3D Gaussian reconstruction from uncalibrated sparse-view videos and computationally efficient high-resolution synthesis. It employs Scale-synchronized Camera Calibration to resolve metric scale ambiguity, Gaussian-wise Foreground Masking to reconstruct clean foregrounds, and High-resolution Side-tuning for efficient 2K rendering.

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
