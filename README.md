<div align="center">
<h1>Modality-Aware Out-of-Distribution Detection for Multi-Modal Action Recognition</h1>

[**Lars Doorenbos**](https://scholar.google.com/citations?user=i2LqZCwAAAAJ&hl=en)<sup>1,2</sup>&nbsp;·&nbsp;
**Duc Manh Vu**<sup>1</sup>&nbsp;·&nbsp;
[**Serdar Ozsoy**](https://scholar.google.com/citations?user=6jXE6SYAAAAJ&hl=en)<sup>1</sup> &nbsp;·&nbsp;
[**Juergen Gall**](https://scholar.google.de/citations?user=1CLaPMEAAAAJ)<sup>1,2</sup>

<sup>1</sup>University of Bonn &nbsp;&nbsp; <sup>2</sup>Lamarr Institute for Machine Learning and Artificial Intelligence &nbsp;&nbsp;

<h3>⭐ ECCV 2026</h3>

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/pdf/2606.24404)

</div>

## Abstract 
The incorporation of additional modalities into action recognition models increases their performance across a wide range of settings. However, how this additional information can contribute to making the models more robust remains underexplored, particularly for the case of multi-modal out-of-distribution (OOD) detection. 
While methods exist that regularize the multi-modal training process with OOD detection in mind, they still apply off-the-shelf OOD detectors designed for the uni-modal case during inference, discarding important information. 
Based on an interesting relationship we find between the multi-modal and uni-modal predictions, we propose to use this signal to build a post-hoc detector explicitly designed for the multi-modal scenario. 
We combine this new source of information with a feature-space score, which detects off-manifold samples in the multi-modal space, and normalize them by the multi-modal logits. 
In doing so, the proposed hybrid detector is compatible with existing training-time approaches and consistently improves performance.
Experiments on a wide range of established datasets from the MultiOOD benchmark show that, on average, our approach outperforms the state of the art. 
Our results show the importance of explicitly considering the different modalities at inference time for multi-modal OOD detection.


## Installation & Data
This code is based on the [MultiOOD](https://github.com/donghao51/MultiOOD/tree/main?tab=readme-ov-file) and [DPU](https://github.com/lili0415/dpu-ood-detection) repositories. Please refer to those for instructions on downloading the datasets.

## Training

Our method is post-hoc. Therefore, training is identical to DPU. For instance, for HMDB, use:
```
python train_video_flow.py --near_ood --dataset HMDB --lr 0.0001 --seed 0 --bsz 64 --num_workers 4 --start_epoch 10 --use_single_pred --use_irm --use_a2d --a2d_max_hellinger --a2d_ratio 0.5 --use_npmix --max_ood_hellinger --a2d_ratio_ood 0.5 --ood_entropy_ratio 0.5 --nepochs 50 --appen '1' --save_best --save_checkpoint --datapath '/path/to/HMDB51/'
```

## Testing

First, obtain all representations from the ID and OOD datasets for faster processing:
```
python test_video_flow.py --bsz 64 --num_workers 8 --near_ood --dataset 'HMDB' --appen 'a2d1_' --resumef 'models/log_video_flow_HMDB_near_ood_lr_0.0001_bsz_16_50_adam_single_pred_a2d_max_hellinger_0.5_max_ood_hellinger_0.5_entropy_0.5_start_epoch_10_nn_k_3_mixup_alpha_10.0a2d1_best.pt' --datapath '/path/to/HMDB51/'
```

Then, compute the OOD score:
```
python compute_performance.py --near_ood --low_var 0.005 --dataset 'HMDB' --appen 'a2d1_' --resumef 'models/log_video_flow_HMDB_near_ood_lr_0.0001_bsz_16_50_adam_single_pred_a2d_max_hellinger_0.5_max_ood_hellinger_0.5_entropy_0.5_start_epoch_10_nn_k_3_mixup_alpha_10.0a2d1_best.pt'
```

## Citation

```bibtex
@article{doorenbos2026modality,
  title={Modality-Aware Out-of-Distribution Detection for Multi-Modal Action Recognition},
  author={Doorenbos, Lars and Vu, Duc Manh and Ozsoy, Serdar and Gall, Juergen},
  journal={arXiv preprint arXiv:2606.24404},
  year={2026}
}
```