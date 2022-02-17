## Open-World Semi-Supervised Learning
Kaidi Cao*, Maria Brbić*, Jure Leskovec

[Project website](http://snap.stanford.edu/orca)
_________________

This repo contains the reference source code in PyTorch of the ORCA algorithm. ORCA is a pipeline that recognizes previously seen classes and discovers novel, never-before-seen classes at the same time.. For more details please check our paper [Open-World Semi-Supervised Learning](https://arxiv.org/pdf/2102.03526.pdf) (ICLR '22). 

### Dependencies

The code is built with following libraries:

- [PyTorch==1.9](https://pytorch.org/)
- [sklearn==1.0.1](https://scikit-learn.org/)

### Usage

##### Get Started

We use SimCLR for pretraining. The weights used in our paper can be downloaded in this [link](https://drive.google.com/file/d/19tvqJYjqyo9rktr3ULTp_E33IqqPew0D/view?usp=sharing).

- To train on CIFAR-100, run

```bash
python orca_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5
```

- To train on ImageNet-100, first use ```gen_imagenet_list.py``` to generate corresponding splitting lists, then run

```bash
python orca_imagenet.py --labeled-num 50 --labeled-ratio 0.5
```

### Citing

If you find our code useful, please consider citing:

```
@inproceedings{
    cao2022openworld,
    title={Open-World Semi-Supervised Learning},
    author={Kaidi Cao and Maria Brbic and Jure Leskovec},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=O-r8LOR-CCA}
}
```