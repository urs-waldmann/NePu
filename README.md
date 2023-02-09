# Neural Puppeteer (NePu)
Official Code Release for the ACCV'22 paper "Neural Puppeteer: Keypoint-Based Neural Rendering of Dynamic Shapes"

[Project Page](https://urs-waldmann.github.io/NePu/) | [Paper](https://urs-waldmann.github.io/NePu/docs/nepu_paper.pdf) | [Supplementary](https://urs-waldmann.github.io/NePu/docs/nepu_supp.pdf) | [Data](https://zenodo.org/record/7149178)

**Abstract**

We introduce Neural Puppeteer, an efficient neural rendering pipeline for articulated shapes. By inverse rendering, we can predict 3D keypoints from multi-view 2D silhouettes alone, without requiring texture information. Furthermore, we can easily predict 3D keypoints of the same class of shapes with one and the same trained model and generalize more easily from training with synthetic data which we demonstrate by successfully applying zero-shot synthetic to real-world experiments. We demonstrate the flexibility of our method by fitting models to synthetic videos of different animals and a human, and achieve quantitative results which outperform our baselines. Our method uses 3D keypoints in conjunction with individual local feature vectors and a global latent code to allow for an efficient representation of time-varying and articulated shapes such as humans and animals. In contrast to previous work, we do not perform reconstruction in the 3D domain, but project the 3D features into 2D cameras and perform reconstruction of 2D RGB-D images from these projected features, which is significantly faster than volumetric rendering. Our synthetic dataset will be publicly available, to further develop the evolving field of animal pose and shape reconstruction. 

# Install

All experiments with NePu were run using CUDA version 11.6 and the official pytorch docker image `nvcr.io/nvidia/pytorch:22.02-py3`, as published by nvidia [here](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). Additionally, you will need to install the ```imageio``` library.

Alternatively, we provide the `nepu_env.yaml` file that holds all python requirements for this project. To conveniently install them automatically with anaconda you can use:
```
conda env create -f nepu_env.yaml
conda activate nepu
```


# Datasets

You find our synthetic data sets [here](https://zenodo.org/record/7149178). Download and extract all folders. Copy all four extracted folders to ```./data/```.

# Training

To train NePu, please run:

``` python train.py -exp_name EXP_NAME -cfg_file CFG_FILE -data DATA_TYPE```,

where the ```CFG_FILE``` is the path to a ```.yaml```-file specifiying the configurations, like this one [here](./configs/nepu.yaml). ```DATA_TYPE``` can be one of the categories of our synthetic dataset, namely ```giraffes, pigeons, cows, humans```.


# Rendering

To render multiple views of the test set, run:

``` python test.py -exp_name EXP_NAME -checkpoint CKPT -data DATA_TYPE ```,

where ```CKPT``` specifies the epoch of the trained weights and the other command line arguments are the same as above.

TODO: novel views

# Inverse-Rendering

For our inverse-rendering-based 3D keypoint detection run:

```
python sil2kps.py -exp_name EXP_NAME -checkpoint CKPT -cams CAM_IDS -data DATA_TYPE 
```

where ```CAM_IDS``` spcifies the views used for 3D keypoint detection and the other command line arguments are the same as above.

# Pretrained Models

TODO

# Contact

For questions, comments and to discuss ideas please contact ```{Urs Waldmann, Simon Giebenhain, Ole Johannsen}``` via ```firstname.lastname (at] uni-konstanz {dot| de```.

# Citation

```
@inproceedings{giewald2022nepu,
title={Neural Puppeteer: Keypoint-Based Neural Rendering of Dynamic Shapes},
author={Giebenhain, Simon and Waldmann, Urs and Johannsen, Ole and Goldluecke, Bastian},
booktitle={Asian Conference on Computer Vision (ACCV)},
year={2022},
}
```

# Acknowledgment

# License
