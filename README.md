# Neural Puppeteer (NePU)
Official Code Release for the ACCV'22 paper "Neural Puppeteer: Keypoint-Based Neural Rendering of Dynamic Shapes"

[Project Page](https://urs-waldmann.github.io/NePu/) | [Paper](todo-insert-link) | [Supplementary](todo-insert-link) | [Data](todo-insert-link)

TODO teaser GIF

# Install

All experiments with NePu were run using CUDA version 11.6 and the official pytorch docker image `nvcr.io/nvidia/pytorch:22.02-py3`, as published by nvidia [here](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). Additionally, you will need to install the ```imageio```library.

Alternatively, we provide the `nepu_env.yaml` file that holds all python requirements for this project. To conveniently install them automatically with anaconda you can use:
```
conda env create -f nepu_env.yml
conda activate nepu
```


# Datasets

TODO

# Training

To train NePu please run 

``` python train.py -exp_name EXP_NAME -cfg_file CFG_FILE -data DATA_TYPE```

where the ```CFG_FILE``` is the path to a ```.yaml```-file specifiying the configurations, describes in more detail [here](todo-insert-link-here). ```DATA_TYPE```can be one of the categories of our synthetic dataset, namely ```giraffe, pigeon, cow, human```.


# Rendering

To render multiple views of the test set run 

``` python test.py -exp_name EXP_NAME -checkpoint CKPT -data DATA_TYPE ```

where ```CKPT``` specifies the epoch of the trained weights.
TODO: custom and novel views

# Inverse-Rendering

For our inverse-rendering-based 3D keypoint detection run

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
