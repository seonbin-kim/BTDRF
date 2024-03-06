# BTD-RF
This repository is an implementation for the paper: BTD-RF: 3D Scene Reconstruction Using Block-Term Tensor Decomposition.  
The further cleaning up of our README page is processing.

![model_architecture](https://github.com/seonbin-kim/BTDRF/assets/90370359/f130e50b-329c-4323-aff7-c552e74e02a6)


## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.13.1 and Ubuntu 22.04 + Pytorch 1.13.1

Install environment:
```
conda create -n BTD-RF python=3.8
conda activate BTD-RF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard einops
```


## Training
You can train BTD-RF using the script `train.py`

```
python train.py --config configs/lego.txt
```
    

## Citation
```
@misc{kim2024btdrf,
  author = {Seon Bin Kim and Sangwon Kim and Dasom Ahn and Byoung Chul Ko},
  title = {BTD-RF: 3D Scene Reconstruction Using Block-Term Tensor Decomposition},
  year = {2024}
}
```
