# GDN_Inpainting
An open source code for paper "Pixel-wise Dense Detector for Image Inpainting" (PG 2020)

<div align=center>
  <img width="750" src="https://github.com/Evergrow/GDN_Inpainting/blob/master/images/teaser.gif/" hspace="10">
</div>

Deep inpainting technique fills the semantically correct and visually plausible contents in the missing regions of corrupted images. Above results are presented by our framework.

## Prerequisites
* Ubuntu 16.04
* Python 3
* NVIDIA GPU CUDA + cuDNN
* TensorFlow 1.12.0

## Usage
### Set up
* Clone this repo:
```python
git clone https://github.com/Evergrow/GDN_Inpainting.git
cd GDN_Inpainting
```
* Install [TensorFlow](https://www.tensorflow.org/) and dependencies
* Download datasets: We use [Places2](http://places2.csail.mit.edu/), [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. Some common inpainting datasets such as [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [ImageNet](http://www.image-net.org/) are also available.
* Collect masks: We upload the [script](https://github.com/Evergrow/GDN_Inpainting/blob/master/mask_processing.py) processing raw mask [QD-IMD](https://github.com/karfly/qd-imd) as the training mask. [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the testing mask. Note that the square mask is not a good choice for training our framewrok, while the test mask is freestyle.

### Training
* Modify gpu id, dataset path, mask path, and checkpoint path in the [config file](https://github.com/Evergrow/GDN_Inpainting/blob/master/config.yml). Adjusting some other parameters if you like.
* Run ```python train.py``` and view training progress ```tensorboard --logdir [path to checkpoints]```

### Testing
Choose the input image, mask and model to test:
```python
python test.py --image [input path] --mask [mask path] --output [output path] --checkpoint_dir [model path]
```
### Pretrained models
[Celeba-HQ](https://drive.google.com/drive/folders/1vJ0hhtPtKqp-nj8CAVGuKk3ZKlZ3JCX5?usp=sharing) and [Places2](https://drive.google.com/drive/folders/1vQhKOyzFJM_upEyPNzXfmSmb9ZCxsvjE?usp=sharing) pretrained models are released for quick testing. Download the models using Google Drive links and move them into your ./checkpoints directory.

##
