# Mask TextSpotter v3
This is a PyTorch implemntation of the ECCV 2020 paper [Mask TextSpotter v3](https://arxiv.org/abs/2007.09482). Mask TextSpotter v3 is an end-to-end trainable scene text spotter that adopts a Segmentation Proposal Network (SPN) instead of an RPN. Mask TextSpotter v3 significantly improves robustness to rotations, aspect ratios, and shapes.

## Relationship to Mask TextSpotter
Here we label the Mask TextSpotter series as Mask TextSpotter v1 ([ECCV 2018 paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.pdf), [code](https://github.com/lvpengyuan/masktextspotter.caffe2)), Mask TextSpotter v2 ([TPAMI paper](https://ieeexplore.ieee.org/document/8812908), [code](https://github.com/MhLiao/MaskTextSpotter)), and Mask TextSpotter v3 (ECCV 2020 paper).

This project is under a lincense of Creative Commons Attribution-NonCommercial 4.0 International. Part of the code is inherited from [Mask TextSpotter v2](https://github.com/MhLiao/MaskTextSpotter), which is under an MIT license.


## Installation

### Requirements:
- Python3 (Python3.7 is recommended)
- PyTorch >= 1.4 (1.4 is recommended)
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9 (This is very important!)
- OpenCV
- CUDA >= 9.0 (10.0.130 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name masktextspotter -y
  conda activate masktextspotter

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 editdistance 

  # install PyTorch
  conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

  export INSTALL_DIR=$PWD

  # install pycocotools
  cd $INSTALL_DIR
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install

  # install apex (optional)
  cd $INSTALL_DIR
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  python setup.py install --cuda_ext --cpp_ext

  # clone repo
  cd $INSTALL_DIR
  git clone https://github.com/MhLiao/MaskTextSpotterV3.git
  cd MaskTextSpotterV3

  # build
  python setup.py build develop


  unset INSTALL_DIR
```

## Models
Download Trained [model](https://drive.google.com/file/d/1XQsikiNY7ILgZvmvOeUf9oPDG4fTp0zs/view?usp=sharing)

## Demo 
You can run a demo script for a single image inference by ```python tools/demo.py```.

## Datasets
The datasets are the same as Mask TextSpotter v2.

Download the ICDAR2013([Google Drive](https://drive.google.com/open?id=1sptDnAomQHFVZbjvnWt2uBvyeJ-gEl-A), [BaiduYun](https://pan.baidu.com/s/18W2aFe_qOH8YQUDg4OMZdw)) and ICDAR2015([Google Drive](https://drive.google.com/open?id=1HZ4Pbx6TM9cXO3gDyV04A4Gn9fTf2b5X), [BaiduYun](https://pan.baidu.com/s/16GzPPzC5kXpdgOB_76A3cA)) as examples.

The SCUT dataset used for training can be downloaded [here](https://drive.google.com/file/d/1BpE2GEFF7Ay7jPqgaeHxMmlXvM-1Es5_/view?usp=sharing).

The converted labels of Total-Text dataset can be downloaded [here](https://1drv.ms/u/s!ArsnjfK83FbXgcpti8Zq9jSzhoQrqw?e=99fukk).

The converted labels of SynthText can be downloaded [here](https://1drv.ms/u/s!ArsnjfK83FbXgb5vgOOVPYywgCWuQw?e=UPuNTa).

The root of the dataset directory should be ```MaskTextSpotterV3/datasets/```.

## Testing
### Prepar dataset
An example of the path of test images: ```MaskTextSpotterV3/datasets/icdar2015/test_iamges```

### Check the config file (configs/finetune.yaml) for some parameters.
test dataset: ```TEST.DATASETS```; 

input size: ```INPUT.MIN_SIZE_TEST''';

model path: ```MODEL.WEIGHT```;

output directory: ```OUTPUT_DIR```

### run ```sh test.sh```


## Training
Place all the training sets in ```MaskTextSpotterV3/datasets/``` and check ```DATASETS.TRAIN``` in the config file.
### Pretrain
Trained with SynthText

```python3 -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/pretrain/seg_rec_poly_fuse_feature.yaml ```
### Finetune
Trained with a mixure of SynthText, icdar2013, icdar2015, scut-eng-char, and total-text

check the initial weights in the config file.

```python3 -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/mixtrain/seg_rec_poly_fuse_feature.yaml ```

## Evaluation
### Evaluation for Total-Text dataset

```
cd evaluation/totaltext/e2e/
# edit "result_dir" in script.py
python script.py
```

### Evaluation for the Rotated ICDAR 2013 dataset
First, generate the Rotated ICDAR 2013 dataset
```
cd tools
# set the specific rotating angle in convert_dataset.py
python convert_dataset.py
```
Then, run testing (change test set in YAML) and evaluate by ```evaluation/rotated_icdar2013/e2e/script.py```

## Citing the related works

Please cite the related works in your publications if it helps your research:

    @inproceedings{liao2020mask,
      title={Mask TextSpotter v3: Segmentation Proposal Network for Robust Scene Text Spotting},
      author={Liao, Minghui and Pang, Guan and Huang, Jing and Hassner, Tal and Bai, Xiang},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2020}
    }

    @article{liao2019mask,
      author={M. {Liao} and P. {Lyu} and M. {He} and C. {Yao} and W. {Wu} and X. {Bai}},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title={Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes},
      year={2019},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TPAMI.2019.2937086}
    }
    
    @inproceedings{lyu2018mask,
      title={Mask textspotter: An end-to-end trainable neural network for spotting text with arbitrary shapes},
      author={Lyu, Pengyuan and Liao, Minghui and Yao, Cong and Wu, Wenhao and Bai, Xiang},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      pages={67--83},
      year={2018}
    }
    
