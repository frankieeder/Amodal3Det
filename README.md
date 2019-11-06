## Documentation
This repository containts a re-implementation in Tensorflow of Deng et al's approach used in "Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images"

Slides:
https://docs.google.com/presentation/d/18EJWAZ90onmsTlhOMb76VrFumY50cNyLHIsopiih41A/edit#slide=id.g56eb6faf5d_0_38

Paper:
https://docs.google.com/document/d/1EC-AM8B99veG0Wf1moL0jlVG9KfAvJEW0KIl_xaeN-4/edit

________________________________________________________________________________________________

# Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images

By [Zhuo Deng](http://www.dnnseye.com), [Longin Jan Latecki](https://cis.temple.edu/~latecki/) (Temple University).
This paper was published in CVPR 2017.

## License 

Code is released under the GNU GENERAL PUBLIC LICENSE (refer to the LICENSE file for details).

## Cite The Paper
If you use this project for your research, please consider citing:

    @inproceedings{zhuo17amodal3det,
        author = {Zhuo Deng and Longin Jan Latecki},
        booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
        title = {Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images},
        year = {2017}
    }


## Contents
1. [System requirements](#system)
2. [Basic Installation](#install)
3. [Usage](#usage)

## System requirements
The code is tested on the following system:
1. OS: Ubuntu 14.04
2. Hardware: Nvidia Titan X (GPU usage: ~9GB)
3. Software: Caffe, CUDA-7.5, cuDNN v4, Matlab 2015a, Anaconda2

## Basic Installation
1. clone the Amodal3Det repository: 
    ```Shell
    git clone https://github.com/phoenixnn/Amodal3Det.git

    ```
2. build Caffe:
    ```Shell
    # assume you clone the repo into the local your_root_dir
    cd your_root_dir
    make all -j8 && make pycaffe
    ```
3. install cuDNN:
    ```Shell
    sudo cp cudnn_folder/include/cudnn.h /usr/local/cuda-7.5/include/
    sudo cp cudnn_folder/lib64/*.so* /usr/local/cuda-7.5/lib64/
    ```

## Usage
1. Download NYUV2 dataset with 3D annotations and unzip:
    ```Shell
    wget 'https://cis.temple.edu/~latecki/TestData/DengCVPR2017/NYUV2_3D_dataset.zip' -P your_root_dir/dataset/NYUV2/
    ```
2. Download precomputed 2D segment proposals based on MCG3D and unzip:
    ```Shell
    wget 'https://cis.temple.edu/~latecki/TestData/DengCVPR2017/Segs.zip' -P your_root_dir/matlab/NYUV2/
    ```
3. Download pretrained models and unzip:
    ```Shell
    wget 'https://cis.temple.edu/~latecki/TestData/DengCVPR2017/pretrained.zip' -P your_root_dir/rgbd_3det/
    ```
    VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), 
    but is provided here for your convenience.

4. Run "your_root_dir/matlab/NYUV2/pipeline.m" in Matlab to extact required data.
5. Set up training/test data:

   run "setup_training_data.py" and "setup_testing_data.py" under your_root_dir/rgbd_3det/data respectively
6. Train model:
    ```Shell
    cd your_root_dir
    ./trainNet.sh
    ```
7. Test model: run "test_cnn.py"


