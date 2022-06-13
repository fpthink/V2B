# 3D Siamese Voxel-to-BEV Tracker for Sparse Point Clouds

## Introduction

This repository is released for V2B in our [NeurIPS 2021 paper (poster)](https://arxiv.org/pdf/2111.04426.pdf). 

**Note**: In order to make the code structure clearer and more reasonable, we refactored the entire project. If you are more familiar with [P2B](https://github.com/HaozheQi/P2B) and the code of our previously published version, you can continue to refer to the code of the [first version](https://drive.google.com/file/d/1yPjC3hES0pU4pzbQsUv9hbCNYu0p4XUo/view?usp=sharing).

## Environment settings
* Create an environment for v2b
```
conda create -n V2B python=3.7
conda activate V2B
```

* Install pytorch and torchvision
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0
```

* Install dependencies.
```
pip install -r requirements.txt
```

* Build `_ext` module.
```
cd V2B_main/lib/pointops
python setup.py install
cd ../../
```

## Data preparation
### [KITTI dataset](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)
* Download the [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Unzip the downloaded files and place them under the same parent folder.

### [nuScenes dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)
* Download the Full dataset (v1.0) from [nuScenes](https://www.nuscenes.org/).
  
    Note that base on the offical code [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit), we modify and use it to convert nuScenes format to KITTI format. It requires metadata from nuScenes-lidarseg. Thus, you should replace *category.json* and *lidarseg.json* in the Full dataset (v1.0). We provide these two json files in the nuscenes_json folder.

    Executing the following code to convert nuScenes format to KITTI format
    ```
    cd nuscenes-devkit-master/python-sdk/nuscenes/scripts
    python export_kitti.py --nusc_dir=<nuScenes dataset path> --nusc_kitti_dir=<output dir> --split=<dataset split>
    ```

    Note that the parameter of "split" should be "train_track" or "val". In our paper, we use the model trained on the KITTI dataset to evaluate the generalization of the model on the nuScenes dataset.
	
### [Waymo open dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)
* We follow the benchmark created by [LiDAR-SOT](https://github.com/TuSimple/LiDAR_SOT) based on the waymo open dataset. You can download and process the waymo dataset as guided by [their code](https://github.com/TuSimple/LiDAR_SOT), and use our code to test model performance on this benchmark.
* The benchmark they built have many things that we don't use, but the following processing results are necessary:
```
[waymo_sot]
    [benchmark]
        [validation]
            [vehicle]
                bench_list.json
                easy.json
                medium.json
                hard.json
            [pedestrian]
                bench_list.json
                easy.json
                medium.json
                hard.json
    [pc]
        [raw_pc]
            Here are some segment.npz files containing raw point cloud data
    [gt_info]
        Here are some segment.npz files containing tracklet and bbox data
```

**Node**: After you get the dataset, please modify the path variable ```data_dir&val_data_dir``` about the dataset under configuration file ```V2B_main/utils/options```.

## Evaluation

Train a new model:
```
python main.py --which_dataset KITTI/NUSCENES --category_name category_name
```

Test a model:
```
python main.py --which_dataset KITTI/NUSCENES/WAYMO --category_name category_name --train_test test
```
For more preset parameters or command debugging parameters, please refer to the relevant code and change it according to your needs.

**Recommendations**: 
- We have provided some pre-trained models under ```V2B_main/results``` folder, you can use and test them directly.  
- Since both kitti and waymo are datasets constructed from 64-line LiDAR, nuScenes is a 32-line LiDAR. We recommend you: train your model on KITTI and verify the generalization ability of your model on waymo. Train on nuScenes or simply skip this dataset. We do not recommend that you verify the generalization ability of your model on nuScenes. 

## Todo

```
1. Provide visualization codes.
2. Provide test results on waymo open dataset so that you can use the results directly in your paper.
```

## Citation

If you find the code or trained models useful, please consider citing:

```
@inproceedings{hui2021v2b,
  title={3D Siamese Voxel-to-BEV Tracker for Sparse Point Clouds},
  author={Hui, Le and Wang, Lingpeng and Cheng, Mingmei and Xie, Jin and Yang, Jian},
  booktitle={NeurIPS},
  year={2021}
}
```

## Acknowledgements

- Thank Qi for his implementation of [P2B](https://github.com/HaozheQi/P2B).
- Thank Pang for the [3D-SOT benchmark](https://arxiv.org/pdf/2103.06028.pdf) based on the waymo open dataset.

## License
This repository is released under MIT License.

