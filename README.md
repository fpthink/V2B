# 3D Siamese Voxel-to-BEV Tracker for Sparse Point Clouds

## Introduction

This repository is released for V2B in our [NeurIPS 2021 paper (poster)](https://arxiv.org/pdf/2111.04426.pdf). Here we include our V2B model (PyTorch) and code for data preparation, training and testing on KITTI tracking dataset.
We also provide the conversion code for the nuScenes data set, which is modified on the basis of the official code from nuScenes to KITTI.

## Preliminary
* conda 
```
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0
```

* Install dependencies.
```
    pip install -r requirements.txt
```

* Build `_ext` module.
```
   cd lib/pointops && python setup.py install && cd ../../
```

* Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)

	Download [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) in the dataset and place them under the same parent folder.
* Download the Full dataset (v1.0) from [nuScenes](https://www.nuscenes.org/).
  
    Note that base on the offical code [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit), we modify and use it to convert nuScenes format to KITTI format. It requires metadata from nuScenes-lidarseg. Thus, you should replace *category.json* and *lidarseg.json* in the Full dataset (v1.0). We provide these two json files in the nuscenes_json folder.

    Executing the following code to convert nuScenes format to KITTI format
    ```
    cd nuscenes-devkit-master/python-sdk/nuscenes/scripts
    python export_kitti.py --nusc_dir=<nuScenes dataset path> --nusc_kitti_dir=<output dir> --split=<dataset split>
    ```

    Note that the parameter of "split" should be "train_track" or "val". In our paper, we use the model trained on the KITTI dataset to evaluate the generalization of the model on the nuScenes dataset.
	

## Evaluation

Train a new model on KITTI or nuScenes data:
```
python train_tracking.py --which_dataset=<dataset type> --data_dir=<data path> 
```

Test a new model on KITTI or nuScenes data:
```
python test_tracking.py --which_dataset=<dataset type> --data_dir=<data path> --model_dir=<model output folder path> --model=<model name>
```

Please refer to the code for setting of other optional arguments, including data split, training and testing parameters, etc.

## ToDO

```
- Provide pretrained models and visualization codes
- Results on the Waymo dataset.
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

Thank Qi for his implementation of [P2B](https://github.com/HaozheQi/P2B).

