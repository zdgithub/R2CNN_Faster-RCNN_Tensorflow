# R2CNN_Faster_RCNN_Tensorflow

## Abstract
This is a tensorflow re-implementation of [R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/abs/1706.09579).      

## Addition
This project has been modified from the forked source [R2CNN_Faster_RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow) to be consistant with the [R2CNN](https://arxiv.org/abs/1706.09579) paper.

## Requirements
1、tensorflow >= 1.2     
2、cuda8.0     
3、python2.7 (anaconda2 recommend)    
4、[opencv(cv2)](https://pypi.org/project/opencv-python/) 

## Download Model
1、please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to data/pretrained_weights.     
2、please download [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) pre-trained model on Imagenet, put it to data/pretrained_weights/mobilenet.      
3、please download [trained model](https://github.com/DetectionTeamUCAS/Models/tree/master/R2CNN_Faster-RCNN_Tensorflow) by this project, put it to output/trained_weights.

## Data Prepare
1、please download [DOTA](https://captain-whu.github.io/DOTA/dataset.html)      
2、crop data, reference:
```  
cd $PATH_ROOT/data/io/DOTA
python train_crop.py 
python val_crop.py
```
3、data format
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│    ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```  

## Compile
```  
cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```

```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace
```

## Demo

**Select a configuration file in the folder (libs/configs/) and copy its contents into cfgs.py, then download the corresponding [weights](https://github.com/DetectionTeamUCAS/Models/tree/master/R2CNN_Faster-RCNN_Tensorflow).** 

### DOTA     

```   
python demo_rh.py --src_folder='/PATH/TO/DOTA/IMAGES_ORIGINAL/' 
                  --image_ext='.png' 
                  --des_folder='/PATH/TO/SAVE/RESULTS/' 
                  --save_res=False
                  --gpu='0'
```

### FDDB
```   
python camera_demo.py --gpu='0'
```

## Eval
```  
python eval.py --img_dir='/PATH/TO/DOTA/IMAGES/' 
               --image_ext='.png' 
               --test_annotation_path='/PATH/TO/TEST/ANNOTATION/'
               --gpu='0'
```

## Inference
```  
python inference.py --data_dir='/PATH/TO/DOTA/IMAGES_CROP/'      
                    --gpu='0'
```

## Train
1、If you want to train your own data, please note:  
```     
(1) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/cfgs.py
(2) Add category information in $PATH_ROOT/libs/label_name_dict/lable_dict.py     
(3) Add data_name to line 75 of $PATH_ROOT/data/io/read_tfrecord.py 
```     

2、make tfrecord
```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/VOCdevkit/VOCdevkit_train/' 
                                   --xml_dir='Annotation'
                                   --image_dir='JPEGImages'
                                   --save_name='train' 
                                   --img_format='.png' 
                                   --dataset='DOTA'
```     

3、train
```  
cd $PATH_ROOT/tools
python train.py
```

## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
``` 

## Citation
Some relevant achievements based on this code.     

    @article{[yang2018position](https://ieeexplore.ieee.org/document/8464244),
		title={Position Detection and Direction Prediction for Arbitrary-Oriented Ships via Multitask Rotation Region Convolutional Neural Network},
		author={Yang, Xue and Sun, Hao and Sun, Xian and  Yan, Menglong and Guo, Zhi and Fu, Kun},
		journal={IEEE Access},
		volume={6},
		pages={50839-50849},
		year={2018},
		publisher={IEEE}
	}
    
    @article{[yang2018r-dfpn](http://www.mdpi.com/2072-4292/10/1/132),
		title={Automatic ship detection in remote sensing images from google earth of complex scenes based on multiscale rotation dense feature pyramid networks},
		author={Yang, Xue and Sun, Hao and Fu, Kun and Yang, Jirui and Sun, Xian and Yan, Menglong and Guo, Zhi},
		journal={Remote Sensing},
		volume={10},
		number={1},
		pages={132},
		year={2018},
		publisher={Multidisciplinary Digital Publishing Institute}
	} 
