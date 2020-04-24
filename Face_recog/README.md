# Requirements:
- python 3.5 
- pytorch 1.4
- opencv 
- tensorboard (for Face Recognition -> will be changed to visdom)
- visdom


# Person identification using Face Recognition 
We mainly use ArcFace: Additive Angular Margin Loss for Deep Face Recognition method.
The code is based on [ArcFace-pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) implementation.

This repository is not completed.
We will update new training code and add Person re-identification for undetected face.

<img src="https://user-images.githubusercontent.com/37200420/77849986-86800e00-720a-11ea-82e7-59111a8963f4.JPG">
<img src="https://user-images.githubusercontent.com/37200420/77849994-8e3fb280-720a-11ea-9dde-c18e51001996.JPG" width="40%">
<img src="https://user-images.githubusercontent.com/37200420/77849995-90a20c80-720a-11ea-9d64-b567e2b56778.JPG" width="40%">

# How to use:
```
1. Prepare AnotherMissOh dataset -> run prepare_AnotherMissOh.py after set your json and visual data path
2. run train.py
3. 

```

# Data structure:
```
Face/
    imgs/
    	Anna/
	     file_name.jpg
	     file_name.jpg
	     file_name.jpg
	     ...
	Deogi/
	     file_name.jpg
	     file_name.jpg
	     file_name.jpg
	     ...
		
	...

```

# Person Detection
![entire_2](https://user-images.githubusercontent.com/37200420/80203446-22623580-8662-11ea-884e-bbe50c8dc1b6.png)

We mainly use YOLOv2.
The code is based on [YOLOv2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch) implementation.

We are constantly training about person detection.

# Dataset:
We used AnotherMissOh dataset.

Currently, only 1 episode of AnotherMissOh data was used for learning, but it will be added other episodes continuously.

All episodes have image files and annotated json file.

You could find AnotherMissOh dataset in [this link](https://drive.google.com/open?id=1jcAhHCmq3fyhJ9Ggm9EA1Tf_xT3Roe48)

After unzip, you set your data path in 'Yolo_v2_pytorch/src/anotherMissOh_dataset.py' as below

```
img_path = 'D:\PROPOSAL\VTT\data\AnotherMissOh\AnotherMissOh_images\AnotherMissOh01/'
json_dir = 'D:\PROPOSAL\VTT\data\AnotherMissOh\AnotherMissOh_Visual/AnotherMissOh01_visual.json'
```

# Data structure:
The data structure is the same as VOC and AnotherMissOh has the following structure, but we take care of it in the dataloader.


```
AnotherMissOh1/
	AnotherMissOh1_visual.json
	
	AnotherMissOh1/
	    001/
		0078/
		     IMAGE_0000004295.jpg
		     IMAGE_0000004303.jpg
		     IMAGE_0000004311.jpg
		     ...
		0079/
		     IMAGE_0000004370.jpg
		     IMAGE_0000004378.jpg
		     IMAGE_0000004386.jpg
		     ...

		...
	    002/

```
# Trained models
You could find all trained models in [this link](https://drive.google.com/drive/folders/1LvDpPkkZ_18Zhf70rXUDaLoGFp2x6M5G)

And make 'pre_model' folder and put the models.

# How to use:
```
1. Set your path of dataset
2. run train_main.py
3. 

```
