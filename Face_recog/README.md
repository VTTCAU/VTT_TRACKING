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
We mainly use YOLOv2.
The code is based on [YOLOv2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch) implementation.

We are constantly training about person detection.

# Dataset:
We used AnotherMissOh dataset.

Currently, only 1 episode of AnotherMissOh data was used for learning, but it will be added other episodes continuously.

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
