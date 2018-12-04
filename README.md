![tracking](https://user-images.githubusercontent.com/37200420/49422450-3f7d3680-f7d7-11e8-8d4d-52739dcae85c.JPG)

![caption2](https://user-images.githubusercontent.com/37200420/48906820-04ab0280-eea9-11e8-8d53-ff05c047fe14.png)

# VTT_TRACKING

This repository is for VTT tracking research results.
Even if a scene change occurs in the video, it gives the same ID.
The configuration is largely based on multi-object tracking using re-id method and the image2text method is a separate module.

We mainly use Image-Text-Embedding method and Person ReID baseline.

current code borrows heavily from Image-Text-Embedding. The images were taken from CUHK PEDES dataset.

# Prerequisites

- NVIDIA GPU + CUDA + CuDNN
- Matconvnet (Unzip matlab) + Matlab 2017b
- Pytorch 0.4 + Python 3.6
- Install requirements

# Preprocess Datasets

- For Visual Tracking
 1. Unzip friends2.zip and after [download](https://drive.google.com/open?id=1AZMyo5rwsS_KECYAEckJ5bE3Q64eKrYR) unzip into /MOT_Re-Id/
 2. Download pre-trained model [Download](https://drive.google.com/open?id=1gD2-8vfV-DzdgyKBktW1CNQYM4ayeFtp)
 3. Locate the pre-trained model into /MOT_Re-Id/model/ft_resNet50/
 
- For Image2Text
 1. Download [GoogleNews](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
 2. Download [CUHK-PEDES](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/CUHK-PEDES-prepare)
 3. Pre-trained model (currently uploading in progress)

# Useage

- Visual Tracking

dataset structre: 
```
/MOT_Re-Id/Friends
		└ ep1
		  └ gallery
		    └ 0001 (frame), 0002, 0003, ....
		      └ data
		        └ 0001.png, 0002.png, ... (detection results)
```

run /MOR_Re-Id/MOT_reid.py
 
- Image2Text

run src/find_pic_feature_word2_plus

# Output
tracker_results.json has tracking coordinates.

coordinates information is as follows.

"coordinates" : x1, y1, x2, y2, id_number

```
{
	"dataset": "Friends_EP1",
	"coordinates": [
		[
			252, 338, 584, 819, 1
		],
		[
			688, 376, 951, 748, 2
		],
		[
  ...
}
```

### Todos

 - Write MORE example 
 - Currently uploading in progress
 
 
# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)

License
----

MIT
