# semantic-stixels
An implementation of semantic stixel computation

![semantic-stixels](https://github.com/gishi523/semantic-stixels/wiki/images/semantic-stixels.png)

## Description
- An implementation of the Semantic Stixel computation based on [1][2].
	- Extracts Semantic Stixels from a dense disparity map and a pixel-level semantic scene labeling
	- Jointly infers geometric and semantic layout of traffic scenes
- For semantic segmentation, OpenCV DNN module and Enet is used

## References
- [1] Schneider, L., Cordts, M., Rehfeld, T., Pfeiffer, D., Enzweiler, M., Franke, U., ... & Roth, S. (2016, June). Semantic stixels: Depth is not enough. In 2016 IEEE Intelligent Vehicles Symposium (IV) (pp. 110-117). IEEE.
- [2] Cordts, M., Rehfeld, T., Schneider, L., Pfeiffer, D., Enzweiler, M., Roth, S., ... & Franke, U. (2017). The stixel world: A medium-level representation of traffic scenes. Image and Vision Computing, 68, 40-52.

## Requirement
- OpenCV (recommended latest version)
- OpenMP (optional)

## How to build
```
$ git clone https://github.com/gishi523/semantic-stixels.git
$ cd semantic-stixels
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to run
### Command-line arguments
```
Usage: semantic_stixels [params] image-format-1 image-format-2
	-h, --help
		print help message.
	image-format-1
		input left image sequence.
	image-format-2
		input right image or disparity sequence.
	--input-type (value:0)
        type of input image pair (0:left-right 1:left-disparity)
	--camera
		path to camera parameters.
	--start-number (value:1)
	    start frame number.
	--model
	    path to a binary file of model contains trained weights.
	--classes
	    path to a text file with names of classes.
	--colors
	    path to a text file with colors for each class.
	--geometry
	    path to a text file with geometry id (0:ground 1:object 2:sky) for each class.
	--width (value:1024)
		input image width for neural network.
	--height (value:512)
	    input image height for neural network.
	--backend (value:0)
		computation backend. see cv::dnn::Net::setPreferableBackend.
	--target (value:0)
	    target device. see cv::dnn::Net::setPreferableTarget.
	--depth-only
		compute without semantic segmentation.
	--sgm-scaledown
        scaledown sgm input images for speedup.
	--wait-deley (value:1)
        deley time of cv::waitKey.
```

### Example
#### Input left-image and right-image
```
cd semantic_stixels

./build/semantic_stixels \
path_to_left_images/stuttgart_00_000000_%06d_leftImg8bit.png \
path_to_right_images/stuttgart_00_000000_%06d_rightImg8bit.png \
--camera=camera_parameters/cityscapes.xml \
--model=enet/Enet-model-best.net \
--classes=enet/classes.txt \
--colors=enet/colors.txt \
--geometry=enet/geometry.txt \
--target=1
```

#### Input left-image and disparity
Pass `--input-type=1`.
```
cd semantic_stixels

./build/semantic_stixels \
path_to_left_images/munich_%06d_000019_leftImg8bit.png \
path_to_disparities/munich_%06d_000019_disparity.png \
--input-type=1 \
--camera=camera_parameters/cityscapes.xml \
--model=enet/Enet-model-best.net \
--classes=enet/classes.txt \
--colors=enet/colors.txt \
--geometry=enet/geometry.txt \
--target=1
```

#### Optional arguments
If you have manually built OpenCV DNN module with CUDA backend,
you can pass `DNN_BACKEND_CUDA(=5)` and `DNN_TARGET_CUDA(=6)` to run the semantic segmentation faster.
```
--backend=5 --target=6
```

With `--depth-only` argument, you can test slanted stixel computation with depth information only.

```
cd semantic_stixels

./build/semantic_stixels \

path_to_left_images/imgleft%09d.pgm \
path_to_right_images/imgright%09d.pgm \
--camera=camera_parameters/daimler_urban_seg.xml \
--depth-only
```
