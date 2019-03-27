# semantic-stixels
An implementation of semantic stixel computation

![semantic-stixels](https://github.com/gishi523/semantic-stixels/wiki/images/semantic-stixels.png)

## Description
- An implementation of the Semantic Stixel computation based on [1].
- Extracts Semantic Stixels from a dense disparity map and a pixel-level semantic scene labeling
- Jointly infers geometric and semantic layout of traffic scenes
- For easy to use, OpenCV dnn module and Enet model is used as implementation of semantic segmentation

## References
- [1] L. Schneider, M. Cordts, T. Rehfeld, D. Pfeiffer, M. Enzweiler, U. Franke, M. Pollefeys, S. Roth, "Semantic Stixels: Depth is not enough", in: IEEE Intelligent Vehicles Symposium, 2016.

## Demo
- <a href="https://youtu.be/XdTRI3HrYjc" target="_blank">Semantic Stixel Computation Demo</a>

## Requirement
- OpenCV (version 3.3 or higher, with dnn module)
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
Usage: semantic_stixels left-image [params]
	-h, --help
		Print help message.
	left-image
		Path to input left image or image sequence.
	--right-image
		Path to input right image or image sequence. Either right-image or disparity must be specified.
	--disparity
		Path to input disparity or disparity sequence. Format follows Cityscapes Dataset (https://github.com/mcordts/cityscapesScripts) Either right-image or disparity must be specified.
	--camera
		Path to camera parameters.
	--model
		Path to a binary file of model contains trained weights.
	--classes
		Path to a text file with names of classes.
	--colors
		Path to a text file with colors for an every class.
	--geometry
		Path to a text file with geometry (0:ground 1:object 2:sky -1:any) for an every class.
	--width
		Input image width for neural network.
	--height
		Input image height for neural network.
	--downscale
		Downscale disparity map.
	--wait-deley (value:1)
		Deley time of cv::waitKey.
	--backend (value:0)
		Choose one of computation backends: 0: automatically (by default), 1: Halide language (http://halide-lang.org/), 2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), 3: OpenCV implementation
	--target (value:0)
		Choose one of target computation devices: 0: CPU target (by default), 1: OpenCL, 2: OpenCL fp16 (half-float precision), 3: VPU
```

### Example
#### 1. KITTI (left-image and right-image)
```
cd semantic_stixels

./build/semantic_stixels kitti/2011_09_26_drive_0011_sync/image_02/data/%010d.png \
--right-image=kitti/2011_09_26_drive_0011_sync/image_03/data/%010d.png \
--camera=camera_parameters/kitti.xml \
--model=enet/Enet-model-best.net \
--classes=enet/classes.txt --colors=enet/colors.txt --geometry=enet/geometry.txt \
--width=1024 --height=512 --target=1
```

#### 2. Cityscapes (left-image and disparity)
```
cd semantic_stixels

./build/semantic_stixels cityscapes/munich/munich_%06d_000019_leftImg8bit.png \
--disparity=cityscapes/munich_disparity/munich_%06d_000019_disparity.png \
--camera=camera_parameters/cityscapes.xml \
--model=enet/Enet-model-best.net \
--classes=enet/classes.txt --colors=enet/colors.txt --geometry=enet/geometry.txt \
--width=1024 --height=512 --target=1 --downscale --wait-deley=0
```
