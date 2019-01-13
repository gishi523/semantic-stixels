#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "semantic_stixels.h"
#include "semi_global_matching.h"
#include "draw.h"

static std::string keys =
	"{ help  h      | | Print help message. }"
	"{ @left-image  | | Path to input left image or image sequence. }"
	"{ right-image  | | Path to input right image or image sequence. "
						"Either right-image or disparity must be specified. }"
	"{ disparity    | | Path to input disparity or disparity sequence. "
						"Format follows Cityscapes Dataset (https://github.com/mcordts/cityscapesScripts) "
						"Either right-image or disparity must be specified. }"
	"{ camera       | | Path to camera parameters. }"
	"{ model        | | Path to a binary file of model contains trained weights. }"
	"{ classes      | | Path to a text file with names of classes. }"
	"{ colors       | | Path to a text file with colors for an every class. }"
	"{ geometry     | | Path to a text file with geometry (0:ground 1:object 2:sky -1:any) for an every class. }"
	"{ width        | | Input image width for neural network. }"
	"{ height       | | Input image height for neural network. }"
	"{ downscale    | | Downscale disparity map. }"
	"{ wait-deley   | 1 | Deley time of cv::waitKey. }"
	"{ backend      | 0 | Choose one of computation backends: "
						"0: automatically (by default), "
						"1: Halide language (http://halide-lang.org/), "
						"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
						"3: OpenCV implementation }"
	"{ target       | 0 | Choose one of target computation devices: "
						"0: CPU target (by default), "
						"1: OpenCL, "
						"2: OpenCL fp16 (half-float precision), "
						"3: VPU }";

static void convertTo8bitGray(const cv::Mat& src, cv::Mat& dst)
{
	const int nchannels = src.channels();
	CV_Assert(nchannels == 1 || nchannels == 3);

	if (src.channels() == 3)
		cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	else
		src.copyTo(dst);

	if (dst.type() == CV_16U)
	{
		cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
		dst.convertTo(dst, CV_8U);
	}
}

static void convertToDisparity(const cv::Mat& src, cv::Mat& dst, int maxd)
{
	CV_Assert(src.depth() == CV_16U);

	cv::Mat tmp;
	if (src.channels() == 3)
		cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
	else
		tmp = src;

	const float denom = 1.f / 256;
	maxd = 256 * maxd + 1;

	dst.create(src.size(), CV_32F);
	for (int y = 0; y < dst.rows; y++)
	{
		const ushort* srcptr = tmp.ptr<ushort>(y);
		float* dstptr = dst.ptr<float>(y);
		for (int x = 0; x < dst.cols; x++)
		{
			const ushort p = srcptr[x];
			dstptr[x] = p > 0 && p < maxd ? denom * (p - 1) : -1.f;
		}
	}
}

static void softmax(const cv::Mat& score)
{
	const int chns = score.size[1];
	const int rows = score.size[2];
	const int cols = score.size[3];

	cv::Mat denom(rows, cols, CV_32F);
	std::vector<cv::Mat> expScores(chns);

	for (int ch = 0; ch < chns; ch++)
	{
		cv::Mat channel(rows, cols, CV_32F, (void*)score.ptr<float>(0, ch));
		cv::Mat& expScore = expScores[ch];
		cv::exp(channel, expScore);

		if (ch == 0)
			expScore.copyTo(denom);
		else
			cv::add(denom, expScore, denom);
	}

	for (int ch = 0; ch < chns; ch++)
	{
		cv::Mat channel(rows, cols, CV_32F, (void*)score.ptr<float>(0, ch));
		cv::Mat& expScore = expScores[ch];
		cv::divide(expScore, denom, channel);
	}
}

std::vector<std::string> readClasses(const std::string& file)
{
	std::ifstream ifs(file.c_str());
	if (ifs.fail())
		CV_Error(cv::Error::StsError, "File " + file + " not found");

	std::vector<std::string> classes;
	std::string line;
	while (std::getline(ifs, line))
		classes.push_back(line);
	return classes;
}

std::vector<int> readGeometry(const std::string& file)
{
	std::ifstream ifs(file);
	if (ifs.fail())
		CV_Error(cv::Error::StsError, "File " + file + " not found");

	std::vector<int> l2g;
	std::string line;
	while (std::getline(ifs, line))
	{
		std::istringstream ss(line);
		int g;
		ss >> g;
		l2g.push_back(g);
	}
	return l2g;
}

std::vector<cv::Vec3b> readColors(const std::string& file)
{
	std::ifstream ifs(file);
	if (ifs.fail())
		CV_Error(cv::Error::StsError, "File " + file + " not found");

	std::vector<cv::Vec3b> colors;
	std::string line;
	while (std::getline(ifs, line))
	{
		std::istringstream ss(line);
		int b, g, r;
		ss >> b >> g >> r;
		colors.push_back(cv::Vec3b(b, g, r));
	}
	return colors;
}

class Enet
{
public:

	Enet() {}

	Enet(const std::string& model, int width, int height, int backendId = 0, int targetId = 0)
	{
		read(model, width, height, backendId, targetId);
	}

	void read(const std::string& model, int width, int height, int backendId = 0, int targetId = 0)
	{
		scale = 0.00392f;
		mean = cv::Scalar::all(0);
		swapRB = true;
		blobSize = cv::Size(width, height);

		net = cv::dnn::readNet(model);
		net.setPreferableBackend(backendId);
		net.setPreferableTarget(targetId);

		std::cout << "input size:" << blobSize << std::endl;
	}

	void forward(const cv::Mat& frame, cv::Mat& score)
	{
		// Create a 4D blob from a frame
		cv::dnn::blobFromImage(frame, blob, scale, blobSize, mean, swapRB, false);

		// Set input blob
		net.setInput(blob);

		// Make forward pass
		net.forward(score);

		// Apply softmax
		softmax(score);
	}

private:

	cv::Mat blob;
	float scale;
	cv::Scalar mean;
	bool swapRB;
	cv::Size blobSize;
	cv::dnn::Net net;
};

class SGMWrapper
{

public:

	SGMWrapper(int numDisparities)
	{
		SemiGlobalMatching::Parameters param;
		param.numDisparities = numDisparities / 2;
		param.max12Diff = -1;
		param.medianKernelSize = -1;
		sgm_ = cv::Ptr<SemiGlobalMatching>(new SemiGlobalMatching(param));
	}

	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1, double scaleFactor = 1)
	{
		cv::pyrDown(I1, I1_);
		cv::pyrDown(I2, I2_);

		sgm_->compute(I1_, I2_, D1_, D2_);

		scaleFactor *= 2;

		cv::resize(D1_, D1, cv::Size(), scaleFactor, scaleFactor, cv::INTER_CUBIC);
		cv::resize(D2_, D2, cv::Size(), scaleFactor, scaleFactor, cv::INTER_CUBIC);
		D1 *= scaleFactor;
		D2 *= scaleFactor;
		cv::medianBlur(D1, D1, 3);
		cv::medianBlur(D2, D2, 3);
		SemiGlobalMatching::LRConsistencyCheck(D1, D2, 5);
	}

private:
	cv::Mat I1_, I2_, D1_, D2_, D2;
	cv::Ptr<SemiGlobalMatching> sgm_;
};

int main(int argc, char** argv)
{
	const cv::CommandLineParser parser(argc, argv, keys);
	if (argc < 2 || parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	if (!parser.has("right-image") && !parser.has("disparity"))
		CV_Error(cv::Error::StsError, "Either right-image or disparity must be specified");

	const int hasDisaprity = parser.has("disparity");

	// input images
	const std::string cap1File = parser.get<std::string>("@left-image");
	const std::string cap2File = hasDisaprity ? parser.get<std::string>("disparity") : parser.get<std::string>("right-image");
	cv::VideoCapture cap1(cap1File);
	cv::VideoCapture cap2(cap2File);

	// stereo SGM
	const int numDisparities = 128;
	SGMWrapper sgm(numDisparities);

	// Enet
	const std::string model = parser.get<std::string>("model");
	const int width = parser.get<int>("width");
	const int height = parser.get<int>("height");
	const int backendId = parser.get<int>("backend");
	const int targetId = parser.get<int>("target");
	Enet enet(model, width, height, backendId, targetId);

	// read camera parameters
	const cv::FileStorage fs(parser.get<std::string>("camera"), cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	// semantic stixels
	SemanticStixelWorld::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.camera.height = fs["Height"];
	param.camera.tilt = fs["Tilt"];
	param.dmax = numDisparities;

	// read downscale parameters
	const bool downscale = parser.has("downscale");
	const float scaleFactor = downscale ? 0.5f : 1.f;
	if (downscale)
	{
		param.camera.fu *= scaleFactor;
		param.camera.fv *= scaleFactor;
		param.camera.u0 *= scaleFactor;
		param.camera.v0 *= scaleFactor;
	}

	// read classes and geometry
	const std::vector<std::string> classes = readClasses(parser.get<std::string>("classes"));
	const std::vector<int> l2g = readGeometry(parser.get<std::string>("geometry"));

	SemanticStixelWorld stixelWorld(l2g, param);
	std::vector<SemanticStixel> stixels;
	cv::Mat I1, I2, gray1, gray2;
	cv::Mat disparity, disparityColor;
	cv::Mat score;

	// draw params
	const auto colors = readColors(parser.get<std::string>("colors"));
	enum { DRAW_LABEL = 0, DRAW_DEPTH = 1 };
	int drawMode = DRAW_LABEL;
	const int deley = parser.get<int>("wait-deley");
	const bool putTime = false;
	cv::Mat legend;
	drawLegend(legend, classes, colors);

	for (int frameno = 0;; frameno++)
	{
		cap1 >> I1;
		cap2 >> I2;

		if (I1.empty() || I2.empty())
		{
			std::cout << "Hit any key to exit." << std::endl;
			cv::waitKey(0);
			break;
		}

		const auto t0 = std::chrono::steady_clock::now();

		if (hasDisaprity)
		{
			// convert disparity following the Cityscapes dataset format
			convertToDisparity(I2, disparity, numDisparities);

			if (downscale)
			{
				cv::resize(disparity, disparity, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
				disparity *= scaleFactor;
			}
		}
		else
		{
			// compute dispaliry by SGM
			convertTo8bitGray(I1, gray1);
			convertTo8bitGray(I2, gray2);
			sgm.compute(gray1, gray2, disparity, scaleFactor);
			disparity.convertTo(disparity, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
		}

		const auto t1 = std::chrono::steady_clock::now();

		// compute segmentation
		enet.forward(I1, score);

		const auto t2 = std::chrono::steady_clock::now();

		// compute semantic stixels
		stixelWorld.compute(disparity, score, stixels);

		const auto t3 = std::chrono::steady_clock::now();

		const auto duration01 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		const auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

		////////////////////////////////////////////////////////////////////////////////////
		// draw results
		////////////////////////////////////////////////////////////////////////////////////
		const cv::Size drawSize = disparity.size();
		const int drawW = drawSize.width;
		const int drawH = drawSize.height;
		cv::Mat draw(2 * drawH, 2 * drawW, CV_8UC3);
		cv::Mat roi0 = draw(cv::Rect(0, 0, drawW, drawH));
		cv::Mat roi1 = draw(cv::Rect(drawW, 0, drawW, drawH));
		cv::Mat roi2 = draw(cv::Rect(0, drawH, drawW, drawH));
		cv::Mat roi3 = draw(cv::Rect(drawW, drawH, drawW, drawH));

		cv::resize(I1, I1, drawSize);
		I1.copyTo(roi0);

		// colorize disparity
		disparity.convertTo(disparityColor, CV_8U, 255. / numDisparities);
		cv::applyColorMap(disparityColor, disparityColor, cv::COLORMAP_JET);
		disparityColor.setTo(cv::Scalar::all(0), disparity < 0);
		cv::resize(disparityColor, roi1, drawSize);

		// colorize segmentation
		cv::Mat segmentImg;
		colorizeSegmentation(score, colors, segmentImg);
		cv::resize(segmentImg, segmentImg, drawSize, 0, 0, cv::INTER_NEAREST);
		cv::addWeighted(I1, 0.1, segmentImg, 0.9, 0, roi2);

		// draw stixels
		cv::Mat stixelImg(disparity.size(), CV_8UC3);

		if (drawMode == DRAW_LABEL)
		{
			stixelImg = cv::Vec3b(255, 255, 255);
			for (const auto& stixel : stixels)
				drawStixel(stixelImg, stixel, colors[stixel.semanticId]);
			cv::resize(stixelImg, stixelImg, drawSize, 0, 0);
		}
		else
		{
			stixelImg = cv::Vec3b(0, 0, 0);
			for (const auto& stixel : stixels)
				if (stixel.geometricId == SemanticStixel::GEOMETRIC_ID_OBJ)
					drawStixel(stixelImg, stixel, dispToColor(stixel.disp, 64));
			cv::resize(stixelImg, stixelImg, drawSize, 0, 0);
		}
		cv::addWeighted(I1, 0.3, stixelImg, 0.7, 0, roi3);

		// put information
		putInfo(roi1, "dispaliry input", putTime ? 1e-3 * duration01 : -1);
		putInfo(roi2, "semantic input", putTime ? 1e-3 * duration12 : -1);
		putInfo(roi3, "semantic stixels", putTime ? 1e-3 * duration23 : -1);

		cv::imshow("Semantic Stixels Demo", draw);
		cv::imshow("Legend", legend);

		const char c = cv::waitKey(deley);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
		if (c == 'm')
			drawMode = !drawMode;
	}

	return 0;
}
