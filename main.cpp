#include <iostream>
#include <fstream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "semi_global_matching.h"
#include "semantic_stixels.h"
#include "draw.h"

static std::string keys =
"{ help  h         |   | print help message. }"
"{ @image-format-1 |   | input left image sequence. }"
"{ @image-format-2 |   | input right image or disparity sequence. }"
"{ input-type      | 0 | type of input image pair (0:left-right 1:left-disparity) }"
"{ camera          |   | path to camera parameters. }"
"{ start-number    | 1 | start frame number. }"
"{ model           |   | path to a binary file of model contains trained weights. }"
"{ classes         |   | path to a text file with names of classes. }"
"{ colors          |   | path to a text file with colors for each class. }"
"{ geometry        |   | path to a text file with geometry id (0:ground 1:object 2:sky) for each class. }"
"{ width           | 1024 | input image width for neural network. }"
"{ height          |  512 | input image height for neural network. }"
"{ backend         | 0 | computation backend. see cv::dnn::Net::setPreferableBackend. }"
"{ target          | 0 | target device. see cv::dnn::Net::setPreferableTarget. }"
"{ depth-only      |   | compute without semantic segmentation. }"
"{ sgm-scaledown   |   | scaledown sgm input images for speedup. }"
"{ wait-deley      | 1 | deley time of cv::waitKey. }";

class SGMWrapper
{
public:

	SGMWrapper(int numDisparities, bool scaleDown = false) : scaleDown_(scaleDown)
	{
		SemiGlobalMatching::Parameters param;
		param.numDisparities = scaleDown ? numDisparities / 2 : numDisparities;
		param.medianKernelSize = -1;
		param.max12Diff = 1;
		param.pathType = SemiGlobalMatching::SCAN_4PATH;
		sgm_ = cv::makePtr<SemiGlobalMatching>(param);
	}

	static void convertToGray(const cv::Mat& src, cv::Mat& dst)
	{
		if (src.type() == CV_8UC3)
			cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
		else
			dst = src;
	}

	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1)
	{
		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());

		convertToGray(I1, G1_);
		convertToGray(I2, G2_);

		if (scaleDown_)
		{
			cv::pyrDown(G1_, I1_);
			cv::pyrDown(G2_, I2_);
			sgm_->compute(I1_, I2_, D1_, D2_);
			D1_.convertTo(Df_, CV_32F, 2. / SemiGlobalMatching::DISP_SCALE);
			cv::resize(Df_, D1, I1.size(), 0, 0, cv::INTER_NEAREST);
		}
		else
		{
			sgm_->compute(G1_, G2_, D1_, D2_);
			D1_.convertTo(D1, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
		}
	}

private:

	cv::Mat G1_, G2_, I1_, I2_, D1_, D2_, Df_;
	cv::Ptr<SemiGlobalMatching> sgm_;
	bool scaleDown_;
};

class InferenceEngine
{
public:

	InferenceEngine() {}

	void init(const std::string& model, int backendId = 0, int targetId = 0)
	{
		net_ = cv::dnn::readNet(model);
		net_.setPreferableBackend(backendId);
		net_.setPreferableTarget(targetId);
	}

	void infer(const cv::Mat& src, cv::Mat& dst)
	{
		cv::dnn::blobFromImage(src, tensor_, 1. / 255, cv::Size(), cv::Scalar(), true, false);
		net_.setInput(tensor_);
		net_.forward(dst);

		// squeeze batch dim [1,20,1024,512] -> [20,1024,512]
		dst = cv::Mat(3, &dst.size.p[1], CV_32F, dst.data);
	}

private:

	cv::dnn::Net net_;
	cv::Mat tensor_;
};

static void preprocessImage(cv::Mat& image, cv::Size inputSize, bool isDisparity = false)
{
	if (!isDisparity && image.type() == CV_16U)
		cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8U);

	if (image.size() != inputSize)
		cv::resize(image, image, inputSize, 0, 0, isDisparity ? cv::INTER_NEAREST : cv::INTER_LINEAR);
}

static void convertToDisparity(const cv::Mat& src, cv::Mat& dst, int maxd, float scaleFactor = 1)
{
	CV_Assert(src.depth() == CV_16U);
	CV_Assert(src.channels() == 1);

	const float scale = scaleFactor / 256;

	dst.create(src.size(), CV_32F);
	for (int y = 0; y < dst.rows; y++)
	{
		const ushort* ptrSrc = src.ptr<ushort>(y);
		float* ptrDst = dst.ptr<float>(y);
		for (int x = 0; x < dst.cols; x++)
		{
			const int p = ptrSrc[x];
			const float d = scale * (p - 1);
			ptrDst[x] = p > 0 && d < maxd ? d : -1;
		}
	}
}

void scaleCameraParams(CameraParameters& camera, float factorx, float factory)
{
	camera.fu *= factorx;
	camera.fv *= factory;
	camera.u0 *= factorx;
	camera.v0 *= factory;
}

static CameraParameters readCameraParams(const std::string& filename)
{
	const cv::FileStorage fs(filename, cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	CameraParameters camera;
	camera.fu = fs["FocalLengthX"];
	camera.fv = fs["FocalLengthY"];
	camera.u0 = fs["CenterX"];
	camera.v0 = fs["CenterY"];
	camera.baseline = fs["BaseLine"];
	camera.height = fs["Height"];
	camera.tilt = fs["Tilt"];
	return camera;
}

template <class T = int, class GetValue>
static std::vector<T> readLines(const std::string& file, GetValue getValue)
{
	std::ifstream ifs(file);
	if (ifs.fail())
		CV_Error(cv::Error::StsError, "File " + file + " not found");
	std::vector<T> values;
	std::string line;
	while (std::getline(ifs, line))
		values.push_back(getValue(line));
	return values;
}

static std::vector<std::string> readClasses(const std::string& file)
{
	return readLines<std::string>(file, [](const std::string& line) { return line; });
}

static std::vector<cv::Vec3b> readColors(const std::string& file)
{
	return readLines<cv::Vec3b>(file, [](const std::string& line)
	{
		std::istringstream ss(line);
		int b, g, r;
		ss >> b >> g >> r;
		return cv::Vec3b(b, g, r);
	});
}

static std::vector<int> readGeometry(const std::string& file)
{
	return readLines<int>(file, [](const std::string& line) { return std::stoi(line); });
}

int main(int argc, char** argv)
{
	const cv::CommandLineParser parser(argc, argv, keys);
	if (argc < 2 || parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	const std::string format1 = parser.get<std::string>("@image-format-1");
	const std::string format2 = parser.get<std::string>("@image-format-2");
	const int startno = parser.get<int>("start-number");

	// check input image
	cv::Mat I1 = cv::imread(cv::format(format1.c_str(), startno), cv::IMREAD_UNCHANGED);
	cv::Mat I2 = cv::imread(cv::format(format2.c_str(), startno), cv::IMREAD_UNCHANGED);
	CV_Assert(!I1.empty() && !I2.empty() && I1.size() == I2.size());

	enum { INPUT_LEFT_AND_RIGHT = 0, INPUT_LEFT_AND_DISP };
	const int inputType = parser.get<int>("input-type");
	if (inputType == INPUT_LEFT_AND_RIGHT) CV_Assert(I1.type() == I2.type());
	if (inputType == INPUT_LEFT_AND_DISP)  CV_Assert(I2.type() == CV_16U);

	// setup stereo matching
	const int numDisparities = 64;
	SGMWrapper sgm(numDisparities, parser.has("sgm-scaledown"));
	cv::Mat disparity, disparityColor;

	// get input size
	const cv::Size inputSize(parser.get<int>("width"), parser.get<int>("height"));
	const float factorx = 1.f * inputSize.width / I1.cols;
	const float factory = 1.f * inputSize.height / I1.rows;

	// setup semantic stixels
	SemanticStixels::Parameters param;
	param.dmax = numDisparities;
	param.camera = readCameraParams(parser.get<std::string>("camera"));
	scaleCameraParams(param.camera, factorx, factory);

	// setup track bar
	const int stixelWidth[2] = { 4, 8 };
	const int stixelYResolution[2] = { 4, 8 };
	int stixelW = 1, stixelH = 0;
	cv::namedWindow("trackbar");
	cv::resizeWindow("trackbar", cv::Size(512, 256));
	cv::createTrackbar("stixel W", "trackbar", &stixelW, 1);
	cv::createTrackbar("stixel H", "trackbar", &stixelH, 1);

	// setup semantic segmentation, if enabled
	const bool withSemantic = !parser.has("depth-only");
	InferenceEngine net;
	cv::Mat predict, predictColor, legend;
	ColorTable colors;
	std::vector<std::string> classes;
	if (withSemantic)
	{
		CV_Assert(I1.type() == CV_8UC3);

		const std::string model = parser.get<std::string>("model");
		const int backendId = parser.get<int>("backend");
		const int targetId = parser.get<int>("target");
		net.init(model, backendId, targetId);

		classes = readClasses(parser.get<std::string>("classes"));
		colors = readColors(parser.get<std::string>("colors"));
		param.geometry = readGeometry(parser.get<std::string>("geometry"));

		drawLegend(legend, classes, colors);
		cv::imshow("legend", legend);
	}

	// setup semantic stixels
	auto stixelWorld = SemanticStixels::create(param);
	std::vector<Stixel> stixels;
	cv::Mat drawDepth, drawSemantic;

	const int deley = parser.get<int>("wait-deley");

	for (int frameno = startno;; frameno++)
	{
		I1 = cv::imread(cv::format(format1.c_str(), frameno), cv::IMREAD_UNCHANGED);
		I2 = cv::imread(cv::format(format2.c_str(), frameno), cv::IMREAD_UNCHANGED);
		if (I1.empty() || I2.empty())
		{
			std::cout << "imread failed." << std::endl;
			frameno = startno - 1;
			continue;
		}

		preprocessImage(I1, inputSize);
		preprocessImage(I2, inputSize, inputType == INPUT_LEFT_AND_DISP);

		const auto t1 = std::chrono::steady_clock::now();

		// compute dispaliry
		if (inputType == INPUT_LEFT_AND_RIGHT)
			sgm.compute(I1, I2, disparity);
		else if (inputType == INPUT_LEFT_AND_DISP)
			convertToDisparity(I2, disparity, numDisparities, factorx);

		const auto t2 = std::chrono::steady_clock::now();

		// semantic segmentation
		if (withSemantic)
			net.infer(I1, predict);

		const auto t3 = std::chrono::steady_clock::now();

		// compute semantic stixels
		param.stixelWidth = stixelWidth[stixelW];
		param.stixelYResolution = stixelYResolution[stixelH];
		stixelWorld->setParameters(param);

		if (withSemantic)
			stixelWorld->compute(disparity, predict, stixels);
		else
			stixelWorld->compute(disparity, stixels);

		const auto t4 = std::chrono::steady_clock::now();

		using duration = std::chrono::microseconds;
		const auto duration12 = std::chrono::duration_cast<duration>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<duration>(t3 - t2).count();
		const auto duration34 = std::chrono::duration_cast<duration>(t4 - t3).count();

		// colorize disparity and confidence map
		colorizeDisparity(disparity, disparityColor, numDisparities);

		// colorize segmentation
		if (withSemantic)
		{
			colorizeSegmentation(predict, predictColor, colors);
			cv::addWeighted(I1, 0.5, predictColor, 0.5, 0, predictColor);
		}

		// draw stixels
		drawDepthStixels(I1, stixels, drawDepth, numDisparities);
		if (withSemantic)
			drawSemanticStixels(I1, stixels, drawSemantic, colors);

		// put time
		cv::putText(disparityColor, cv::format("dispaliry time: %4.1f [msec]", 1e-3 * duration12),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(disparityColor, cv::format("inference time: %4.1f [msec]", 1e-3 * duration23),
			cv::Point(50, 80), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(disparityColor, cv::format("semantic-stixel time: %4.1f [msec]", 1e-3 * duration34),
			cv::Point(50, 110), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("depth iput", disparityColor);
		cv::imshow("semantic stixels (depth representation)", drawDepth);
		if (withSemantic)
		{
			cv::imshow("semantic iput", predictColor);
			cv::imshow("semantic stixels (semantic representation)", drawSemantic);
		}

		const char c = cv::waitKey(deley);
		if (c == 27)
			break;
	}

	return 0;
}
