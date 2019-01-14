#include "semantic_stixels.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "matrix.h"
#include "cost_function.h"

#define UNUSED(x) ((void)x)

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for schedule(dynamic))
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic)")
#endif
#else
#define OMP_PARALLEL_FOR
#endif

using CameraParameters = SemanticStixelWorld::CameraParameters;

struct Line
{
	Line(float a = 0, float b = 0) : a(a), b(b) {}
	Line(const cv::Point2f& pt1, const cv::Point2f& pt2)
	{
		a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
		b = -a * pt1.x + pt1.y;
	}
	float a, b;
};

// estimate road model from camera tilt and height
static Line calcRoadModelCamera(const CameraParameters& camera)
{
	const float sinTilt = sinf(camera.tilt);
	const float cosTilt = cosf(camera.tilt);
	const float a = (camera.baseline / camera.height) * cosTilt;
	const float b = (camera.baseline / camera.height) * (camera.fu * sinTilt - camera.v0 * cosTilt);
	return Line(a, b);
}

// estimate road model from v-disparity
static Line calcRoadModelVD(const cv::Mat1f& disparity, const CameraParameters& camera,
	int samplingStep = 2, int minDisparity = 10, int maxIterations = 32, float inlierRadius = 1, float maxCameraHeight = 5)
{
	const int w = disparity.rows;
	const int h = disparity.cols;

	// sample v-disparity points
	std::vector<cv::Point2f> points;
	points.reserve(h * w);
	for (int u = 0; u < w; u += samplingStep)
		for (int v = 0; v < h; v += samplingStep)
			if (disparity(u, v) >= minDisparity)
				points.push_back(cv::Point2f(static_cast<float>(h - 1 - v), disparity(u, v)));

	if (points.empty())
		return Line(0, 0);

	// estimate line by RANSAC
	cv::RNG random;
	Line bestLine;
	int maxInliers = 0;
	for (int iter = 0; iter < maxIterations; iter++)
	{
		// sample 2 points and get line parameters
		const cv::Point2f& pt1 = points[random.next() % points.size()];
		const cv::Point2f& pt2 = points[random.next() % points.size()];
		if (pt1.x == pt2.x)
			continue;

		const Line line(pt1, pt2);

		// estimate camera tilt and height
		const float tilt = atanf((line.a * camera.v0 + line.b) / (camera.fu * line.a));
		const float height = camera.baseline * cosf(tilt) / line.a;

		// skip if not within valid range
		if (height <= 0.f || height > maxCameraHeight)
			continue;

		// count inliers within a radius and update the best line
		int inliers = 0;
		for (const auto& pt : points)
			if (fabs(line.a * pt.x + line.b - pt.y) <= inlierRadius)
				inliers++;

		if (inliers > maxInliers)
		{
			maxInliers = inliers;
			bestLine = line;
		}
	}

	// apply least squares fitting using inliers around the best line
	double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
	int n = 0;
	for (const auto& pt : points)
	{
		const float x = pt.x;
		const float y = pt.y;
		const float yhat = bestLine.a * x + bestLine.b;
		if (fabs(yhat - y) <= inlierRadius)
		{
			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
			n++;
		}
	}

	const float a = static_cast<float>((n * sxy - sx * sy) / (n * sxx - sx * sx));
	const float b = static_cast<float>((sxx * sy - sxy * sx) / (n * sxx - sx * sx));
	return Line(a, b);
}

static void computeColumns(const cv::Mat1f& src, cv::Mat1f& dst, int stixelWidth, float verticalScaleDown = -1)
{
	const int w = src.cols / stixelWidth;
	const int h = src.rows;

	// compute horizontal median of each column
	dst.create(w, h);

	int v;
OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		std::vector<float> buf(stixelWidth);
		for (int u = 0; u < w; u++)
		{
			// compute horizontal median
			for (int du = 0; du < stixelWidth; du++)
				buf[du] = src(v, u * stixelWidth + du);
			std::sort(std::begin(buf), std::end(buf));
			const float m = buf[stixelWidth / 2];

			// disparities are stored in reverse order so that v = 0 points the bottom
			// and transposed for memory efficiency
			dst(u, h - 1 - v) = m;
		}
	}

	// scale down the image in height
	if (verticalScaleDown > 1.f)
		cv::resize(dst, dst, cv::Size(), 1. / verticalScaleDown, 1., cv::INTER_NEAREST);
}

static void computeColumns(const cv::Mat1f& src, cv::Mat1f& dst, const cv::Size& dstSize)
{
	const double stixelWidth = 1. * src.cols / dstSize.width;
	const int w = dstSize.width;
	const int h = src.rows;

	const int iStixelWidth = cvRound(stixelWidth);
	const int scaleQ16 = 1 << 16;
	const int stixelWidthQ16 = static_cast<int>(scaleQ16 * stixelWidth);

	// compute horizontal median of each column
	dst.create(w, h);

	int v;
OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		std::vector<float> buf(iStixelWidth);
		for (int u = 0, suQ16 = 0; u < w; u++, suQ16 += stixelWidthQ16)
		{
			// compute horizontal median
			const int su = suQ16 >> 16;
			for (int du = 0; du < iStixelWidth; du++)
				buf[du] = src(v, su + du);
			std::sort(std::begin(buf), std::end(buf));
			const float m = buf[iStixelWidth / 2];

			// disparities are stored in reverse order so that v = 0 points the bottom
			// and transposed for memory efficiency
			dst(u, h - 1 - v) = m;
		}
	}

	// scale down the image in height
	if (dstSize.height != h)
		cv::resize(dst, dst, cv::Size(dstSize.height, w), 0, 0, cv::INTER_NEAREST);

	CV_Assert(dst.cols == dstSize.height && dst.rows == dstSize.width);
}

void computeMinCostAndLabel(const std::vector<Matrixf>& costsLs, const std::vector<int>& l2g,
	float* minCosts, int* minLabels, int u, int vT, int vB = 0)
{
	CV_Assert(costsLs.size() == l2g.size());

	const int nlabels = static_cast<int>(costsLs.size());

	for (int c = 0; c < 3; c++)
	{
		minCosts[c] = std::numeric_limits<float>::max();
		minLabels[c] = 0;
	}

	for (int l = 0; l < nlabels; l++)
	{
		const Matrixf& costsL = costsLs[l];
		const float cost = vB > 0 ? costsL(u, vT) - costsL(u, vB - 1) : costsL(u, vT);

		// semantic label to geometric class
		const int g = l2g[l];
		if (g > 2)
			continue;

		if (g < 0)
		{
			for (int c = 0; c < 3; c++)
			{
				if (cost < minCosts[c])
				{
					minCosts[c] = cost;
					minLabels[c] = l;
				}
			}
		}
		else
		{
			if (cost < minCosts[g])
			{
				minCosts[g] = cost;
				minLabels[g] = l;
			}
		}
	}
}

SemanticStixelWorld::SemanticStixelWorld(const std::vector<int> l2g, const Parameters& param)
	: l2g_(l2g), param_(param)
{
	if (param.verticalScaleDown > 1.f)
	{
		// scale camera parameters
		const float invScale = 1.f / param.verticalScaleDown;
		param_.camera.v0 *= invScale;
		param_.camera.tilt *= invScale;
		param_.camera.height *= invScale;
	}
}

void SemanticStixelWorld::compute(const cv::Mat& disparity, const cv::Mat& score,
	std::vector<SemanticStixel>& stixels)
{
	//CV_Assert(disparity.cols == score.size[3] && disparity.rows == score.size[2]);

	const int stixelWidth = param_.stixelWidth;
	const int fnmax = static_cast<int>(param_.dmax);
	const float verticalScaleDown = param_.verticalScaleDown;

	//////////////////////////////////////////////////////////////////////////////
	// disparity pre-process
	//////////////////////////////////////////////////////////////////////////////

	// reduce and reorder disparity map
	cv::Mat1f columns;
	computeColumns(disparity, columns, stixelWidth, verticalScaleDown);

	const int w = columns.rows;
	const int h = columns.cols;

	//////////////////////////////////////////////////////////////////////////////
	// segmentation pre-process
	//////////////////////////////////////////////////////////////////////////////

	const int chns = score.size[1];
	const int rows = score.size[2];
	const int cols = score.size[3];
	const float wl = param_.wl;

	std::vector<cv::Mat1f> channels(chns);
	for (int ch = 0; ch < chns; ch++)
	{
		cv::Mat channel(rows, cols, CV_32F, (void*)score.ptr<float>(0, ch));
		computeColumns(channel, channels[ch], cv::Size(w, h));
	}

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	
	// get camera parameters
	CameraParameters camera = param_.camera;

	// compute road model (assumes planar surface)
	Line line;
	if (param_.roadEstimation == ROAD_ESTIMATION_AUTO)
	{
		line = calcRoadModelVD(columns, camera);

		// when AUTO mode, update camera tilt and height
		camera.tilt = atanf((line.a * camera.v0 + line.b) / (camera.fu * line.a));
		camera.height = camera.baseline * cosf(camera.tilt) / line.a;
	}
	else if (param_.roadEstimation == ROAD_ESTIMATION_CAMERA)
	{
		line = calcRoadModelCamera(camera);
	}
	else
	{
		CV_Error(cv::Error::StsInternal, "No such mode");
	}

	// compute expected ground disparity
	std::vector<float> groundDisparity(h);
	for (int v = 0; v < h; v++)
		groundDisparity[h - 1 - v] = line.a * v + line.b;

	// horizontal row from which road dispaliry becomes negative
	const float vhor = h - 1 + line.b / line.a;
	const int vH = std::min(static_cast<int>(vhor), h - 1);

	// create data cost function of each segment
	NegativeLogDataTermGrd dataTermG(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, param_.pInvD,
		camera, groundDisparity, vhor, param_.sigmaH, param_.sigmaA, verticalScaleDown);
	NegativeLogDataTermObj dataTermO(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO, param_.pInvD,
		camera, param_.deltaz);
	NegativeLogDataTermSky dataTermS(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS, param_.pInvD);

	// create prior cost function of each segment
	const int G = NegativeLogPriorTerm::G;
	const int O = NegativeLogPriorTerm::O;
	const int S = NegativeLogPriorTerm::S;
	NegativeLogPriorTerm priorTerm(h, vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, groundDisparity);

	// cost table
	Matrixf costTable(w, h, 3), dispTable(w, h);
	Matrix<cv::Point> indexTable(w, h, 3);
	Matrixi labelTable(w, h, 3);

	std::vector<Matrixf> costsSem(channels.size());

	int c;
OMP_PARALLEL_FOR
	for (c = 0; c < chns; c++)
	{
		const cv::Mat1f& channel = channels[c];
		CV_Assert(channel.rows == w && channel.cols == h);

		Matrixf& costs = costsSem[c];
		costs.create(w, h);

		for (int u = 0; u < w; u++)
		{
			const float* ptrChannel = channel.ptr<float>(u);
			float* ptrCost = costs.ptr<float>(u);

			float tmpSum = 0.f;
			for (int v = 0; v < h; v++)
			{
				const float prob = ptrChannel[v];
				const float cost = prob > 0.f ? -logf(prob) : 0.f;
				tmpSum += cost;
				ptrCost[v] = tmpSum;
			}
		}
	}

	// process each column
	int u;
OMP_PARALLEL_FOR
	for (u = 0; u < w; u++)
	{
		cv::Mat1f costTable_u(h, 3, costTable.ptr<float>(u));
		cv::Mat1f dispTable_u(h, 1, dispTable.ptr<float>(u));
		cv::Mat1i labelTable_u(h, 3, labelTable.ptr<int>(u));
		cv::Mat_<cv::Point> indexTable_u(h, 3, indexTable.ptr<cv::Point>(u));

		////////////////////////////////////////////////////////////////////////////////////////////
		// pre-computate LUT
		////////////////////////////////////////////////////////////////////////////////////////////

		// data cost LUT
		Matrixf costsG(h), costsO(h, fnmax), costsS(h), sum(h);
		Matrixi valid(h);

		float tmpSumG = 0.f;
		float tmpSumS = 0.f;
		std::vector<float> tmpSumO(fnmax, 0.f);

		float tmpSum = 0.f;
		int tmpValid = 0;

		for (int v = 0; v < h; v++)
		{
			// measured disparity
			const float d = columns(u, v);

			// pre-computation for ground costs
			tmpSumG += dataTermG(d, v);
			costsG(v) = tmpSumG;

			// pre-computation for sky costs
			tmpSumS += dataTermS(d);
			costsS(v) = tmpSumS;

			// pre-computation for object costs
			for (int fn = 0; fn < fnmax; fn++)
			{
				tmpSumO[fn] += dataTermO(d, fn);
				costsO(v, fn) = tmpSumO[fn];
			}

			// pre-computation for mean disparity of stixel
			if (d >= 0.f)
			{
				tmpSum += d;
				tmpValid++;
			}
			sum(v) = tmpSum;
			valid(v) = tmpValid;
		}

#define UPDATE_COST(C1, C2) \
		const float cost##C1##C2 = dataCost##C1 + priorTerm.get##C1##C2(vB, cvRound(d1), cvRound(d2)) + costTable_u(vB - 1, C2); \
		if (cost##C1##C2 < minCost##C1) \
		{ \
			minCost##C1 = cost##C1##C2; \
			minPos##C1 = cv::Point(C2, vB - 1); \
			minDisp##C1 = d1; \
			minLabel##C1 = minLabels[C1]; \
		} \

		////////////////////////////////////////////////////////////////////////////////////////////
		// compute cost tables
		//
		// for paformance optimization, loop is split at vhor and unnecessary computation is ommited
		////////////////////////////////////////////////////////////////////////////////////////////

		// process vT = 0 to vhor
		// in this range, the class sky is not evaluated
		for (int vT = 0; vT <= vH; vT++)
		{
			float minCostG, minCostO, minCostS;
			float minDispG, minDispO, minDispS;
			int minLabelG, minLabelO, minLabelS;
			cv::Point minPosG(G, 0), minPosO(O, 0), minPosS(S, 0);

			float minSemCosts[3];
			int minLabels[3];

			// process vB = 0
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = sum(vT) / std::max(valid(vT), 1);
				const int fn = cvRound(d1);

				// minimization over the semantic labels
				computeMinCostAndLabel(costsSem, l2g_, minSemCosts, minLabels, u, vT);

				// initialize minimum costs
				minCostG = costsG(vT)     + wl * minSemCosts[G] + priorTerm.getG0(vT);
				minCostO = costsO(vT, fn) + wl * minSemCosts[O] + priorTerm.getO0(vT);
				minCostS = N_LOG_0_0;
				minDispG = minDispO = minDispS = d1;

				minLabelG = minLabels[G];
				minLabelO = minLabels[O];
				minLabelS = minLabels[S];
			}

			for (int vB = 1; vB <= vT; vB++)
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = (sum(vT) - sum(vB - 1)) / std::max(valid(vT) - valid(vB - 1), 1);
				const float d2 = dispTable_u(vB - 1);
				const int fn = cvRound(d1);

				// minimization over the semantic labels
				computeMinCostAndLabel(costsSem, l2g_, minSemCosts, minLabels, u, vT, vB);

				// compute data terms costs
				const float dataCostG = costsG(vT) - costsG(vB - 1) + wl * minSemCosts[G];
				const float dataCostO = costsO(vT, fn) - costsO(vB - 1, fn) + wl * minSemCosts[O];

				// compute priors costs and update costs
				UPDATE_COST(G, G);
				UPDATE_COST(G, O);
				UPDATE_COST(O, G);
				UPDATE_COST(O, O);
			}

			costTable_u(vT, G) = minCostG;
			costTable_u(vT, O) = minCostO;
			costTable_u(vT, S) = minCostS;

			indexTable_u(vT, G) = minPosG;
			indexTable_u(vT, O) = minPosO;
			indexTable_u(vT, S) = minPosS;

			labelTable_u(vT, G) = minLabelG;
			labelTable_u(vT, O) = minLabelO;
			labelTable_u(vT, S) = minLabelS;

			dispTable_u(vT) = minDispO;

			UNUSED(minDispG);
			UNUSED(minDispS);
		}

		// process vT = vhor to h
		// in this range, the class ground is not evaluated
		for (int vT = vH + 1; vT < h; vT++)
		{
			float minCostG, minCostO, minCostS;
			float minDispG, minDispO, minDispS;
			int minLabelG, minLabelO, minLabelS;
			cv::Point minPosG(G, 0), minPosO(O, 0), minPosS(S, 0);

			float minSemCosts[3];
			int minLabels[3];

			// process vB = 0
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = sum(vT) / std::max(valid(vT), 1);
				const int fn = cvRound(d1);

				// minimization over the semantic labels
				computeMinCostAndLabel(costsSem, l2g_, minSemCosts, minLabels, u, vT);

				// initialize minimum costs
				minCostG = N_LOG_0_0;
				minCostO = costsO(vT, fn) + wl * minSemCosts[O] + priorTerm.getO0(vT);
				minCostS = N_LOG_0_0;
				minDispG = minDispO = minDispS = d1;

				minLabelG = minLabels[G];
				minLabelO = minLabels[O];
				minLabelS = minLabels[S];
			}

			// process vB = 1 to vH + 1
			// in this range, transition from sky is not allowed
			for (int vB = 1; vB <= std::min(vH + 1, vT); vB++)
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = (sum(vT) - sum(vB - 1)) / std::max(valid(vT) - valid(vB - 1), 1);
				const float d2 = dispTable_u(vB - 1);
				const int fn = cvRound(d1);

				// minimization over the semantic labels
				computeMinCostAndLabel(costsSem, l2g_, minSemCosts, minLabels, u, vT, vB);

				// compute data terms costs
				const float dataCostO = costsO(vT, fn) - costsO(vB - 1, fn) + wl * minSemCosts[O];
				const float dataCostS = costsS(vT) - costsS(vB - 1) + wl * minSemCosts[S];

				// compute priors costs and update costs
				UPDATE_COST(O, G);
				UPDATE_COST(O, O);
				UPDATE_COST(S, G);
				UPDATE_COST(S, O);
			}

			// process vB = vH + 2 to vT
			// in this range, transition from ground is not allowed
			for (int vB = vH + 2; vB <= vT; vB++)
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = (sum(vT) - sum(vB - 1)) / std::max(valid(vT) - valid(vB - 1), 1);
				const float d2 = dispTable_u(vB - 1);
				const int fn = cvRound(d1);

				// minimization over the semantic labels
				computeMinCostAndLabel(costsSem, l2g_, minSemCosts, minLabels, u, vT, vB);

				// compute data terms costs
				const float dataCostO = costsO(vT, fn) - costsO(vB - 1, fn) + wl * minSemCosts[O];
				const float dataCostS = costsS(vT) - costsS(vB - 1) + wl * minSemCosts[S];

				// compute priors costs and update costs
				UPDATE_COST(O, O);
				UPDATE_COST(O, S);
				UPDATE_COST(S, O);
			}

			costTable_u(vT, G) = minCostG;
			costTable_u(vT, O) = minCostO;
			costTable_u(vT, S) = minCostS;

			indexTable_u(vT, G) = minPosG;
			indexTable_u(vT, O) = minPosO;
			indexTable_u(vT, S) = minPosS;

			labelTable_u(vT, G) = minLabelG;
			labelTable_u(vT, O) = minLabelO;
			labelTable_u(vT, S) = minLabelS;

			dispTable_u(vT) = minDispO;

			UNUSED(minDispG);
			UNUSED(minDispS);
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	// backtracking step
	//////////////////////////////////////////////////////////////////////////////
	stixels.clear();
	for (int u = 0; u < w; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, h - 1, c);
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, h - 1);
			}
		}

		while (minPos.y > 0)
		{
			const cv::Point p1 = minPos;
			const cv::Point p2 = indexTable(u, p1.y, p1.x);
			const int c = p1.x;

			SemanticStixel stixel;
			stixel.u = stixelWidth * u + stixelWidth / 2;
			stixel.vT = h - 1 - p1.y;
			stixel.vB = h - 1 - (p2.y + 1);
			stixel.width = stixelWidth;
			stixel.disp = c == O ? dispTable(u, p1.y) : 0;
			stixel.geometricId = c;
			stixel.semanticId = labelTable(u, p1.y, p1.x);

			if (verticalScaleDown > 1.f)
			{
				stixel.vT = cvRound(verticalScaleDown * stixel.vT);
				stixel.vB = cvRound(verticalScaleDown * stixel.vB);
			}

			stixels.push_back(stixel);

			minPos = p2;
		}
	}
}
