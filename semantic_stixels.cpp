#include "semantic_stixels.h"
#include "draw.h"

#define _USE_MATH_DEFINES
#include <math.h>

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

////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////

// maximum cost
static constexpr float Cinf = std::numeric_limits<float>::max();

// model complexity prior
static constexpr float Cmc = 10;

// mathematical constants
static constexpr float PI = static_cast<float>(M_PI);
static constexpr float SQRT2 = static_cast<float>(M_SQRT2);
static constexpr float SQRTPI = static_cast<float>(2. / M_2_SQRTPI);

// geometric class id
static constexpr int G = GEOMETRIC_ID_GROUND;
static constexpr int O = GEOMETRIC_ID_OBJECT;
static constexpr int S = GEOMETRIC_ID_SKY;
static constexpr int GEO_ID_BIT = 2;
static constexpr int NO_GEOMETRY = 3;

// structural priors
static constexpr float alphaGapP  = 1;
static constexpr float betaGapP   = 0;
static constexpr float alphaGapN  = 1;
static constexpr float betaGapN   = 0;
static constexpr float alphaGravP = 1;
static constexpr float betaGravP  = 1;
static constexpr float alphaGravN = 1;
static constexpr float betaGravN  = 1;
static constexpr float alphaOrd   = 1;
static constexpr float betaOrd    = 1;

// disparity measurement uncertainty
static constexpr float sigmaD[3] =
{
	1.f,
	1.f,
	2.f
};

static constexpr float sigmaDSq[3] =
{
	sigmaD[G] * sigmaD[G],
	sigmaD[O] * sigmaD[O],
	sigmaD[S] * sigmaD[S]
};

static constexpr float invSigmaDSq[3] =
{
	1.f / sigmaDSq[G],
	1.f / sigmaDSq[O],
	1.f / sigmaDSq[S]
};

// range of depth into witch objects are allowed to extend
static constexpr float deltaZ = 0.3f;

// camera height and tilt uncertainty
constexpr float sigmaH = 0.05f;
constexpr float sigmaA = 0.005f;
constexpr float sigmaHSq = sigmaH * sigmaH;
constexpr float sigmaASq = sigmaA * sigmaA;

// outlier rate
static constexpr float pOutG = 0.15f;
static constexpr float pOutO = 0.15f;
static constexpr float pOutS = 0.4f;

// probability of invalid disparity
static constexpr float pInvD = 0.25f;
static constexpr float pInvG = 0.34f;
static constexpr float pInvO = 0.3f;
static constexpr float pInvS = 0.36f;
static constexpr float pC = 1.f / 3;

// semantic cost weight
static constexpr float wsem = 0.5f;

static constexpr float squared(float x)
{
	return x * x;
}

template <typename T>
static constexpr float floatCast(T x)
{
	return static_cast<float>(x);
}

////////////////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////////////////

struct Line
{
	Line(float a = 0, float b = 0) : a(a), b(b) {}
	Line(const cv::Vec2f& vec) : a(vec[0]), b(vec[1]) {}
	Line(const cv::Point2f& pt1, const cv::Point2f& pt2)
	{
		a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
		b = -a * pt1.x + pt1.y;
	}
	inline float operator()(int x) const { return a * x + b; }
	float vhor() const { return -b / a; }
	float a, b;
};

////////////////////////////////////////////////////////////////////////////////////
// Cost functions
////////////////////////////////////////////////////////////////////////////////////
using Parameters = SemanticStixels::Parameters;

struct NegativeLogDataTermGrd
{
	NegativeLogDataTermGrd(int dmin, int dmax, const CameraParameters& camera, int vmax, const Line& road)
	{
		init(dmin, dmax, camera, vmax, road);
	}

	inline float operator()(float d, int v) const
	{
		if (d < 0.f)
			return nLogPInvD;

		// [Experimental] this error saturation suppresses misdetection like "object below ground"
		const float error = std::max(d - fn_[v], 0.f);
		const float nLogPGaussian = cb_[v] + ca_[v] * squared(error);
		const float nLogPData = std::min(nLogPUniform, nLogPGaussian);
		return nLogPData + nLogPValD;
	}

	// pre-compute constant terms
	void init(int dmin, int dmax, const CameraParameters& camera, int vmax, const Line& road)
	{
		const float bf = camera.baseline * camera.fu;
		const float invHcam = 1.f / camera.height;

		// uniform distribution term
		nLogPUniform = logf(floatCast(dmax - dmin)) - logf(pOutG);

		// probability of invalid and valid disparity
		const float pInv = pInvG * pInvD / pC;
		nLogPInvD = -logf(pInv);
		nLogPValD = -logf(1.f - pInv);

		// gaussian distribution term
		cb_.resize(vmax);
		ca_.resize(vmax);

		// expected disparity
		fn_.resize(vmax);

		for (int v = 0; v < vmax; v++)
		{
			const float fn = std::max(road(v), 0.f);
			const float sigmaRSq = squared(invHcam * fn) * sigmaHSq + squared(invHcam * bf) * sigmaASq;
			const float sigmaGSq = sigmaDSq[G] + sigmaRSq;
			const float sigma = sqrtf(sigmaGSq);

			const float tau = SQRT2 * sigma;
			const float itau = 1.f / tau;

			// normalize function
			const float Z = 0.5f * (erff(itau * (dmax - fn)) - erff(itau * (dmin - fn)));

			// gaussian distribution term
			ca_[v] = squared(itau);
			cb_[v] = logf(Z) + logf(tau * SQRTPI) - logf(1.f - pOutG);

			// expected disparity
			fn_[v] = fn;
		}
	}

	float nLogPUniform, nLogPInvD, nLogPValD;
	std::vector<float> cb_, ca_, fn_;
};

struct NegativeLogDataTermObj
{
	NegativeLogDataTermObj(int dmin, int dmax, const CameraParameters& camera)
	{
		init(dmin, dmax, camera);
	}

	inline float operator()(float d, int fn) const
	{
		if (d < 0.f)
			return nLogPInvD;

		const float error = d - fn;
		const float nLogPGaussian = cb_[fn] + ca_[fn] * squared(error);
		const float nLogPData = std::min(nLogPUniform, nLogPGaussian);
		return nLogPData + nLogPValD;
	}

	// pre-compute constant terms
	void init(int dmin, int dmax, const CameraParameters& camera)
	{
		const int fnmax = dmax;

		const float bf = camera.baseline * camera.fu;
		const float invDeltaD = deltaZ / bf;

		// uniform distribution term
		nLogPUniform = logf(floatCast(dmax - dmin)) - logf(pOutO);

		// probability of invalid and valid disparity
		const float pInv = pInvO * pInvD / pC;
		nLogPInvD = -logf(pInv);
		nLogPValD = -logf(1.f - pInv);

		// gaussian distribution term
		cb_.resize(fnmax);
		ca_.resize(fnmax);

		for (int fn = 0; fn < fnmax; fn++)
		{
			const float sigmaZSq = squared(invDeltaD * fn * fn);
			const float sigmaOSq = sigmaDSq[O] + sigmaZSq;
			const float sigma = sqrtf(sigmaOSq);

			const float tau = SQRT2 * sigma;
			const float itau = 1.f / tau;

			// normalize function
			const float Z = 0.5f * (erff(itau * (dmax - fn)) - erff(itau * (dmin - fn)));

			// gaussian distribution term
			ca_[fn] = squared(itau);
			cb_[fn] = logf(Z) + logf(tau * SQRTPI) - logf(1.f - pOutO);
		}
	}

	float nLogPUniform, nLogPInvD, nLogPValD;
	std::vector<float> cb_, ca_;
};

struct NegativeLogDataTermSky
{
	NegativeLogDataTermSky(int dmin, int dmax, const CameraParameters& camera)
	{
		init(dmin, dmax, camera);
	}

	inline float operator()(float d) const
	{
		if (d < 0.f)
			return nLogPInvD;

		const float error = d;
		const float nLogPGaussian = cb_ + ca_ * squared(error);
		const float nLogPData = std::min(nLogPUniform, nLogPGaussian);
		return nLogPData + nLogPValD;
	}

	// pre-compute constant terms
	void init(int dmin, int dmax, const CameraParameters& camera)
	{
		// uniform distribution term
		nLogPUniform = logf(floatCast(dmax - dmin)) - logf(pOutS);

		// probability of invalid and valid disparity
		const float pInv = pInvS * pInvD / pC;
		nLogPInvD = -logf(pInv);
		nLogPValD = -logf(1.f - pInv);

		const float tau = SQRT2 * sigmaD[S];
		const float itau = 1.f / tau;

		// normalize function
		const float fn = 0;
		const float Z = 0.5f * (erff(itau * (dmax - fn)) - erff(itau * (dmin - fn)));

		// gaussian distribution term
		ca_ = squared(itau);
		cb_ = logf(Z) + logf(tau * SQRTPI) - logf(1.f - pOutS);
	}

	float nLogPUniform, nLogPInvD, nLogPValD;
	float cb_, ca_;
};

////////////////////////////////////////////////////////////////////////////////////
// Static functions
////////////////////////////////////////////////////////////////////////////////////

static cv::Mat1f getch(const cv::Mat1f& src, int id)
{
	return cv::Mat1f(src.size[1], src.size[2], (float*)src.ptr<float>(id));
}

static void create3d(cv::Mat1f& mat, int size0, int size1, int size2)
{
	const int sizes[3] = { size0, size1, size2 };
	mat.create(3, sizes);
}

static float calcSum(const cv::Mat1f& src, int srcu, int srcv, int w, int h)
{
	float sum = 0;
	for (int dv = 0; dv < h; dv++)
		for (int du = 0; du < w; du++)
			sum += src(srcv + dv, srcu + du);;
	return sum;
}

static float calcMean(const cv::Mat1f& src, int srcu, int srcv, int w, int h, int threshold)
{
	float sum = 0;
	int cnt = 0;
	for (int dv = 0; dv < h; dv++)
	{
		for (int du = 0; du < w; du++)
		{
			const float d = src(srcv + dv, srcu + du);
			if (d >= 0)
			{
				sum += d;
				cnt++;
			}

		}
	}
	return cnt >= threshold ? sum / cnt : -1;
}

static void reduceTranspose(const cv::Mat1f& src, cv::Mat1f& dst, int stixelW, int stixelH,
	bool hasInvalidValue = false)
{
	const int umax = src.cols / stixelW;
	const int vmax = src.rows / stixelH;

	dst.create(umax, vmax);

	if (hasInvalidValue)
	{
		const int threshold = stixelW * stixelW / 2;
		for (int dstv = 0, srcv = 0; dstv < vmax; dstv++, srcv += stixelH)
			for (int dstu = 0, srcu = 0; dstu < umax; dstu++, srcu += stixelW)
				dst(dstu, dstv) = calcMean(src, srcu, srcv, stixelW, stixelH, threshold);
	}
	else
	{
		const float invArea = 1.f / (stixelW * stixelH);
		for (int dstv = 0, srcv = 0; dstv < vmax; dstv++, srcv += stixelH)
			for (int dstu = 0, srcu = 0; dstu < umax; dstu++, srcu += stixelW)
				dst(dstu, dstv) = invArea * calcSum(src, srcu, srcv, stixelW, stixelH);
	}
}

static void reduceTranspose(const cv::Mat1f& src, cv::Mat1f& dst, int ch, int stixelW, int stixelH,
	bool hasInvalidValue = false)
{
	const cv::Mat1f _src = getch(src, ch);
	cv::Mat1f _dst = getch(dst, ch);
	reduceTranspose(_src, _dst, stixelW, stixelH, hasInvalidValue);
}

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
static Line calcRoadModelVD(const cv::Mat1f& disparity, const CameraParameters& camera, int stixelH,
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
				points.push_back(cv::Point2f(floatCast(stixelH * v), disparity(u, v)));

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
			if (fabsf(line.a * pt.x + line.b - pt.y) <= inlierRadius)
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
		if (fabsf(yhat - y) <= inlierRadius)
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

static cv::Vec2d calcCostScale(const cv::Mat1f& predict)
{
	const int chns = predict.size[0];
	std::vector<double> minvs(chns);
	std::vector<double> maxvs(chns);

	OMP_PARALLEL_FOR
	for (int ch = 0; ch < chns; ch++)
		cv::minMaxIdx(getch(predict, ch), &minvs[ch], &maxvs[ch]);

	const double minv = *std::min_element(std::begin(minvs), std::end(minvs));
	const double maxv = *std::max_element(std::begin(maxvs), std::end(maxvs));

	const double a = -255. / (maxv - minv);
	const double b = -a * maxv;
	return cv::Vec2d(a, b);
}

static void calcSAT(const cv::Mat1f& src, cv::Mat1f& dst, int ch, const cv::Vec2d& scale)
{
	const cv::Mat1f channel = getch(src, ch);
	const int umax = src.size[1];
	const int vmax = src.size[2];

	const float a = floatCast(scale[0]);
	const float b = floatCast(scale[1]);

	for (int u = 0; u < umax; u++)
	{
		const float* ptrSrc = channel.ptr<float>(u);
		float* ptrDst = dst.ptr<float>(u, ch);
		float tmpSum = 0.f;
		for (int v = 0; v < vmax; v++)
		{
			tmpSum += a * ptrSrc[v] + b;
			ptrDst[v] = tmpSum;
		}
	}
}

static inline float priorCostGG(float dGrdB, float dGrdT)
{
	const float delta = dGrdB - dGrdT;
	if (delta > 0)
		return alphaGapP + betaGapP * delta;
	if (delta < 0)
		return alphaGapN - betaGapN * delta;
	return 0.f;
}

static inline float priorCostGO(float dGrdB, float dObjT)
{
	const float delta = dGrdB - dObjT;
	if (delta > 0)
		return alphaGravP + betaGravP * delta;
	if (delta < 0)
		return alphaGravN - betaGravN * delta;
	return 0.f;
}

static inline float priorCostOG(float dObjB, float dGrdT)
{
	const float delta = dObjB - dGrdT;
	if (delta < 0)
		return Cinf;
	return 0.f;
}

static inline float priorCostOO(float dObjB, float dObjT)
{
	const float delta = dObjT - dObjB;
	if (delta > 0)
		return alphaOrd + betaOrd * delta;
	return 0.f;
}

static inline short packIndex(int geoId, int v)
{
	return (v << GEO_ID_BIT) | geoId;
}

static inline cv::Point unpackIndex(short packed)
{
	return { packed & ((1 << GEO_ID_BIT) - 1), packed >> GEO_ID_BIT };
}

struct BestCost
{
	inline void init(const cv::Vec3f& _costs, float _dispO)
	{
		costs = _costs;

		points[G] = packIndex(G, 0);
		points[O] = packIndex(O, 0);
		points[S] = packIndex(S, 0);

		dispO = _dispO;
	}

	inline void init(const cv::Vec3f& _costs, const cv::Vec3b& _labels, float _dispO)
	{
		costs = _costs;
		labels = _labels;

		points[G] = packIndex(G, 0);
		points[O] = packIndex(O, 0);
		points[S] = packIndex(S, 0);

		dispO = _dispO;
	}

	template <int C1, int C2>
	inline void update(int vT, float cost)
	{
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, float disp)
	{
		static_assert(C1 == O, "C1 must be class Obj");
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			dispO = disp;
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, int label)
	{
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			labels[C1] = label;
		}
	}

	template <int C1, int C2>
	inline void update(int vT, float cost, float disp, int label)
	{
		static_assert(C1 == O, "C1 must be class Obj");
		if (cost < costs[C1])
		{
			costs[C1] = cost;
			points[C1] = packIndex(C2, vT - 1);
			labels[C1] = label;
			dispO = disp;
		}
	}

	cv::Vec3f costs;
	cv::Vec3s points;
	cv::Vec3b labels;
	float dispO;
};

using DataTermG = NegativeLogDataTermGrd;
using DataTermO = NegativeLogDataTermObj;
using DataTermS = NegativeLogDataTermSky;

struct DispCostLUT
{
	DispCostLUT(const DataTermG& dataTermG, const DataTermO& dataTermO, const DataTermS& dataTermS,
		int vmax, int dmax) : dataTermG(dataTermG), dataTermO(dataTermO), dataTermS(dataTermS), dmax(dmax)
	{
		costG.create(1, vmax);
		costO.create(dmax, vmax);
		costS.create(1, vmax);
		sumD.create(1, vmax);
		cntD.create(1, vmax);

		tmpSumG = 0.f;
		tmpSumS = 0.f;
		tmpSumO.assign(dmax, 0.f);
		tmpSumD = 0.f;
		tmpCntD = 0;
	}

	inline void add(int v, float d)
	{
		// pre-computation for ground costs
		tmpSumG += dataTermG(d, v);
		costG(v) = tmpSumG;

		// pre-computation for sky costs
		tmpSumS += dataTermS(d);
		costS(v) = tmpSumS;

		// pre-computation for object costs
		for (int fn = 0; fn < dmax; fn++)
		{
			tmpSumO[fn] += dataTermO(d, fn);
			costO(fn, v) = tmpSumO[fn];
		}

		// pre-computation for mean disparity of stixel
		if (d >= 0.f)
		{
			tmpSumD += d;
			tmpCntD++;
		}
		sumD(v) = tmpSumD;
		cntD(v) = tmpCntD;
	}

	inline float sumG(int vB) const
	{
		return costG(vB);
	}
	inline float sumG(int vB, int vT) const
	{
		return costG(vB) - costG(vT - 1);
	}

	inline float sumO(int fn, int vB) const
	{
		return costO(fn, vB);
	}
	inline float sumO(int fn, int vB, int vT) const
	{
		return costO(fn, vB) - costO(fn, vT - 1);
	}

	inline float sumS(int vB) const
	{
		return costS(vB);
	}
	inline float sumS(int vB, int vT) const
	{
		return costS(vB) - costS(vT - 1);
	}

	inline float meanD(int vB) const
	{
		return sumD(vB) / std::max(cntD(vB), 1);
	}
	inline float meanD(int vB, int vT) const
	{
		return (sumD(vB) - sumD(vT - 1)) / std::max((cntD(vB) - cntD(vT - 1)), 1);
	}

	const DataTermG& dataTermG;
	const DataTermO& dataTermO;
	const DataTermS& dataTermS;

	cv::Mat1f costG, costO, costS, sumD;
	cv::Mat1i cntD;

	float tmpSumG, tmpSumS, tmpSumD;
	int tmpCntD, dmax;
	std::vector<float> tmpSumO;
};

static void processOneColumn(int u, int vmax, int dmax, const cv::Mat1f& disparity, const Line& road,
	const DataTermG& dataTermG, const DataTermO& dataTermO, const DataTermS& dataTermS,
	cv::Mat3f& costTable, cv::Mat3s& indexTable, cv::Mat1f& dispTable)
{
	const int vhor = static_cast<int>(road.vhor());

	// compute data cost LUT
	DispCostLUT LUT(dataTermG, dataTermO, dataTermS, vmax, dmax);
	const float* disparityU = disparity.ptr<float>(u);
	for (int v = 0; v < vmax; v++)
		LUT.add(v, disparityU[v]);

	////////////////////////////////////////////////////////////////////////////////////////////
	// compute cost tables
	//
	// for paformance optimization, loop is split at vhor and unnecessary computation is ommited
	////////////////////////////////////////////////////////////////////////////////////////////
	cv::Vec3f* costTableU = costTable.ptr<cv::Vec3f>(u);
	cv::Vec3s* indexTableU = indexTable.ptr<cv::Vec3s>(u);
	float* dispTableU = dispTable.ptr<float>(u);

	// process vB = 0 to vhor
	// in this range, the class ground is not evaluated
	for (int vB = 0; vB <= vhor; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB);
			const int fn = cvRound(dO1);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = LUT.sumO(fn, vB);
			const float dataCostS = LUT.sumS(vB);

			// initialize best cost
			bestCost.init({ dataCostG, dataCostO, dataCostS }, dO1);
		}

		for (int vT = 1; vT <= vB; vT++)
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB, vT);
			const float dO2 = dispTableU[vT - 1];
			const int fn = cvRound(dO1);

			// compute data cost
			const float dataCostO = LUT.sumO(fn, vB, vT);
			const float dataCostS = LUT.sumS(vB, vT);

			// compute total cost
			const cv::Vec3f& prevCost = costTableU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(dO1, dO2);
			const float costOS = dataCostO + prevCost[S] + Cmc;
			const float costSO = dataCostS + prevCost[O] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, dO1);
			bestCost.update<O, S>(vT, costOS, dO1);
			bestCost.update<S, O>(vT, costSO);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		dispTableU[vB] = bestCost.dispO;
	}

	// process vhor + 1 to vmax
	// in this range, the class sky is not evaluated
	for (int vB = vhor + 1; vB < vmax; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB);
			const int fn = cvRound(dO1);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = LUT.sumO(fn, vB);
			const float dataCostS = Cinf;

			// initialize best cost
			bestCost.init({ dataCostG, dataCostO, dataCostS }, dO1);
		}

		// process vT = 1 to vhor
		// in this range, transition from/to ground is not allowed
		for (int vT = 1; vT <= vhor; vT++)
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB, vT);
			const float dO2 = dispTableU[vT - 1];
			const int fn = cvRound(dO1);

			// compute data cost
			const float dataCostO = LUT.sumO(fn, vB, vT);

			const cv::Vec3f& prevCost = costTableU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(dO1, dO2);
			const float costOS = dataCostO + prevCost[S] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, dO1);
			bestCost.update<O, S>(vT, costOS, dO1);
		}

		// process vT = vhor + 1 to vB
		// in this range, transition from sky is not allowed
		for (int vT = vhor + 1; vT <= vB; vT++)
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB, vT);
			const float dO2 = dispTableU[vT - 1];
			const int fn = cvRound(dO1);
			const float dG1 = road(vT);
			const float dG2 = road(vT - 1);

			// compute data cost
			const float dataCostG = LUT.sumG(vB, vT);
			const float dataCostO = LUT.sumO(fn, vB, vT);

			const cv::Vec3f& prevCost = costTableU[vT - 1];

			// compute total cost
			const float costGG = dataCostG + prevCost[G] + Cmc + priorCostGG(dG1, dG2);
			const float costGO = dataCostG + prevCost[O] + Cmc + priorCostGO(dG1, dO2);
			const float costOG = dataCostO + prevCost[G] + Cmc + priorCostOG(dO1, dG2);
			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(dO1, dO2);

			// update best cost
			bestCost.update<G, G>(vT, costGG);
			bestCost.update<G, O>(vT, costGO);
			bestCost.update<O, G>(vT, costOG, dO1);
			bestCost.update<O, O>(vT, costOO, dO1);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		dispTableU[vB] = bestCost.dispO;
	}
}

std::pair<float, int> calcMinCostAndLabel(const cv::Mat1f& SATsem,
	const std::vector<int>& labels, int vB, int vT, int iniLabel = -1)
{
	auto calcCost = [&](int label)
	{
		return vT > 0 ? SATsem(label, vB) - SATsem(label, vT - 1) : SATsem(label, vB);
	};

	float minCost = iniLabel >= 0 ? calcCost(iniLabel) : Cinf;
	int minLabel = iniLabel;
	for (int label : labels)
	{
		const float cost = calcCost(label);
		if (cost < minCost)
		{
			minCost = cost;
			minLabel = label;
		}
	}
	return { minCost, minLabel };
}

static void processOneColumn(int u, int vmax, int dmax, const cv::Mat1f& disparity, const Line& road,
	const DataTermG& dataTermG, const DataTermO& dataTermO, const DataTermS& dataTermS,
	const cv::Mat1f& SATsem, const std::vector<int> G2L[],
	cv::Mat3f& costTable, cv::Mat3s& indexTable, cv::Mat3b& labelTable, cv::Mat1f& dispTable)
{
	const int vhor = static_cast<int>(road.vhor());
	const int iniLabel = !G2L[NO_GEOMETRY].empty() ? G2L[NO_GEOMETRY].front() : -1;

	// compute data cost LUT
	DispCostLUT LUT(dataTermG, dataTermO, dataTermS, vmax, dmax);
	const float* disparityU = disparity.ptr<float>(u);
	for (int v = 0; v < vmax; v++)
		LUT.add(v, disparityU[v]);

	////////////////////////////////////////////////////////////////////////////////////////////
	// compute cost tables
	//
	// for paformance optimization, loop is split at vhor and unnecessary computation is ommited
	////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat1f SATsemU = getch(SATsem, u);
	cv::Vec3f* costTableU = costTable.ptr<cv::Vec3f>(u);
	cv::Vec3s* indexTableU = indexTable.ptr<cv::Vec3s>(u);
	cv::Vec3b* labelTableU = labelTable.ptr<cv::Vec3b>(u);
	float* dispTableU = dispTable.ptr<float>(u);

	// process vB = 0 to vhor
	// in this range, the class ground is not evaluated
	for (int vB = 0; vB <= vhor; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB);
			const int fn = cvRound(dO1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, 0, iniLabel);
			const auto [minSemCostS, minLabelS] = calcMinCostAndLabel(SATsemU, G2L[S], vB, 0, iniLabel);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = LUT.sumO(fn, vB) + wsem * minSemCostO;
			const float dataCostS = LUT.sumS(vB) + wsem * minSemCostS;

			// initialize best cost
			bestCost.init({ dataCostG, dataCostO, dataCostS }, cv::Vec3b(0, minLabelO, minLabelS), dO1);
		}

		for (int vT = 1; vT <= vB; vT++)
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB, vT);
			const float dO2 = dispTableU[vT - 1];
			const int fn = cvRound(dO1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, vT, iniLabel);
			const auto [minSemCostS, minLabelS] = calcMinCostAndLabel(SATsemU, G2L[S], vB, vT, iniLabel);

			// compute data cost
			const float dataCostO = LUT.sumO(fn, vB, vT) + wsem * minSemCostO;
			const float dataCostS = LUT.sumS(vB, vT) + wsem * minSemCostS;

			// compute total cost
			const cv::Vec3f& prevCost = costTableU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(dO1, dO2);
			const float costOS = dataCostO + prevCost[S] + Cmc;
			const float costSO = dataCostS + prevCost[O] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, dO1, minLabelO);
			bestCost.update<O, S>(vT, costOS, dO1, minLabelO);
			bestCost.update<S, O>(vT, costSO, minLabelS);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		labelTableU[vB] = bestCost.labels;
		dispTableU[vB] = bestCost.dispO;
	}

	// process vhor + 1 to vmax
	// in this range, the class sky is not evaluated
	for (int vB = vhor + 1; vB < vmax; vB++)
	{
		BestCost bestCost;

		// process vT = 0
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB);
			const int fn = cvRound(dO1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, 0, iniLabel);

			// compute data cost
			const float dataCostG = Cinf;
			const float dataCostO = LUT.sumO(fn, vB) + wsem * minSemCostO;
			const float dataCostS = Cinf;

			// initialize best cost
			bestCost.init({ dataCostG, dataCostO, dataCostS }, cv::Vec3b(0, minLabelO, 0), dO1);
		}

		// process vT = 1 to vhor
		// in this range, transition from/to ground is not allowed
		for (int vT = 1; vT <= vhor; vT++)
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB, vT);
			const float dO2 = dispTableU[vT - 1];
			const int fn = cvRound(dO1);

			// minimization over the semantic labels
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, vT, iniLabel);

			// compute data cost
			const float dataCostO = LUT.sumO(fn, vB, vT) + wsem * minSemCostO;

			const cv::Vec3f& prevCost = costTableU[vT - 1];

			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(dO1, dO2);
			const float costOS = dataCostO + prevCost[S] + Cmc;

			// update best cost
			bestCost.update<O, O>(vT, costOO, dO1, minLabelO);
			bestCost.update<O, S>(vT, costOS, dO1, minLabelO);
		}

		// process vT = vhor + 1 to vB
		// in this range, transition from sky is not allowed
		for (int vT = vhor + 1; vT <= vB; vT++)
		{
			// compute mean disparity within the range of vB to vT
			const float dO1 = LUT.meanD(vB, vT);
			const float dO2 = dispTableU[vT - 1];
			const int fn = cvRound(dO1);
			const float dG1 = road(vT);
			const float dG2 = road(vT - 1);

			// minimization over the semantic labels
			const auto [minSemCostG, minLabelG] = calcMinCostAndLabel(SATsemU, G2L[G], vB, vT, iniLabel);
			const auto [minSemCostO, minLabelO] = calcMinCostAndLabel(SATsemU, G2L[O], vB, vT, iniLabel);

			// compute data cost
			const float dataCostG = LUT.sumG(vB, vT) + wsem * minSemCostG;
			const float dataCostO = LUT.sumO(fn, vB, vT) + wsem * minSemCostO;

			const cv::Vec3f& prevCost = costTableU[vT - 1];

			// compute total cost
			const float costGG = dataCostG + prevCost[G] + Cmc + priorCostGG(dG1, dG2);
			const float costGO = dataCostG + prevCost[O] + Cmc + priorCostGO(dG1, dO2);
			const float costOG = dataCostO + prevCost[G] + Cmc + priorCostOG(dO1, dG2);
			const float costOO = dataCostO + prevCost[O] + Cmc + priorCostOO(dO1, dO2);

			// update best cost
			bestCost.update<G, G>(vT, costGG, minLabelG);
			bestCost.update<G, O>(vT, costGO, minLabelG);
			bestCost.update<O, G>(vT, costOG, dO1, minLabelO);
			bestCost.update<O, O>(vT, costOO, dO1, minLabelO);
		}

		costTableU[vB] = bestCost.costs;
		indexTableU[vB] = bestCost.points;
		labelTableU[vB] = bestCost.labels;
		dispTableU[vB] = bestCost.dispO;
	}
}

static void extractStixels(const cv::Mat3f& costTable, const cv::Mat3s& indexTable,
	const cv::Mat1f& dispTable, std::vector<Stixel>& stixels)
{
	const int umax = costTable.rows;
	const int vmax = costTable.cols;

	for (int u = 0; u < umax; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, vmax - 1)[c];
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, vmax - 1);
			}
		}

		while (minPos.y > 0)
		{
			const int geoId = minPos.x;
			const int v = minPos.y;

			const cv::Point p1 = minPos;
			const cv::Point p2 = unpackIndex(indexTable(u, v)[geoId]);

			Stixel stixel;
			stixel.uL = u;
			stixel.vT = p2.y + 1;
			stixel.vB = p1.y + 1;
			stixel.width = 1;
			stixel.geoId = geoId;
			stixel.semId = 0;
			stixel.disp = dispTable(u, v);

			if (geoId == O)
				stixels.push_back(stixel);

			minPos = p2;
		}
	}
}

static void extractStixels(const cv::Mat3f& costTable, const cv::Mat3s& indexTable,
	const cv::Mat3b& labelTable, const cv::Mat1f& dispTable, std::vector<Stixel>& stixels)
{
	const int umax = costTable.rows;
	const int vmax = costTable.cols;

	for (int u = 0; u < umax; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, vmax - 1)[c];
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, vmax - 1);
			}
		}

		while (minPos.y > 0)
		{
			const int geoId = minPos.x;
			const int v = minPos.y;

			const cv::Point p1 = minPos;
			const cv::Point p2 = unpackIndex(indexTable(u, v)[geoId]);

			Stixel stixel;
			stixel.uL = u;
			stixel.vT = p2.y + 1;
			stixel.vB = p1.y + 1;
			stixel.width = 1;
			stixel.geoId = geoId;
			stixel.semId = labelTable(u, v)[geoId];
			stixel.disp = geoId == O ? dispTable(u, v) : 0;

			stixels.push_back(stixel);

			minPos = p2;
		}
	}
}

class SemanticStixelsImpl : public SemanticStixels
{
public:

	SemanticStixelsImpl(const Parameters& param) : param_(param)
	{
		init();
	}

	void init()
	{
		const auto& L2G = param_.geometry;
		//CV_Assert(!L2G.empty());

		const int chns = static_cast<int>(L2G.size());
		for (int ch = 0; ch < chns; ch++)
		{
			const int geoId = L2G[ch];
			if (geoId >= 0 && geoId <= 2)
				G2L_[geoId].push_back(ch);
			if (geoId < 0)
				G2L_[NO_GEOMETRY].push_back(ch);
		}
	}

	void compute(const cv::Mat& disparity, std::vector<Stixel>& stixels) override
	{
		stixels.clear();

		const int stixelW = param_.stixelWidth;
		const int stixelH = param_.stixelYResolution;
		const int dmin = param_.dmin;
		const int dmax = param_.dmax;

		CV_Assert(disparity.type() == CV_32F);
		CV_Assert(stixelW == STIXEL_WIDTH_4 || stixelW == STIXEL_WIDTH_8);
		CV_Assert(stixelH == STIXEL_Y_RESOLUTION_4 || stixelH == STIXEL_Y_RESOLUTION_8);

		//////////////////////////////////////////////////////////////////////////////
		// process depth input
		//////////////////////////////////////////////////////////////////////////////

		// reduce and reorder disparity map
		reduceTranspose(disparity, disparity_, stixelW, stixelH, true);

		const int umax = disparity_.rows;
		const int vmax = disparity_.cols;

		// estimate road model
		CameraParameters camera = param_.camera;
		Line road = calcRoadModelCamera(camera);
		if (param_.roadEstimation == ROAD_ESTIMATION_AUTO)
		{
			road = calcRoadModelVD(disparity_, camera, stixelH);
			camera.tilt = atanf((road.a * camera.v0 + road.b) / (camera.fu * road.a));
			camera.height = camera.baseline * cosf(camera.tilt) / road.a;
		}
		road.a *= stixelH; // correct slope according to stixel Y resolution

		const int vhor = static_cast<int>(road.vhor());
		if (vhor < 0 || vhor >= vmax)
			return;

		//////////////////////////////////////////////////////////////////////////////
		// dynamic programming
		//////////////////////////////////////////////////////////////////////////////
		NegativeLogDataTermGrd dataTermG(dmin, dmax, camera, vmax, road);
		NegativeLogDataTermObj dataTermO(dmin, dmax, camera);
		NegativeLogDataTermSky dataTermS(dmin, dmax, camera);

		costTable_.create(umax, vmax);
		indexTable_.create(umax, vmax);
		dispTable_.create(umax, vmax);

		OMP_PARALLEL_FOR
		for (int u = 0; u < umax; u++)
		{
			processOneColumn(u, vmax, dmax, disparity_, road, dataTermG, dataTermO, dataTermS,
				costTable_, indexTable_, dispTable_);
		}

		extractStixels(costTable_, indexTable_, dispTable_, stixels);

		for (auto& stixel : stixels)
		{
			stixel.uL *= stixelW;
			stixel.vT *= stixelH;
			stixel.vB *= stixelH;
			stixel.width = stixelW;
		}
	}

	void compute(const cv::Mat& disparity, const cv::Mat& predict, std::vector<Stixel>& stixels) override
	{
		stixels.clear();

		const int stixelW = param_.stixelWidth;
		const int stixelH = param_.stixelYResolution;
		const int dmin = param_.dmin;
		const int dmax = param_.dmax;

		CV_Assert(disparity.type() == CV_32F && predict.type() == CV_32F);
		CV_Assert(disparity.rows == predict.size[1] && disparity.cols == predict.size[2]);
		CV_Assert(stixelW == STIXEL_WIDTH_4 || stixelW == STIXEL_WIDTH_8);
		CV_Assert(stixelH == STIXEL_Y_RESOLUTION_4 || stixelH == STIXEL_Y_RESOLUTION_8);

		//////////////////////////////////////////////////////////////////////////////
		// process depth input
		//////////////////////////////////////////////////////////////////////////////

		// reduce and reorder disparity map
		reduceTranspose(disparity, disparity_, stixelW, stixelH, true);

		const int umax = disparity_.rows;
		const int vmax = disparity_.cols;

		// estimate road model
		CameraParameters camera = param_.camera;
		Line road = calcRoadModelCamera(camera);
		if (param_.roadEstimation == ROAD_ESTIMATION_AUTO)
		{
			road = calcRoadModelVD(disparity_, camera, stixelH);
			camera.tilt = atanf((road.a * camera.v0 + road.b) / (camera.fu * road.a));
			camera.height = camera.baseline * cosf(camera.tilt) / road.a;
		}
		road.a *= stixelH; // correct slope according to stixel Y resolution

		const int vhor = static_cast<int>(road.vhor());
		if (vhor < 0 || vhor >= vmax)
			return;

		//////////////////////////////////////////////////////////////////////////////
		// process semantic input
		//////////////////////////////////////////////////////////////////////////////
		const int chns = predict.size[0];
		create3d(predict_, chns, umax, vmax);
		create3d(SATsem_, umax, chns, vmax);

		OMP_PARALLEL_FOR
		for (int ch = 0; ch < chns; ch++)
			reduceTranspose(predict, predict_, ch, stixelW, stixelH);

		const auto costScale = calcCostScale(predict_);

		OMP_PARALLEL_FOR
		for (int ch = 0; ch < chns; ch++)
			calcSAT(predict_, SATsem_, ch, costScale);

		//////////////////////////////////////////////////////////////////////////////
		// dynamic programming
		//////////////////////////////////////////////////////////////////////////////
		NegativeLogDataTermGrd dataTermG(dmin, dmax, camera, vmax, road);
		NegativeLogDataTermObj dataTermO(dmin, dmax, camera);
		NegativeLogDataTermSky dataTermS(dmin, dmax, camera);

		costTable_.create(umax, vmax);
		indexTable_.create(umax, vmax);
		labelTable_.create(umax, vmax);
		dispTable_.create(umax, vmax);

		OMP_PARALLEL_FOR
		for (int u = 0; u < umax; u++)
		{
			processOneColumn(u, vmax, dmax, disparity_, road, dataTermG, dataTermO, dataTermS,
				SATsem_, G2L_, costTable_, indexTable_, labelTable_, dispTable_);
		}

		extractStixels(costTable_, indexTable_, labelTable_, dispTable_, stixels);

		for (auto& stixel : stixels)
		{
			stixel.uL *= stixelW;
			stixel.vT *= stixelH;
			stixel.vB *= stixelH;
			stixel.width = stixelW;
		}
	}

	void setParameters(const Parameters& param) override
	{
		param_ = param;
	}

private:

	cv::Mat1f disparity_;
	cv::Mat3f costTable_;
	cv::Mat3s indexTable_;
	cv::Mat3b labelTable_;
	cv::Mat1f dispTable_;

	cv::Mat1f predict_;
	cv::Mat1f SATsem_;
	std::vector<int> G2L_[4];

	Parameters param_;
};

cv::Ptr<SemanticStixels> SemanticStixels::create(const Parameters& param)
{
	return cv::makePtr<SemanticStixelsImpl>(param);
}
