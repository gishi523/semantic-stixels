#include "draw.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const ColorTable& getColorTable()
{
	// Equals the colormap "Turbo" proposed by Google.
	// https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
	// https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a

	// define the basemap
	static const float r[] = { 0.18995f,0.19483f,0.19956f,0.20415f,0.20860f,0.21291f,0.21708f,0.22111f,0.22500f,0.22875f,0.23236f,0.23582f,0.23915f,0.24234f,0.24539f,0.24830f,0.25107f,0.25369f,0.25618f,0.25853f,0.26074f,0.26280f,0.26473f,0.26652f,0.26816f,0.26967f,0.27103f,0.27226f,0.27334f,0.27429f,0.27509f,0.27576f,0.27628f,0.27667f,0.27691f,0.27701f,0.27698f,0.27680f,0.27648f,0.27603f,0.27543f,0.27469f,0.27381f,0.27273f,0.27106f,0.26878f,0.26592f,0.26252f,0.25862f,0.25425f,0.24946f,0.24427f,0.23874f,0.23288f,0.22676f,0.22039f,0.21382f,0.20708f,0.20021f,0.19326f,0.18625f,0.17923f,0.17223f,0.16529f,0.15844f,0.15173f,0.14519f,0.13886f,0.13278f,0.12698f,0.12151f,0.11639f,0.11167f,0.10738f,0.10357f,0.10026f,0.09750f,0.09532f,0.09377f,0.09287f,0.09267f,0.09320f,0.09451f,0.09662f,0.09958f,0.10342f,0.10815f,0.11374f,0.12014f,0.12733f,0.13526f,0.14391f,0.15323f,0.16319f,0.17377f,0.18491f,0.19659f,0.20877f,0.22142f,0.23449f,0.24797f,0.26180f,0.27597f,0.29042f,0.30513f,0.32006f,0.33517f,0.35043f,0.36581f,0.38127f,0.39678f,0.41229f,0.42778f,0.44321f,0.45854f,0.47375f,0.48879f,0.50362f,0.51822f,0.53255f,0.54658f,0.56026f,0.57357f,0.58646f,0.59891f,0.61088f,0.62233f,0.63323f,0.64362f,0.65394f,0.66428f,0.67462f,0.68494f,0.69525f,0.70553f,0.71577f,0.72596f,0.73610f,0.74617f,0.75617f,0.76608f,0.77591f,0.78563f,0.79524f,0.80473f,0.81410f,0.82333f,0.83241f,0.84133f,0.85010f,0.85868f,0.86709f,0.87530f,0.88331f,0.89112f,0.89870f,0.90605f,0.91317f,0.92004f,0.92666f,0.93301f,0.93909f,0.94489f,0.95039f,0.95560f,0.96049f,0.96507f,0.96931f,0.97323f,0.97679f,0.98000f,0.98289f,0.98549f,0.98781f,0.98986f,0.99163f,0.99314f,0.99438f,0.99535f,0.99607f,0.99654f,0.99675f,0.99672f,0.99644f,0.99593f,0.99517f,0.99419f,0.99297f,0.99153f,0.98987f,0.98799f,0.98590f,0.98360f,0.98108f,0.97837f,0.97545f,0.97234f,0.96904f,0.96555f,0.96187f,0.95801f,0.95398f,0.94977f,0.94538f,0.94084f,0.93612f,0.93125f,0.92623f,0.92105f,0.91572f,0.91024f,0.90463f,0.89888f,0.89298f,0.88691f,0.88066f,0.87422f,0.86760f,0.86079f,0.85380f,0.84662f,0.83926f,0.83172f,0.82399f,0.81608f,0.80799f,0.79971f,0.79125f,0.78260f,0.77377f,0.76476f,0.75556f,0.74617f,0.73661f,0.72686f,0.71692f,0.70680f,0.69650f,0.68602f,0.67535f,0.66449f,0.65345f,0.64223f,0.63082f,0.61923f,0.60746f,0.59550f,0.58336f,0.57103f,0.55852f,0.54583f,0.53295f,0.51989f,0.50664f,0.49321f,0.47960f };
	static const float g[] = { 0.07176f,0.08339f,0.09498f,0.10652f,0.11802f,0.12947f,0.14087f,0.15223f,0.16354f,0.17481f,0.18603f,0.19720f,0.20833f,0.21941f,0.23044f,0.24143f,0.25237f,0.26327f,0.27412f,0.28492f,0.29568f,0.30639f,0.31706f,0.32768f,0.33825f,0.34878f,0.35926f,0.36970f,0.38008f,0.39043f,0.40072f,0.41097f,0.42118f,0.43134f,0.44145f,0.45152f,0.46153f,0.47151f,0.48144f,0.49132f,0.50115f,0.51094f,0.52069f,0.53040f,0.54015f,0.54995f,0.55979f,0.56967f,0.57958f,0.58950f,0.59943f,0.60937f,0.61931f,0.62923f,0.63913f,0.64901f,0.65886f,0.66866f,0.67842f,0.68812f,0.69775f,0.70732f,0.71680f,0.72620f,0.73551f,0.74472f,0.75381f,0.76279f,0.77165f,0.78037f,0.78896f,0.79740f,0.80569f,0.81381f,0.82177f,0.82955f,0.83714f,0.84455f,0.85175f,0.85875f,0.86554f,0.87211f,0.87844f,0.88454f,0.89040f,0.89600f,0.90142f,0.90673f,0.91193f,0.91701f,0.92197f,0.92680f,0.93151f,0.93609f,0.94053f,0.94484f,0.94901f,0.95304f,0.95692f,0.96065f,0.96423f,0.96765f,0.97092f,0.97403f,0.97697f,0.97974f,0.98234f,0.98477f,0.98702f,0.98909f,0.99098f,0.99268f,0.99419f,0.99551f,0.99663f,0.99755f,0.99828f,0.99879f,0.99910f,0.99919f,0.99907f,0.99873f,0.99817f,0.99739f,0.99638f,0.99514f,0.99366f,0.99195f,0.98999f,0.98775f,0.98524f,0.98246f,0.97941f,0.97610f,0.97255f,0.96875f,0.96470f,0.96043f,0.95593f,0.95121f,0.94627f,0.94113f,0.93579f,0.93025f,0.92452f,0.91861f,0.91253f,0.90627f,0.89986f,0.89328f,0.88655f,0.87968f,0.87267f,0.86553f,0.85826f,0.85087f,0.84337f,0.83576f,0.82806f,0.82025f,0.81236f,0.80439f,0.79634f,0.78823f,0.78005f,0.77181f,0.76352f,0.75519f,0.74682f,0.73842f,0.73000f,0.72140f,0.71250f,0.70330f,0.69382f,0.68408f,0.67408f,0.66386f,0.65341f,0.64277f,0.63193f,0.62093f,0.60977f,0.59846f,0.58703f,0.57549f,0.56386f,0.55214f,0.54036f,0.52854f,0.51667f,0.50479f,0.49291f,0.48104f,0.46920f,0.45740f,0.44565f,0.43399f,0.42241f,0.41093f,0.39958f,0.38836f,0.37729f,0.36638f,0.35566f,0.34513f,0.33482f,0.32473f,0.31489f,0.30530f,0.29599f,0.28696f,0.27824f,0.26981f,0.26152f,0.25334f,0.24526f,0.23730f,0.22945f,0.22170f,0.21407f,0.20654f,0.19912f,0.19182f,0.18462f,0.17753f,0.17055f,0.16368f,0.15693f,0.15028f,0.14374f,0.13731f,0.13098f,0.12477f,0.11867f,0.11268f,0.10680f,0.10102f,0.09536f,0.08980f,0.08436f,0.07902f,0.07380f,0.06868f,0.06367f,0.05878f,0.05399f,0.04931f,0.04474f,0.04028f,0.03593f,0.03169f,0.02756f,0.02354f,0.01963f,0.01583f };
	static const float b[] = { 0.23217f,0.26149f,0.29024f,0.31844f,0.34607f,0.37314f,0.39964f,0.42558f,0.45096f,0.47578f,0.50004f,0.52373f,0.54686f,0.56942f,0.59142f,0.61286f,0.63374f,0.65406f,0.67381f,0.69300f,0.71162f,0.72968f,0.74718f,0.76412f,0.78050f,0.79631f,0.81156f,0.82624f,0.84037f,0.85393f,0.86692f,0.87936f,0.89123f,0.90254f,0.91328f,0.92347f,0.93309f,0.94214f,0.95064f,0.95857f,0.96594f,0.97275f,0.97899f,0.98461f,0.98930f,0.99303f,0.99583f,0.99773f,0.99876f,0.99896f,0.99835f,0.99697f,0.99485f,0.99202f,0.98851f,0.98436f,0.97959f,0.97423f,0.96833f,0.96190f,0.95498f,0.94761f,0.93981f,0.93161f,0.92305f,0.91416f,0.90496f,0.89550f,0.88580f,0.87590f,0.86581f,0.85559f,0.84525f,0.83484f,0.82437f,0.81389f,0.80342f,0.79299f,0.78264f,0.77240f,0.76230f,0.75237f,0.74265f,0.73316f,0.72393f,0.71500f,0.70599f,0.69651f,0.68660f,0.67627f,0.66556f,0.65448f,0.64308f,0.63137f,0.61938f,0.60713f,0.59466f,0.58199f,0.56914f,0.55614f,0.54303f,0.52981f,0.51653f,0.50321f,0.48987f,0.47654f,0.46325f,0.45002f,0.43688f,0.42386f,0.41098f,0.39826f,0.38575f,0.37345f,0.36140f,0.34963f,0.33816f,0.32701f,0.31622f,0.30581f,0.29581f,0.28623f,0.27712f,0.26849f,0.26038f,0.25280f,0.24579f,0.23937f,0.23356f,0.22835f,0.22370f,0.21960f,0.21602f,0.21294f,0.21032f,0.20815f,0.20640f,0.20504f,0.20406f,0.20343f,0.20311f,0.20310f,0.20336f,0.20386f,0.20459f,0.20552f,0.20663f,0.20788f,0.20926f,0.21074f,0.21230f,0.21391f,0.21555f,0.21719f,0.21880f,0.22038f,0.22188f,0.22328f,0.22456f,0.22570f,0.22667f,0.22744f,0.22800f,0.22831f,0.22836f,0.22811f,0.22754f,0.22663f,0.22536f,0.22369f,0.22161f,0.21918f,0.21650f,0.21358f,0.21043f,0.20706f,0.20348f,0.19971f,0.19577f,0.19165f,0.18738f,0.18297f,0.17842f,0.17376f,0.16899f,0.16412f,0.15918f,0.15417f,0.14910f,0.14398f,0.13883f,0.13367f,0.12849f,0.12332f,0.11817f,0.11305f,0.10797f,0.10294f,0.09798f,0.09310f,0.08831f,0.08362f,0.07905f,0.07461f,0.07031f,0.06616f,0.06218f,0.05837f,0.05475f,0.05134f,0.04814f,0.04516f,0.04243f,0.03993f,0.03753f,0.03521f,0.03297f,0.03082f,0.02875f,0.02677f,0.02487f,0.02305f,0.02131f,0.01966f,0.01809f,0.01660f,0.01520f,0.01387f,0.01264f,0.01148f,0.01041f,0.00942f,0.00851f,0.00769f,0.00695f,0.00629f,0.00571f,0.00522f,0.00481f,0.00449f,0.00424f,0.00408f,0.00401f,0.00401f,0.00410f,0.00427f,0.00453f,0.00486f,0.00529f,0.00579f,0.00638f,0.00705f,0.00780f,0.00863f,0.00955f,0.01055f };
	static std::vector<cv::Vec3b> s_colorTable;
	if (s_colorTable.empty())
	{
		s_colorTable.resize(256);
		for (int i = 0; i < 256; i++)
		{
			s_colorTable[i][0] = std::min(cvRound(255 * b[i]), 255);
			s_colorTable[i][1] = std::min(cvRound(255 * g[i]), 255);
			s_colorTable[i][2] = std::min(cvRound(255 * r[i]), 255);
		}
	}
	return s_colorTable;
}

void colorizeDisparity(const cv::Mat& src, cv::Mat& dst, int numDisparities, int scale)
{
	CV_Assert(src.type() == CV_32F);

	dst.create(src.size(), CV_8UC3);

	const float invScale = 255.f / (numDisparities * scale);
	const auto& colors = getColorTable();

	for (int v = 0; v < src.rows; v++)
	{
		const float* ptrSrc = src.ptr<float>(v);
		cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(v);
		for (int u = 0; u < src.cols; u++)
		{
			const float d = ptrSrc[u];
			ptrDst[u] = d >= 0 ? colors[cv::saturate_cast<uchar>(invScale * d)] : cv::Vec3b(0, 0, 0);
		}
	}
}

void colorizeSegmentation(const cv::Mat& src, cv::Mat& dst, const ColorTable& colors)
{
	const int chns = src.size[0];
	const int rows = src.size[1];
	const int cols = src.size[2];

	// Select classes with maximum score
	cv::Mat maxClass = cv::Mat::zeros(rows, cols, CV_8U);
	cv::Mat maxScore(rows, cols, CV_32F, src.data);

	for (int ch = 1; ch < chns; ch++)
	{
		for (int v = 0; v < rows; v++)
		{
			const float* ptrScore = src.ptr<float>(ch, v);
			uchar* ptrMaxClass = maxClass.ptr<uchar>(v);
			float* ptrMaxScore = maxScore.ptr<float>(v);
			for (int u = 0; u < cols; u++)
			{
				if (ptrScore[u] > ptrMaxScore[u])
				{
					ptrMaxScore[u] = ptrScore[u];
					ptrMaxClass[u] = static_cast<uchar>(ch);
				}
			}
		}
	}

	dst.create(rows, cols, CV_8UC3);
	for (int v = 0; v < rows; v++)
	{
		uchar* ptrMaxClass = maxClass.ptr<uchar>(v);
		cv::Vec3b* ptrSegm = dst.ptr<cv::Vec3b>(v);
		for (int u = 0; u < cols; u++)
			ptrSegm[u] = colors[ptrMaxClass[u]];
	}
}

void drawLegend(cv::Mat& legend, const std::vector<std::string>& classes, const ColorTable& colors)
{
	const int blockHeight = 30;
	const auto textColor = cv::Vec3b(255, 255, 255);
	if (legend.empty())
	{
		CV_Assert(classes.size() == colors.size());
		const int nclasses = static_cast<int>(classes.size());
		legend.create(blockHeight * nclasses, 200, CV_8UC3);
		for (int i = 0; i < nclasses; i++)
		{
			cv::Mat block = legend.rowRange(i * blockHeight, (i + 1) * blockHeight);
			block.setTo(colors[i]);
			cv::putText(block, classes[i], cv::Point(0, blockHeight / 2), 0, 0.5, textColor);
		}
	}
}

void drawDepthStixels(const cv::Mat& img, const std::vector<Stixel>& stixels, cv::Mat& outImg, int dmax)
{
	CV_Assert(img.type() == CV_8U || img.type() == CV_8UC3);

	cv::Mat srcImg;
	if (img.type() == CV_8U)
		cv::cvtColor(img, srcImg, cv::COLOR_GRAY2BGR);
	else
		srcImg = img;

	cv::Mat stixelImg = cv::Mat::zeros(srcImg.size(), CV_8UC3);

	const float invScale = 255.f / dmax;
	const auto& colors = getColorTable();

	auto dispToColor = [&](float disp)
	{
		return colors[cv::saturate_cast<uchar>(invScale * disp)];
	};

	for (const auto& stixel : stixels)
	{
		const cv::Rect rect(stixel.uL, stixel.vT, stixel.width, stixel.vB - stixel.vT);
		if (stixel.geoId != GEOMETRIC_ID_GROUND)
			stixelImg(rect) = dispToColor(stixel.disp);

		const cv::Point tl = rect.tl();
		const cv::Point br = rect.br();
		cv::line(stixelImg, tl, cv::Point(br.x, tl.y), cv::Scalar::all(255), 2);
		cv::line(stixelImg, tl, cv::Point(tl.x, br.y), cv::Scalar::all(255));
	}

	cv::addWeighted(srcImg, 0.5, stixelImg, 0.5, 0, outImg);
}

void drawSemanticStixels(const cv::Mat& img, const std::vector<Stixel>& stixels, cv::Mat& outImg,
	const ColorTable& colors)
{
	CV_Assert(img.type() == CV_8UC3);

	cv::Mat stixelImg = cv::Mat::zeros(img.size(), CV_8UC3);
	for (const auto& stixel : stixels)
	{
		const auto& color = colors[stixel.semId];
		const cv::Rect rect(stixel.uL, stixel.vT, stixel.width, stixel.vB - stixel.vT);
		const cv::Point tl = rect.tl();
		const cv::Point br = rect.br();
		cv::rectangle(stixelImg, rect, color, -1);
		cv::line(stixelImg, tl, cv::Point(br.x, tl.y), cv::Scalar::all(255), 2);
		cv::line(stixelImg, tl, cv::Point(tl.x, br.y), cv::Scalar::all(255));
	}
	cv::addWeighted(img, 0.5, stixelImg, 0.5, 0, outImg);
}
