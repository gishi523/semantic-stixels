#include "draw.h"

#include <opencv2/imgproc.hpp>

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v * (1.f - s);
	tab[2] = v * (1.f - s * h);
	tab[3] = v * (1.f - s * (1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

void drawStixel(cv::Mat& img, const Stixel& stixel, const cv::Scalar& color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
}

void colorizeSegmentation(const cv::Mat& score, const std::vector<cv::Vec3b>& colors, cv::Mat& segm)
{
	const int chns = score.size[1];
	const int rows = score.size[2];
	const int cols = score.size[3];

	// Select classes with maximum score
	cv::Mat maxClass = cv::Mat::zeros(rows, cols, CV_8U);
	cv::Mat maxScore(rows, cols, CV_32F, score.data);
	
	for (int ch = 1; ch < chns; ch++)
	{
		for (int v = 0; v < rows; v++)
		{
			const float* ptrScore = score.ptr<float>(0, ch, v);
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

	segm.create(rows, cols, CV_8UC3);
	for (int v = 0; v < rows; v++)
	{
		uchar* ptrMaxClass = maxClass.ptr<uchar>(v);
		cv::Vec3b* ptrSegm = segm.ptr<cv::Vec3b>(v);
		float* ptrMaxScore = maxScore.ptr<float>(rows);
		for (int u = 0; u < cols; u++)
			ptrSegm[u] = colors[ptrMaxClass[u]];
	}
}

void drawLegend(cv::Mat& legend, const std::vector<std::string>& classes, const std::vector<cv::Vec3b>& colors)
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

void putInfo(cv::Mat& img, std::string info, double time)
{
	if (time >= 0)
		info += cv::format(" computation time: %4.1f [msec]", time);
	cv::putText(img, info, cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));
}
