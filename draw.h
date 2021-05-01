#ifndef __DRAW_H__
#define __DRAW_H__

#include <opencv2/core.hpp>

#include "semantic_stixels.h"

using ColorTable = std::vector<cv::Vec3b>;

void colorizeDisparity(const cv::Mat& src, cv::Mat& dst, int numDisparities, int scale = 1);
void colorizeSegmentation(const cv::Mat& score, cv::Mat& segm, const ColorTable& colors);
void drawLegend(cv::Mat& legend, const std::vector<std::string>& classes, const ColorTable& colors);
void drawDepthStixels(const cv::Mat& img, const std::vector<Stixel>& stixels, cv::Mat& outImg, int dmax = 64);
void drawSemanticStixels(const cv::Mat& img, const std::vector<Stixel>& stixels, cv::Mat& outImg, const ColorTable& colors);

#endif // !__DRAW_H__
