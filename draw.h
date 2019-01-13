#ifndef __DRAW_H__
#define __DRAW_H__

#include <opencv2/core.hpp>

#include "semantic_stixels.h"

cv::Scalar dispToColor(float disp, float maxdisp);
void drawStixel(cv::Mat& img, const Stixel& stixel, const cv::Scalar& color);
void colorizeSegmentation(const cv::Mat& score, const std::vector<cv::Vec3b>& colors, cv::Mat& segm);
void drawLegend(cv::Mat& legend, const std::vector<std::string>& classes, const std::vector<cv::Vec3b>& colors);
void putInfo(cv::Mat& img, std::string info, double time = -1);

#endif // !__DRAW_H__
