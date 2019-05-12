#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ransac/GRANSAC.hpp"
#include "ransac/LineModel.hpp"

class LaneDetector {
 private:
  cv::Point2f quadA[4], quadB[4];
  GRANSAC::RANSAC<Line2DModel, 2> ransac;

  float steeringAngle;
  std::vector<float> xMiddle, yMiddle;
 public:
  LaneDetector();

  void convertToGrayscale(cv::Mat& input);
  void transformPerspective(cv::Mat& input);
  void inversePerspective(cv::Mat& input);

  std::vector<uint16_t> calcHistogramOverX(cv::Mat& image);
  cv::Mat plotHistogram(cv::Mat& image, std::vector<uint16_t> histogram);
  std::vector<std::vector<float>> calculateLanePoints(cv::Mat& input);
  cv::Mat plotLanePoints(std::vector<std::vector<float>> points, cv::Mat image);
  std::vector<std::vector<float>> fitLanePoints(
    std::vector<std::vector<float>> points, cv::Mat grayImage);
  cv::Mat runCurvePipeline(cv::Mat image);
  void calculateSteeringAngle(cv::Mat& image, bool centerCompensation,  bool plot);
  float getSteeringAngle();
};