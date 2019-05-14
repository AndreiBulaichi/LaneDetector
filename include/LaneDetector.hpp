#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ransac/GRANSAC.hpp"
#include "ransac/LineModel.hpp"
#include "kalman/kalman.h"

class LaneDetector {
 private:

  cv::Point2f quadA[4], quadB[4];
  std::vector<float> xMiddle, yMiddle;
  GRANSAC::RANSAC<Line2DModel, 2> ransac;
  float steeringAngle;
  float steeringAngleFiltered;
  Kalman angleFilter;

 public:

  LaneDetector();

  void convertToGrayscale(cv::Mat&);
  void transformPerspective(cv::Mat&);
  void inversePerspective(cv::Mat&);
  std::vector<uint16_t>* calcHistogram(cv::Mat&);
  cv::Mat plotHistogram(cv::Mat&, std::vector<uint16_t>*);
  std::vector<std::vector<uint16_t>> calculateLanePoints(cv::Mat&);
  std::vector<std::vector<float>>* fitLanePoints(std::vector<std::vector<uint16_t>>, cv::Mat&);
  void plotLanePoints(std::vector<std::vector<float>>*, cv::Mat&);

  cv::Mat* runCurvePipeline(cv::Mat&);
  void runLightCurvePipeline(cv::Mat);
  void calculateSteeringAngle(cv::Mat&, bool, bool);
  float getSteeringAngle();
  float getFilteredSteeringAngle();
};