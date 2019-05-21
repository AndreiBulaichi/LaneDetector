#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "ransac/GRANSAC.hpp"
#include "ransac/LineModel.hpp"
#include "kalman/kalman.h"

using namespace std;

class LaneDetector {
  private:

  float resizeRatio;
  uint16_t width, height;

  Kalman angleFilter;
  float steeringAngle;
  float steeringAngleFiltered;
  vector<float> xMiddle, yMiddle;
  cv::Point2f quadA[4], quadB[4];
  GRANSAC::RANSAC<Line2DModel, 2> ransac;

  vector<Kalman> lineFilters;

  public:

  LaneDetector(float,uint16_t, uint16_t);

  void convertToGrayscale(cv::Mat&);
  void transformPerspective(cv::Mat&);
  void inversePerspective(cv::Mat&);

  vector<uint16_t>* calcHistogram(cv::Mat&);
  cv::Mat plotHistogram(std::vector<uint16_t>*);
  vector<vector<uint16_t>> calcLanePoints(cv::Mat&);
  vector<vector<float>>* fitLanePoints(vector<vector<uint16_t>>, cv::Mat&);
  void plotLanePoints(vector<vector<float>>*, cv::Mat&);

  cv::Mat* runCurvePipeline(cv::Mat&);
  void runLightCurvePipeline(cv::Mat&);
  void calcSteeringAngle(cv::Mat&, bool, bool);
  float getSteeringAngle();
  float getFilteredSteeringAngle();
};
