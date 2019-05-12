#include "../include/LaneDetector.hpp"
#include "../LaneDetector/LaneDetector.cpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <valarray>

#include "../include/plot/plot.h"

cv::Mat3b PlotGraph(std::vector<double> data);

int main(int argc, char *argv[]) {
    if (argc != 2) {
      std::cout << "Not enough parameters" << std::endl;
      return -1;
    }

    std::string source = argv[1];
    cv::VideoCapture cap(source);
    if (!cap.isOpened())
      return -1;

    std::valarray<int> fpsMean(100);
    std::valarray<float> angleArray(300);
    std::valarray<float> kalmanArray(300);

    LaneDetector laneDetector;
    cv::Mat frame;
    cap >> frame;
    while (!frame.empty()) {
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      cv::Mat image = laneDetector.runCurvePipeline(frame);
      std::chrono::steady_clock::time_point end = 
        std::chrono::steady_clock::now();
      float duration = std::chrono::duration_cast<
        std::chrono::microseconds>(end - begin).count();
      int FPS = (1/(duration))*1000000;
      fpsMean[fpsMean.size()-1] = FPS;
      angleArray[angleArray.size()-1] = laneDetector.getSteeringAngle();
      kalmanArray[kalmanArray.size()-1] = laneDetector.getFilteredSteeringAngle();
      cv::putText(image, std::to_string(fpsMean.sum()/fpsMean.size()) + " FPS", cv::Point(50, 90), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::putText(image, std::to_string(laneDetector.getFilteredSteeringAngle()), cv::Point(50, 130), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::imshow("Lane", image);
      fpsMean = fpsMean.shift(1);
      angleArray = angleArray.shift(1);
      kalmanArray = kalmanArray.shift(1);
      cap >> frame;
      if(cv::waitKey(1) >= 0)
        break;

      std::vector<double> angleArrayV(angleArray.size());
      for (int i = 0; i < angleArray.size(); ++i)
      {
        angleArrayV[i] = angleArray[i];
      }
      cv::imshow("raw values", PlotGraph(angleArrayV));

      std::vector<double> kalmanArrayV(kalmanArray.size());
      for (int i = 0; i < kalmanArray.size(); ++i)
      {
        kalmanArrayV[i] = kalmanArray[i];
      }
      cv::imshow("filtered values", PlotGraph(kalmanArrayV));
    }

    return 0;
}

cv::Mat3b PlotGraph(std::vector<double> data) {

  cv::Mat plot_result;
  cv::Ptr<cv::plot::Plot2d> plot = cv::plot::Plot2d::create(data);
  plot->setPlotBackgroundColor(cv::Scalar(50, 50, 50)); 
  plot->setPlotLineColor(cv::Scalar(50, 50, 255));
  plot->render(plot_result);          

  return plot_result;
}