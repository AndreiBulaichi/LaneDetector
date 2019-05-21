#include "LaneDetector.hpp"
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <valarray>
#include <polyfit/polyfit.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

constexpr double PI(std::acos(-1));
constexpr uint16_t lineThreshold(50);
constexpr uint16_t maxIterations(10);
constexpr uint16_t histHeight(100);
constexpr uint16_t slices = 15;

LaneDetector::LaneDetector(float resizeRatio , uint16_t width, uint16_t height):
    resizeRatio(resizeRatio),
    width(width*resizeRatio),
    height(height*resizeRatio),
    angleFilter(0.5,50,100,0),
    steeringAngle(0),
    steeringAngleFiltered(0),
    lineFilters(6, Kalman(0.8,32,50,0))
{
   ransac.Initialize(lineThreshold, maxIterations);

   quadA[0] = cv::Point2f(this->width/2 - this->width / 16, this->height/1.6);
   quadA[1] = cv::Point2f(this->width/2 + this->width / 16, this->height/1.6);
   quadA[2] = cv::Point2f(this->width, this->height);
   quadA[3] = cv::Point2f(0, this->height);

   quadB[0] = cv::Point2f(0, 0);
   quadB[1] = cv::Point2f(this->width-1, 0);
   quadB[2] = cv::Point2f(this->width-1, this->height-1);
   quadB[3] = cv::Point2f(0, this->height-1);
}

void LaneDetector::convertToGrayscale(cv::Mat& image)
{
  static vector<cv::Mat1b> channels;
  cvtColor(image, image,  cv::COLOR_RGB2HLS);
  split(image, channels);
  GaussianBlur(channels[1], channels[0], cv::Size(3, 3), 0, 0);
  threshold(channels[0], channels[0], 180, 255, cv::THRESH_BINARY);
  threshold(channels[1], channels[1], 110, 255, cv::THRESH_BINARY);
  threshold(channels[2], channels[2], 80 , 255, cv::THRESH_BINARY);
  bitwise_and(channels[1], channels[2], image);
  bitwise_or(channels[0], image, image);
}

void LaneDetector::transformPerspective(cv::Mat& image)
{
  warpPerspective(image, image, cv::getPerspectiveTransform(quadA, quadB),
      image.size(), cv::INTER_NEAREST);
}

void LaneDetector::inversePerspective(cv::Mat& image)
{
  warpPerspective(image, image, cv::getPerspectiveTransform(quadB, quadA),
      image.size(), cv::INTER_NEAREST);
}

vector<uint16_t>* LaneDetector::calcHistogram(cv::Mat& image)
{
  static vector<uint16_t> histogram(image.cols);

  for(uint16_t i = 0; i < image.cols; ++i)
    histogram[i] = countNonZero(image.col(i));

  return &histogram;
}

cv::Mat LaneDetector::plotHistogram(std::vector<uint16_t>* histogram)
{
  cv::Mat histogramImage(histHeight, width, CV_8UC3, cv::Scalar(0, 0, 0));

  for(uint16_t i = 0; i < width; ++i){
    line(histogramImage, cv::Point(i-1, histHeight-(*histogram)[i-1]),
      cv::Point(i, histHeight-(*histogram)[i]), cv::Scalar(255, 0, 0), 2, 8, 0);
  }
  return histogramImage;
}

std::vector<std::vector<uint16_t>> LaneDetector::calcLanePoints(cv::Mat& image)
{
  // TODO : need to handle RANSAC failure, to few points
  const uint16_t middle = (image.cols)/2;
  const uint16_t sliceSize = (image.rows)/slices;
  uint16_t maxLeftPos,maxRightPos;
  vector<uint16_t>* histogram;
  vector<uint16_t> leftVectorX, leftVectorY, rightVectorX, rightVectorY;
  vector<shared_ptr<GRANSAC::AbstractParameter>> leftPoints;
  vector<shared_ptr<GRANSAC::AbstractParameter>> rightPoints;

  static cv::Mat crop = cv::Mat::zeros(sliceSize, image.cols, CV_8UC1);

  for (uint16_t sliceNum = 0; sliceNum < slices; ++sliceNum)
  {
    crop = cv::Mat(image, cv::Rect(0, sliceNum * sliceSize, image.cols, sliceSize));
    histogram = calcHistogram(crop);
    maxLeftPos = distance(histogram->begin(), max_element(histogram->begin(), histogram->begin() + middle));
    maxRightPos = distance(histogram->begin() + middle, max_element(histogram->begin() + middle, histogram->end()));

    const uint16_t y = (sliceNum * sliceSize) + 0.5 * sliceSize;

    if ((maxLeftPos != middle - 1) && (maxLeftPos != 0))
      leftPoints.push_back(std::make_shared<Point2D>(maxLeftPos, y));
    if ((maxRightPos != middle - 1) && (maxRightPos != 0))
      rightPoints.push_back(std::make_shared<Point2D>(maxRightPos + middle, y));
  }

  if(ransac.Estimate(leftPoints)){
    auto leftInliers = ransac.GetBestInliers();
    for (auto& inliner : leftInliers)
    {
      auto point = std::dynamic_pointer_cast<Point2D>(inliner);
      leftVectorX.push_back(static_cast<uint16_t>(point->m_Point2D[0]));
      leftVectorY.push_back(static_cast<uint16_t>(point->m_Point2D[1]));
    }
  }


  if(ransac.Estimate(rightPoints)){
    auto rightInliers = ransac.GetBestInliers();
    for (auto& inliner : rightInliers)
    {
      auto point = std::dynamic_pointer_cast<Point2D>(inliner);
      rightVectorX.push_back(static_cast<uint16_t>(point->m_Point2D[0]));
      rightVectorY.push_back(static_cast<uint16_t>(point->m_Point2D[1]));
    }
  }

  vector<vector<uint16_t>> ret{
  leftVectorX, leftVectorY, rightVectorX, rightVectorY};
  return ret;
}

void LaneDetector::plotLanePoints(vector<vector<float>>* points, cv::Mat& image)
{
  if(points != nullptr){
    auto pointSize = floor(image.rows / 100);
    for (uint16_t i = 0; i < (*points)[0].size(); ++i)
    {
    circle(image, cv::Point((*points)[0][i],(*points)[1][i]),
      pointSize, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
    for (uint16_t i = 0; i < (*points)[2].size(); ++i)
    {
    circle(image, cv::Point((*points)[2][i],(*points)[3][i]),
      pointSize, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
  }
}

std::vector<std::vector<float>>* LaneDetector::fitLanePoints(std::vector<std::vector<uint16_t>> points, cv::Mat& image)
{
   auto extrapolationSamples = image.rows / 15;
   auto lx = points[1], ly = points[0], rx = points[3], ry = points[2];

   xMiddle.clear();
   yMiddle.clear();

   for (auto pt : points){
     if(pt.empty())
       return nullptr;
   }

   map<uint16_t, uint16_t> leftPointsMap, rightPointsMap;
   static std::vector<float> x, y, leftInterpX, leftInterpY, rightInterpX, rightInterpY;
   x.clear(); y.clear();
   leftInterpX.clear();
   leftInterpY.clear();
   rightInterpX.clear();
   rightInterpY.clear();
   // TODO : track 2nd grade coeffs with kalman
   // 2nd degree polynomial interpolation
   for(uint16_t i = 0; i < lx.size(); ++i)
      leftPointsMap[lx[i]] = ly[i];
   for(uint16_t i = 0; i < rx.size(); ++i)
      rightPointsMap[rx[i]] = ry[i];

   for (auto i = leftPointsMap.begin(); i != leftPointsMap.end(); ++i)
   {
      x.push_back(i->first); y.push_back(i->second);
   }
   for (uint16_t i = leftPointsMap.begin()->first; i < (--leftPointsMap.end())->first; ++i)
      leftInterpX.push_back(i);

   auto coefs = polyfit(x, y, 2);
   // kalman filtering
   coefs[0] = lineFilters[0].getFilteredValue(coefs[0]);
   coefs[1] = lineFilters[1].getFilteredValue(coefs[1]);
   coefs[2] = lineFilters[2].getFilteredValue(coefs[2]);
   leftInterpY = polyval(coefs, leftInterpX);
   
   x.clear(); y.clear();
   for (auto i = rightPointsMap.begin(); i != rightPointsMap.end(); ++i)
   {
      x.push_back(i->first); y.push_back(i->second);
   }
   for (uint16_t i = rightPointsMap.begin()->first; i < (--rightPointsMap.end())->first; ++i)
      rightInterpX.push_back(i);

   coefs = polyfit(x, y, 2);

   // kalman filtering
   coefs[0] = lineFilters[3].getFilteredValue(coefs[0]);
   coefs[1] = lineFilters[4].getFilteredValue(coefs[1]);
   coefs[2] = lineFilters[5].getFilteredValue(coefs[2]);

   rightInterpY = polyval(coefs, rightInterpX);

   // 1nd degree polynomial extrapolation
   // left upmost corner extrapolation
   x.clear(); y.clear();
   x.insert(x.begin(), leftInterpX.begin(), leftInterpX.begin() + extrapolationSamples);
   y.insert(y.begin(), leftInterpY.begin(), leftInterpY.begin() + extrapolationSamples);
   std::vector<float> X;
   for (uint16_t i = 0; i < x[0]; ++i)
     X.push_back(i);
   coefs = polyfit(x, y, 1);
   auto Y = polyval(coefs, X);
   leftInterpX.insert(leftInterpX.begin(), X.begin(), X.end());
   leftInterpY.insert(leftInterpY.begin(), Y.begin(), Y.end());
   // left downmoast corner extrapolation
   x.clear(); y.clear(); X.clear(); Y.clear();
   x.insert(x.begin(), (leftInterpX.begin() + (leftInterpX.size() - extrapolationSamples)), leftInterpX.end());
   y.insert(y.begin(), (leftInterpY.begin() + (leftInterpY.size() - extrapolationSamples)), leftInterpY.end());
   for (uint16_t i = x.back() + 1; i < image.rows; ++i)
     X.push_back(i);
   coefs = polyfit(x, y, 1);
   Y = polyval(coefs, X);
   leftInterpX.insert(leftInterpX.end(), X.begin(), X.end());
   leftInterpY.insert(leftInterpY.end(), Y.begin(), Y.end());
   // right upmost corner extrapolation
   x.clear(); y.clear(); X.clear(); Y.clear();
   x.insert(x.begin(), rightInterpX.begin(), rightInterpX.begin() + extrapolationSamples);
   y.insert(y.begin(), rightInterpY.begin(), rightInterpY.begin() + extrapolationSamples);
   for (uint16_t i = 0; i < x[0]; ++i)
     X.push_back(i);
   coefs = polyfit(x, y, 1);
   Y = polyval(coefs, X);
   rightInterpX.insert(rightInterpX.begin(), X.begin(), X.end());
   rightInterpY.insert(rightInterpY.begin(), Y.begin(), Y.end());
   // right downmoast corner extrapolation
   x.clear(); y.clear(); X.clear(); Y.clear();
   x.insert(x.begin(), rightInterpX.begin() + (rightInterpX.size() - extrapolationSamples), rightInterpX.end());
   y.insert(y.begin(), rightInterpY.begin() + (rightInterpY.size() - extrapolationSamples), rightInterpY.end());
   for (uint16_t i = x.back() + 1; i < image.rows; ++i)
     X.push_back(i);
   coefs = polyfit(x, y, 1);
   Y = polyval(coefs, X);
   rightInterpX.insert(rightInterpX.end(), X.begin(), X.end());
   rightInterpY.insert(rightInterpY.end(), Y.begin(), Y.end());

   for (int i = 0; i < image.rows; ++i)
   {
     xMiddle.push_back((rightInterpY[i]+leftInterpY[i])/2);
     yMiddle.push_back(rightInterpX[i]);
   }
   static std::vector<std::vector<float>> ret;
   ret.clear();
   ret = { leftInterpY, leftInterpX, rightInterpY, rightInterpX};
   return &ret;
}

void LaneDetector::calcSteeringAngle(cv::Mat& image, bool centerCompensation = false,
    bool plot = false)
{
  if(!xMiddle.empty() && !yMiddle.empty())
  {
    /*
    * B : base center point
    * R : reference center point
    * T : lane target point
    */
    auto pointSize = floor(height / 80);
    float tgAlpha = 0;

    int16_t Bx = xMiddle[height / 1.05];
    int16_t By = height / 1.05;
    int16_t Rx = xMiddle[height / 1.05];
    int16_t Ry = height/3;
    int16_t Tx = xMiddle[height/3];
    int16_t Ty = height/3;

    if (plot)
    {
      cv::circle(image, cv::Point(Bx,By),
       pointSize, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
      cv::circle(image, cv::Point(Rx,Ry),
       pointSize, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
      cv::circle(image, cv::Point(Tx,Ty),
       pointSize, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      cv::arrowedLine(image,cv::Point(Bx,By), cv::Point(Tx,Ty),
       cv::Scalar(255, 255, 255), 2, 8, 0, 0.03);
      cv::line(image,cv::Point(Bx,By), cv::Point(Rx,Ry),
       cv::Scalar(0, 255, 255), 2, cv::LINE_4 , 0);
    }

    float RT = Tx-Rx;
    float RB = By-Ry;
    if (RB != 0)
    {
      tgAlpha = RT/RB;
    }
    if (centerCompensation)
    {
      /*
      * C : center of image aligned with B
      * K : reference of C center point
      */
      uint16_t middle = width/2;
      int16_t Cx = middle;
      int16_t Cy = height / 1.05;
      int16_t Kx = middle;
      int16_t Ky = height/3;

      if (plot)
      {
       cv::circle(image, cv::Point(Cx,Cy),
        floor(height / 80), cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
       cv::circle(image, cv::Point(Kx,Ky),
        floor(height / 80), cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
       cv::line(image,cv::Point(Cx,Cy),
        cv::Point(Kx,Ky), cv::Scalar(0, 0, 255), 2, cv::LINE_4 , 0);
       cv::arrowedLine(image,cv::Point(Cx,Cy), cv::Point(Tx,Ty),
        cv::Scalar(255, 255, 255), 2, 8, 0, 0.03);
      }

    float tgBeta = 0;
    float KT = Tx-Kx;
    float KC = Cy-Ky;

    if (KC != 0)
    {
      tgBeta = KT/KC;
    }
    steeringAngle = (atan(tgAlpha) * 180 / PI) + (atan(tgBeta) * 180 / PI);
    steeringAngleFiltered = angleFilter.getFilteredValue(steeringAngle);

    return;
    }
    steeringAngle = atan(tgAlpha) * 180 / PI;
    steeringAngleFiltered = angleFilter.getFilteredValue(steeringAngle);
  }
}

float LaneDetector::getSteeringAngle()
{
  return steeringAngle;
}

float LaneDetector::getFilteredSteeringAngle()
{
  return steeringAngleFiltered;
}

cv::Mat* LaneDetector::runCurvePipeline(cv::Mat& input)
{
   static cv::Mat image;
   resize(input, image, cv::Size(), resizeRatio, resizeRatio);
   transformPerspective(image);
   convertToGrayscale(image);
   auto fittedPoints = fitLanePoints(calcLanePoints(image),image);
   image = cv::Mat::zeros(height, width, CV_8UC3);
   plotLanePoints(fittedPoints, image);
   calcSteeringAngle(image, true, true);
   inversePerspective(image);
   resize(image, image, cv::Size(), 1/resizeRatio, 1/resizeRatio);
   addWeighted(image, 1, input, 1, 0.0, image);
   return &image;
}

void LaneDetector::runLightCurvePipeline(cv::Mat& input)
{
   resize(input, input, cv::Size(), resizeRatio, resizeRatio);
   transformPerspective(input);
   convertToGrayscale(input);
   fitLanePoints(calcLanePoints(input),input);
   calcSteeringAngle(input, true, false);
}
