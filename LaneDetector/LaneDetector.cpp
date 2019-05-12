#include "LaneDetector.hpp"
#include <string>
#include <algorithm> 
#include <vector>
#include <iostream>
#include <math.h>
#include <valarray>
#include <polyfit/polyfit.hpp>

#define PI 3.14159265

LaneDetector::LaneDetector():angleFilter(0.5,50,100,0)
{
  ransac.Initialize(50, 10);
  steeringAngle = 0;
  steeringAngleFiltered = 0;
}

void LaneDetector::convertToGrayscale(cv::Mat& image)
{ 
  const int width = image.cols; const int height = image.rows;
  cv::Mat1b white(height, width);
  std::vector<cv::Mat1b> channels;
  // Extract separate channels from hls
  cv::cvtColor(image, image,  cv::COLOR_RGB2HLS);
  split(image, channels);
  // Get the white component
  cv::GaussianBlur(channels[1], white, cv::Size(3, 3), 0, 0);
  cv::threshold(white, white, 180, 255, cv::THRESH_BINARY);
  // Get the yellow component
  cv::threshold(channels[1], channels[1], 110, 255, cv::THRESH_BINARY);
  cv::threshold(channels[2], channels[2], 80, 255, cv::THRESH_BINARY);
  // Apply yellow component
  cv::bitwise_and(channels[1], channels[2], image);
  // Apply white component
  cv::bitwise_or(white, image, image);
}

void LaneDetector::transformPerspective(cv::Mat& input)
{
    cv::Mat lambda(2, 4, CV_32FC1);
    lambda = cv::getPerspectiveTransform(quadA, quadB);
    cv::warpPerspective(input, input, lambda, input.size());
}

void LaneDetector::inversePerspective(cv::Mat& input)
{
    cv::Mat lambda(2, 4, CV_32FC1);
    lambda = cv::getPerspectiveTransform(quadB, quadA);
    cv::warpPerspective(input, input, lambda, input.size());
}

std::vector<uint16_t> LaneDetector::calcHistogramOverX(cv::Mat& image)
{
  const uint16_t width = image.cols;
  const uint16_t height = image.cols;
  std::vector<uint16_t> histogram(width);
  for(uint16_t i = 0; i < width; ++i)
    histogram[i] = countNonZero(image.col(i));
  
  std::vector<uint16_t> avghistogram(width);
  avghistogram[0] = 0;
  avghistogram[1] = 0;
  avghistogram[2] = 0;
  for(int i = 2; i < width - 2; ++i){
    avghistogram[i] = (histogram[i-2] + histogram[i-1] + histogram[i] +
      histogram[i+1] + histogram[i+2]) / 5;
  }

  return avghistogram;
}

cv::Mat LaneDetector::plotHistogram(cv::Mat& image, std::vector<uint16_t> histogram)
{
  const uint16_t width = image.cols;
  cv::Mat histogramImage(256, width, CV_8UC3, cv::Scalar(0, 0, 0));
  for(uint16_t i = 0; i < width; ++i)
    cv::line(histogramImage,
           cv::Point( i-1, 256 - histogram[i-1]),
           cv::Point( i,   256 - histogram[i]),
           cv::Scalar(255, 0, 0), 2, 8, 0);

  return histogramImage;
}

std::vector<std::vector<float>> LaneDetector::calculateLanePoints(
  cv::Mat& image)
{

  // RANSAC MIGHT BE REMOVED

  const uint16_t slices = 15;
  const uint16_t width = image.cols;
  const uint16_t height = image.rows;
  const uint16_t middle = width/2;
  const uint16_t sliceSize = height/slices;

  std::vector<float> leftVectorX(0);
  std::vector<float> leftVectorY(0);
  std::vector<float> rightVectorX(0);
  std::vector<float> rightVectorY(0);

  std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> leftPoints;
  std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> rightPoints;

  cv::Mat crop = cv::Mat::zeros(sliceSize, width, CV_8UC1);
  for (uint16_t sliceNum = 0; sliceNum < slices; ++sliceNum)
  {
    crop = cv::Mat(image, cv::Rect(0, sliceNum * sliceSize, width, sliceSize));
    auto histogram = calcHistogramOverX(crop);
    auto maxLeft = std::minmax_element(histogram.begin(), histogram.begin() + middle);
    auto maxRight = std::minmax_element(histogram.begin() + middle, histogram.end());
    uint16_t maxLeftPos = maxLeft.second - histogram.begin();
    uint16_t maxRightPos = maxRight.second - (histogram.begin() + middle);

    const uint16_t y = (sliceNum * sliceSize) + 0.5 * sliceSize;

    if ((maxLeftPos != middle - 1) && (maxLeftPos != 0)){
      leftPoints.push_back(std::make_shared<Point2D>(maxLeftPos, y));
    }
    if ((maxRightPos != middle - 1) && (maxRightPos != 0)){
      rightPoints.push_back(std::make_shared<Point2D>(maxRightPos + middle, y));
    }
  }

  ransac.Estimate(leftPoints);
  auto leftInliers = ransac.GetBestInliers();
  ransac.Estimate(rightPoints);
  auto rightInliers = ransac.GetBestInliers();

  for (auto& inliner : leftInliers)
  {
    auto point = std::dynamic_pointer_cast<Point2D>(inliner);
    leftVectorX.push_back(static_cast<float>(point->m_Point2D[0]));
    leftVectorY.push_back(static_cast<float>(point->m_Point2D[1]));
  }
  for (auto& inliner : rightInliers)
  {
    auto point = std::dynamic_pointer_cast<Point2D>(inliner);
    rightVectorX.push_back(static_cast<float>(point->m_Point2D[0]));
    rightVectorY.push_back(static_cast<float>(point->m_Point2D[1]));
  }

  std::vector<std::vector<float>> ret =
    {leftVectorX, leftVectorY, rightVectorX, rightVectorY};

  return ret;
}

cv::Mat LaneDetector::plotLanePoints(
    std::vector<std::vector<float>> points, cv::Mat grayImage)
{
  const uint16_t height = grayImage.rows;
  cv::Mat3b rgb;
  cv::cvtColor(grayImage, rgb, cv::COLOR_GRAY2RGB);

  for (auto i = 0; i < points[0].size(); i++)
  {
    cv::circle(rgb, cv::Point(points[0][i],points[1][i]), 
      floor(height / 100), cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
  }
  for (auto i = 0; i < points[2].size(); i++)
  {
    cv::circle(rgb, cv::Point(points[2][i],points[3][i]), 
      floor(height / 100), cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
  }
  return rgb;
}

std::vector<std::vector<float>> LaneDetector::fitLanePoints(
  std::vector<std::vector<float>> points, cv::Mat image)
{
  const uint16_t height = image.rows;
  const uint16_t width = image.cols;
  uint16_t extrapolationSamples = height / 15;

  auto lx = points[1], ly = points[0], rx = points[3], ry = points[2];

  std::map<float, float> leftPointsMap;
  std::map<float, float> rightPointsMap;
  std::vector<float> x, y;
  std::vector<float> leftInterpX, leftInterpY, rightInterpX, rightInterpY;
  // 2nd degree polynomial interpolation
  for(uint16_t i = 0; i < lx.size(); ++i)
    leftPointsMap[lx[i]] = ly[i];
  for (auto i = leftPointsMap.begin(); i != leftPointsMap.end(); i++)
  {
    x.push_back(i->first);
    y.push_back(i->second);
  }
  for (uint16_t i = leftPointsMap.begin()->first; i < (--leftPointsMap.end())->first; ++i)
  {
    leftInterpX.push_back(i);
  }
  auto coefs = polyfit(x, y, 2);
  leftInterpY = polyval(coefs, leftInterpX);

  x.clear(); y.clear();
  for(uint16_t i = 0; i < rx.size(); ++i)
    rightPointsMap[rx[i]] = ry[i];
  for (auto i = rightPointsMap.begin(); i != rightPointsMap.end(); i++)
  {
    x.push_back(i->first);
    y.push_back(i->second);
  }
  for (uint16_t i = rightPointsMap.begin()->first; i < (--rightPointsMap.end())->first; ++i)
    rightInterpX.push_back(i);

  coefs = polyfit(x, y, 2);
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
  for (uint16_t i = x.back() + 1; i < height; ++i)
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
  for (uint16_t i = x.back() + 1; i < height; ++i)
    X.push_back(i);
  coefs = polyfit(x, y, 1);
  Y = polyval(coefs, X);
  rightInterpX.insert(rightInterpX.end(), X.begin(), X.end());
  rightInterpY.insert(rightInterpY.end(), Y.begin(), Y.end());
  xMiddle.clear();yMiddle.clear();
  for (int i = 0; i < height; ++i)
  {
    xMiddle.push_back((rightInterpY[i]+leftInterpY[i])/2);
    yMiddle.push_back(rightInterpX[i]);
  }

  std::vector<std::vector<float>> ret =
    {leftInterpY, leftInterpX, rightInterpY, rightInterpX};
  return ret;
}

void LaneDetector::calculateSteeringAngle(cv::Mat& image, bool centerCompensation = false,
    bool plot = false)
{
  /*
  * B : base center point
  * R : reference center point
  * T : lane target point
  */
  const int height = image.rows;
  const int width = image.cols;
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
    floor(height / 80), cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
  cv::circle(image, cv::Point(Rx,Ry), 
    floor(height / 80), cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
  cv::circle(image, cv::Point(Tx,Ty), 
    floor(height / 80), cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
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

float LaneDetector::getSteeringAngle()
{
  return steeringAngle;
}

float LaneDetector::getFilteredSteeringAngle()
{
  return steeringAngleFiltered;
}

cv::Mat LaneDetector::runCurvePipeline(cv::Mat input)
{
  const float resizeRatio = 1;
  cv::Mat image;
  input.copyTo(image);
  cv::resize(image, image, cv::Size(), resizeRatio, resizeRatio);
  const int width = image.cols;
  const int height = image.rows;
  quadA[0] = cv::Point2f(width/2 - width / 16, height/1.6);
  quadA[1] = cv::Point2f(width/2 + width / 16, height/1.6);
  quadA[2] = cv::Point2f(width, height);
  quadA[3] = cv::Point2f(0, height);
  quadB[0] = cv::Point2f(0, 0);
  quadB[1] = cv::Point2f(width-1, 0);
  quadB[2] = cv::Point2f(width-1, height-1);
  quadB[3] = cv::Point2f(0, height-1);
  transformPerspective(image);
  convertToGrayscale(image);
  auto points = calculateLanePoints(image);
  points = fitLanePoints(points,image);
  image = cv::Mat::zeros(height, width, CV_8UC1);
  image = plotLanePoints(points, image);
  calculateSteeringAngle(image, true, true);
  inversePerspective(image);
  cv::resize(image, image, cv::Size(), 1/resizeRatio, 1/resizeRatio);
  addWeighted(image, 1, input, 1, 0.0, image);
  return image;
}