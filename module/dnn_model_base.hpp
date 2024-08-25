#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <any>
#include <vector>

// Basic model class interface
class DNNModel
{
public:
    using InitParams = std::any;

    virtual ~DNNModel() {}

    // Initialize the model
    virtual void init(const InitParams &model_files, float confidence_threshold = 0.5, float nms_threshold = 0.4) = 0;

    // Preprocess the image
    virtual cv::Mat preprocess(const cv::Mat &image) = 0;

    // Perform inference
    virtual std::vector<cv::Mat> infer(const cv::Mat &blob) = 0;

    // Postprocess the inference results
    virtual void postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs) = 0;

    // Utility function to draw the results
    virtual void draw_results(cv::Mat &frame) = 0;
};