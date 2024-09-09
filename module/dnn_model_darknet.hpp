#pragma once

#include "dnn_model_base.hpp"
#include "opencv2/dnn.hpp"

// Darknet based model class
class DarknetModel : public ObjectDetection
{
public:
    // Initialize the model
    void init(nlohmann::json init_params) override;

    // Perform inference
    virtual std::vector<Output> infer(Input& input) override;


    // Draw util for OpenCV 
    static void draw(cv::Mat &frame, std::vector<ObjectDetection::Output> output);
private:
    cv::dnn::Net net;

    // Convert Object Detection input to cv::Mat
    cv::Mat convert_to_Mat(const ObjectDetection::Input& input);

    // Preprocess the image
    cv::Mat preprocess(const cv::Mat &image);

    // Perform inference
    std::vector<cv::Mat> infer(const cv::Mat &blob);

    // Postprocess the inference results
    std::vector<ObjectDetection::Output> postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs);
};