#pragma once

#include "interface.hpp"

#include <opencv2/dnn.hpp>

// Imple of DNN Inference with OpenCV DNN Module, Darknet
class BackendOpenCVDarknet : public BackendBase<float>
{
public:
    // Initialize the model
    void init(nlohmann::json init_params) override;

    // Perform inference
    std::vector<Matrix<float>> infer(Image& input) override;
private:
    // OpenCV DNN module
    cv::dnn::Net net;

    // Model Input Shape
    int inference_width = 608;
    int inference_height = 608;
};