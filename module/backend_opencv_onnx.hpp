#pragma once

#include "dnn_model_base.hpp"

#include <opencv2/dnn.hpp>

// Imple of DNN Inference with OpenCV DNN Module, ONNX
class BackendOpenCVONNX : public BackendBase<float>
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