#pragma once

#include "interface.hpp"
#include <optional>
#include <onnxruntime_cxx_api.h>

// Imple of DNN Inference with ONNXRuntime
class BackendONNXRuntime : public BackendBase<float>
{
public:
    // Initialize the model
    void init(nlohmann::json init_params) override;

    // Perform inference
    std::vector<Matrix<float>> infer(Image& input) override;
private:
    // ONNX Runtime
    static Ort::Env env;
    std::optional<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::optional<Ort::TypeInfo> input_type_info;
    std::optional<Ort::TypeInfo> output_type_info;
    ONNXTensorElementDataType input_tensor_type;
    ONNXTensorElementDataType output_tensor_type;
    // check init
    bool init_onnxruntime = false;

    // Model Input Shape
    int inference_width = 608;
    int inference_height = 608;
};