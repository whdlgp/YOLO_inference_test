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
    // check init
    bool init_onnxruntime = false;

    // Model Input Shape
    int inference_width = 608;
    int inference_height = 608;

    // ONNX Runtime
    static Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{ nullptr };

    Ort::MemoryInfo mem_info{ nullptr };
    Ort::RunOptions run_options{ nullptr };

    Ort::Value reusable_input_tensor{ nullptr };
    std::vector<float> reusable_input_data;

    std::vector<std::string> input_names;
    std::vector<const char*> input_names_cstr;
    std::vector<std::string> output_names;
    std::vector<const char*> output_names_cstr;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};
