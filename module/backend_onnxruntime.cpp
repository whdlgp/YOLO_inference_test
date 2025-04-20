#include "backend_onnxruntime.hpp"

#include <iostream>

Ort::Env BackendONNXRuntime::env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "BackendONNXRuntime");

// Initialize the model
void BackendONNXRuntime::init(nlohmann::json init_params)
{
    std::string onnx_path;
    if (!check_and_get(init_params, "onnx_path", onnx_path))
        throw std::invalid_argument("wrong onnx_path");
    if (!check_and_get(init_params, "inference_width", this->inference_width))
        throw std::invalid_argument("wrong inference_width");
    if (!check_and_get(init_params, "inference_height", this->inference_height))
        throw std::invalid_argument("wrong inference_height");
    
    bool use_tensorrt = false, use_fp16 = false, fallback_cuda = false;
    check_and_get(init_params, "tensorrt", use_tensorrt);
    check_and_get(init_params, "fp16",     use_fp16);
    check_and_get(init_params, "cuda",     fallback_cuda);

    // Initialize ONNX Runtime
    
    // Search available providers
    auto providers = Ort::GetAvailableProviders();

    // TensorRT provider
    if (use_tensorrt &&
        std::find(providers.begin(), providers.end(), "TensorrtExecutionProvider") != providers.end())
    {
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
        api.CreateTensorRTProviderOptions(&trt_opts);

        std::vector<const char*> keys, values;
        if (use_fp16)
        {
            keys .push_back("trt_fp16_enable");
            values.push_back("1");
        }
        api.UpdateTensorRTProviderOptions(trt_opts, keys.data(), values.data(), keys.size());
        session_options.AppendExecutionProvider_TensorRT_V2(*trt_opts);
        api.ReleaseTensorRTProviderOptions(trt_opts);
    }

    // CUDA fallback
    if (fallback_cuda &&
        std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end())
    {
        OrtCUDAProviderOptions cuda_opts;
        session_options.AppendExecutionProvider_CUDA(cuda_opts);
    }

    session = Ort::Session(env, onnx_path.c_str(), session_options);

    size_t input_count = session.GetInputCount();
    input_names.resize(input_count);
    input_names_cstr.resize(input_count);
    input_shapes.resize(input_count);
    for (size_t i = 0; i < input_count; i++)
    {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
        input_names[i] = name.get();
        input_names_cstr[i] = input_names[i].c_str();

        auto info = session.GetInputTypeInfo(i);
        input_shapes[i] = info.GetTensorTypeAndShapeInfo().GetShape();
    }

    size_t output_count = session.GetOutputCount();
    output_names.resize(output_count);
    output_names_cstr.resize(output_count);
    output_shapes.resize(output_count);
    for (size_t i = 0; i < output_count; i++)
    {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
        output_names[i] = name.get();
        output_names_cstr[i] = output_names[i].c_str();

        auto info = session.GetOutputTypeInfo(i);
        output_shapes[i] = info.GetTensorTypeAndShapeInfo().GetShape();
    }

    mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Currently, Onlt support one input
    const std::vector<int64_t>& shape = input_shapes[0];
    size_t total_size = 1;
    for (auto s : shape) total_size *= s;

    reusable_input_data.resize(total_size);
    reusable_input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, reusable_input_data.data(), total_size, shape.data(), shape.size());

    run_options = Ort::RunOptions{ nullptr };

    init_onnxruntime = true;
}

// Perform inference
// Input shape must be [NUM IMAGE, HEIGHT, WIDTH, CHAN]
// Only Support NUM IMAGE = 1, CHAN = 3
std::vector<Matrix<float>> BackendONNXRuntime::infer(Image& input)
{
    // Check init
    if (!init_onnxruntime)
        throw std::runtime_error("ONNXRuntime session is not initialized.");

    // Input shape must be [NUM IMAGE, HEIGHT, WIDTH, CHAN]
    // Only Support NUM IMAGE = 1, CHAN = 3
    if (!(input.width > 0 ) || !(input.height > 0 )  || !(input.chan > 0 ))
        throw std::invalid_argument("Invalid Image");
    
    // BGR image only
    if (input.chan != 3)
        throw std::invalid_argument("Unsupported number of channels");

    // Preprocess the image
    std::vector<float> preprocessed = RGB2NCHW(input.data, input.width, input.height, input.chan, inference_width, inference_height);

    // Copy to reusable memory
    std::copy(preprocessed.begin(), preprocessed.end(), reusable_input_data.begin());

    // Run inference
    auto output_tensors = session.Run(run_options,
        input_names_cstr.data(), &reusable_input_tensor, 1,
        output_names_cstr.data(), output_names_cstr.size());

    // Get output tensor shape
    // (Assuming single output)
    Ort::Value& output_tensor = output_tensors.front(); 
    std::vector<int64_t> output_shape = output_shapes[0];

    // Prepare result
    std::vector<Matrix<float>> results(1);

    // Copy Shape
    size_t total_size = 1;
    for (auto s : output_shape) total_size *= s;
    for (auto s : output_shape) results[0].shape.push_back(s);

    // Convert the output tensor to vector of floats
    float* output_data = output_tensor.GetTensorMutableData<float>();
    results[0].data.assign(output_data, output_data + total_size);
    
    return results;
}