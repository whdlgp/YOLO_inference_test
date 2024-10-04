#include "backend_onnxruntime.hpp"

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
    
    // Initialize ONNX Runtime
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // CUDA provider check
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // CUDA device ID 0
    if (status != nullptr)
        throw std::runtime_error("Failed to set CUDA Execution Provider");

    session = Ort::Session(env, onnx_path.c_str(), session_options);

    // Get input/output tensor type
    if (session.has_value())
    {
        input_type_info = session->GetInputTypeInfo(0);
        output_type_info = session->GetOutputTypeInfo(0);
    }

    if (input_type_info.has_value() && output_type_info.has_value())
    {
        input_tensor_type = input_type_info->GetTensorTypeAndShapeInfo().GetElementType();
        output_tensor_type = output_type_info->GetTensorTypeAndShapeInfo().GetElementType();
    }

    if (session.has_value() && input_type_info.has_value() && output_type_info.has_value())
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

    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, 3, inference_height, inference_width}; // NCHW format
    const size_t input_tensor_size = inference_width * inference_height * 3;

    // Create input tensor object from data values
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocessed.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run inference
    Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Get output tensor shape
    Ort::Value& output_tensor = output_tensors.front(); // Assuming single output
    Ort::TensorTypeAndShapeInfo output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_tensor_info.GetShape();

    // Convert the output tensor to vector of floats
    std::vector<Matrix<float>> results(1);
    float* output_data = output_tensor.GetTensorMutableData<float>();
    results[0].data.assign(output_data, output_data + output_tensor_info.GetElementCount());

    // Copy Shape
    for (int64_t elem : output_shape)
        results[0].shape.push_back(elem);
    
    return results;
}