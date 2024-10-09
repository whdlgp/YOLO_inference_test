#include "backend_opencv_darknet.hpp"

#include <opencv2/dnn.hpp>

// Initialize the model
void BackendOpenCVDarknet::init(nlohmann::json init_params)
{
    std::string config_path, weights_path;
    if (!check_and_get(init_params, "config_path", config_path))
        throw std::invalid_argument("wrong config_path");
    if (!check_and_get(init_params, "weights_path", weights_path))
        throw std::invalid_argument("wrong weights_path");
    if (!check_and_get(init_params, "inference_width", this->inference_width))
        throw std::invalid_argument("wrong inference_width");
    if (!check_and_get(init_params, "inference_height", this->inference_height))
        throw std::invalid_argument("wrong inference_height");

    bool cuda = false, fp16 = false;
    check_and_get(init_params, "cuda", cuda);
    check_and_get(init_params, "fp16", fp16);
    
    // Load Net (OpenCV DNN)
    net = cv::dnn::readNetFromDarknet(config_path, weights_path);
    if (cuda)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        if (fp16)
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        else
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        if (fp16)
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU_FP16);
        else
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

// Perform inference
// Input shape must be [NUM IMAGE, HEIGHT, WIDTH, CHAN]
// Only Support NUM IMAGE = 1, CHAN = 3
std::vector<Matrix<float>> BackendOpenCVDarknet::infer(Image& input)
{
    // Input shape must be [NUM IMAGE, HEIGHT, WIDTH, CHAN]
    // Only Support NUM IMAGE = 1, CHAN = 3
    if (!(input.width > 0 ) || !(input.height > 0 )  || !(input.chan > 0 ))
        throw std::invalid_argument("Invalid Image");
    
    // CV_8UC1, CV_8UC3, CV_8UC4
    int type = (input.chan == 1) ? CV_8UC1 :
               (input.chan == 3) ? CV_8UC3 :
               (input.chan == 4) ? CV_8UC4 : -1;
    if (type == -1)
    {
        throw std::invalid_argument("Unsupported number of channels");
    }

    cv::Mat image(input.height, input.width, type, input.data.data());

    // Preprocess image using OpenCV DNN's blobFromImage
    bool swap = false;
    if (input.color == Color::BGR)
        swap = true;
    else if (input.color == Color::RGB)
        swap = false;
    else
        throw std::invalid_argument("Unsupported Color Space");

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(this->inference_width, this->inference_height), cv::Scalar(), swap, false);

    // Set the input to the network
    net.setInput(blob);

    // Perform inference
    std::vector<cv::Mat> net_outputs;
    net.forward(net_outputs, net.getUnconnectedOutLayersNames());

    std::vector<Matrix<float>> results;
    results.reserve(net_outputs.size());
    for (cv::Mat& net_output : net_outputs)
    {
        // Reserve memory for the output vector
        Matrix<float> output;
        output.data.resize(net_output.total());
        output.shape.reserve(net_output.dims);
        
        // Copy Buffer
        std::memcpy(output.data.data(), net_output.data, net_output.total() * sizeof(float));

        // Copy Shape
        for (int i = 0; i < net_output.dims; i++)
            output.shape.push_back(net_output.size[i]);

        results.push_back(output);
    }

    return results;
}