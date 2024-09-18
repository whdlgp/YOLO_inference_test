#include "dnn_model_yolov5.hpp"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <stdexcept>
#include <cstring>

void YOLOv5Model::init(nlohmann::json init_params)
{
    // Set confidence threshold, NMS threshold, width, height
    ObjectDetection::init(init_params);

    std::string onnx_path, names_file;
    if (!check_and_get(init_params, "onnx_path", onnx_path))
        throw std::invalid_argument("wrong onnx_path");
    if (!check_and_get(init_params, "names_file", names_file))
        throw std::invalid_argument("wrong names_file");

    // Load Net (OpenCV DNN)
    net = cv::dnn::readNetFromONNX(onnx_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    // Load class names
    std::ifstream class_file(names_file);
    if (!class_file)
    {
        throw std::runtime_error("failed to open classes.txt");
    }

    std::string line;
    while (std::getline(class_file, line))
    {
        class_names.push_back(line);
    }

    num_classes = static_cast<int>(class_names.size());
}

// Perform inference
std::vector<ObjectDetection::Output> YOLOv5Model::infer(ObjectDetection::Input& input)
{
    // Preprocess the input
    std::vector<uint8_t> preprocessed_input = preprocess(input);

    // Inference using the backend
    std::vector<float> inference_output = infer(preprocessed_input);

    // Postprocess the inference results
    std::vector<ObjectDetection::Output> outputs = postprocess(inference_output, input.width, input.height);

    return outputs;
}

// Preprocess input data
std::vector<uint8_t> YOLOv5Model::preprocess(const ObjectDetection::Input& input)
{
    // Convert ObjectDetection::Input to cv::Mat
    int type = (input.chan == 1) ? CV_8UC1 :
               (input.chan == 3) ? CV_8UC3 :
               (input.chan == 4) ? CV_8UC4 : -1;

    if (type == -1)
    {
        throw std::invalid_argument("Unsupported number of channels");
    }

    cv::Mat image(input.height, input.width, type);
    std::memcpy(image.data, input.data.data(), input.data.size() * sizeof(uint8_t));

    // Preprocess image using OpenCV DNN's blobFromImage
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(this->inference_width, this->inference_height), cv::Scalar(), true, false);

    // Flatten cv::Mat into std::vector<uint8_t> to match the interface
    std::vector<uint8_t> flattened_blob(blob.total() * blob.elemSize());
    std::memcpy(flattened_blob.data(), blob.data, flattened_blob.size());

    return flattened_blob;
}

// Perform inference using backend (OpenCV DNN inference)
std::vector<float> YOLOv5Model::infer(const std::vector<uint8_t>& preprocessed_input)
{
    // Get shape of the blob (Assuming NCHW format from blobFromImage)
    int batch_size = 1;  // Assuming batch size of 1
    int channels = 3;    // Assuming 3 channels (RGB)
    int height = this->inference_height; // Height used in blobFromImage
    int width = this->inference_width;   // Width used in blobFromImage

    // Calculate the total number of elements
    int total_size = batch_size * channels * height * width;

    // Ensure the size matches the input
    if (preprocessed_input.size() != total_size * sizeof(float))
        throw std::invalid_argument("preprocessed input size does not match expected size");

    // Create the 4D blob with the correct shape
    std::vector<int> blob_shape = {batch_size, channels, height, width};
    cv::Mat blob(4, blob_shape.data(), CV_32F, const_cast<uint8_t*>(preprocessed_input.data()));

    // Set the input to the network
    net.setInput(blob);

    // Perform inference
    std::vector<cv::Mat> net_output;
    net.forward(net_output, net.getUnconnectedOutLayersNames());

    // Calc total size of output mats
    size_t total_output_size = 0;
    for (const auto& mat : net_output)
        total_output_size += mat.total();

    // Reserve memory for the output vector
    std::vector<float> output;
    output.reserve(total_output_size);

    // Convert the output from cv::Mat to std::vector<float>
    for (const auto& mat : net_output)
    {
        const float* data = mat.ptr<float>();
        output.insert(output.end(), data, data + mat.total());
    }

    return output;
}

// Postprocess the inference results
std::vector<ObjectDetection::Output> YOLOv5Model::postprocess(const std::vector<float>& inference_output, int origin_width, int origin_height)
{
    std::vector<ObjectDetection::Output> final_results;
    std::vector<std::vector<cv::Rect>> boxes_per_class(num_classes);
    std::vector<std::vector<float>> confidences_per_class(num_classes);
    std::vector<std::vector<int>> indices(num_classes);

    // YOLOv5 output is [1, 25200, 85] where 85 = 4 (bbox) + 1 (obj_conf) + 80 (class_conf)
    const int num_boxes = inference_output.size() / (5 + num_classes);
    
    // Scale factor for resizing bounding boxes back to the original size
    const float scale_x = static_cast<float>(origin_width) / inference_width;
    const float scale_y = static_cast<float>(origin_height) / inference_height;

    for (int i = 0; i < num_boxes; ++i)
    {
        const float* data = &inference_output[i * (5 + num_classes)];

        float x_center = data[0] * scale_x;
        float y_center = data[1] * scale_y;
        float width = data[2] * scale_x;
        float height = data[3] * scale_y;
        float obj_conf = data[4];  // Objectness confidence

        cv::Rect box(
            static_cast<int>(x_center - width / 2),
            static_cast<int>(y_center - height / 2),
            static_cast<int>(width),
            static_cast<int>(height)
        );

        // Iterate over class confidences
        for (int c = 0; c < num_classes; ++c)
        {
            float class_conf = data[5 + c];
            float confidence = obj_conf * class_conf;

            if (confidence >= confidence_threshold)
            {
                boxes_per_class[c].push_back(box);
                confidences_per_class[c].push_back(confidence);
            }
        }
    }

    // Apply Non-Maximum Suppression (NMS) and populate final results
    for (int c = 0; c < num_classes; ++c)
    {
        cv::dnn::NMSBoxes(boxes_per_class[c], confidences_per_class[c], confidence_threshold, nms_threshold, indices[c]);

        for (int idx : indices[c])
        {
            ObjectDetection::Output output;
            output.bbox_x = boxes_per_class[c][idx].x;
            output.bbox_y = boxes_per_class[c][idx].y;
            output.bbox_width = boxes_per_class[c][idx].width;
            output.bbox_height = boxes_per_class[c][idx].height;
            output.class_id = c;
            output.class_name = class_names[c];
            output.confidence = confidences_per_class[c][idx];

            final_results.push_back(output);
        }
    }

    return final_results;
}
