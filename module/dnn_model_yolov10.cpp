#include "dnn_model_yolov10.hpp"

#include <fstream>
#include <opencv2/opencv.hpp>

void YOLOv10Model::init(const InitParams &model_files
                        , float confidence_threshold
                        , float nms_threshold
                        , int inference_width
                        , int inference_height)
{
    // Set confidence threshold, NMS threshold, width, height
    ObjectDetection::init(model_files, confidence_threshold, nms_threshold, inference_width, inference_height);

    if (model_files.type() == typeid(std::tuple<std::string, std::string>))
    {
        auto [onnx_path, names_file] = std::any_cast<std::tuple<std::string, std::string>>(model_files);
        net = cv::dnn::readNetFromONNX(onnx_path);
        // Currently, There some problem with OPENCV 4.10.0 CUDA Backend
        //net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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

        // Set the number of classes based on the size of class_names
        num_classes = static_cast<int>(class_names.size());
    }
    else
    {
        throw std::logic_error("Darknet model requires config, weights, and class names paths");
    }
}

// Perform inference
std::vector<ObjectDetection::Output> YOLOv10Model::infer(ObjectDetection::Input& input)
{
    // Convert to OpenCV Mat
    cv::Mat image = convert_to_Mat(input);

    // BGR24 to NCHW (N = 1)
    cv::Mat blob = preprocess(image);

    // Inference
    std::vector<cv::Mat> results = infer(blob);

    // Post process
    std::vector<ObjectDetection::Output> outputs = postprocess(image, results);

    return outputs;
}

// Convert Object Detection input to cv::Mat
cv::Mat YOLOv10Model::convert_to_Mat(const ObjectDetection::Input& input)
{
    // CV_8UC3, CV_8UC1 
    int type = (input.chan == 1) ? CV_8UC1 :
               (input.chan == 3) ? CV_8UC3 :
               (input.chan == 4) ? CV_8UC4 : -1;
    if (type == -1)
    {
        throw std::invalid_argument("Unsupported number of channels");
    }

    // cv::Mat 
    cv::Mat mat(input.height, input.width, type);

    // data copy
    std::memcpy(mat.data, input.data.data(), input.data.size() * sizeof(uint8_t));

    return mat;
}

// Preprocess the image
cv::Mat YOLOv10Model::preprocess(const cv::Mat &image)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 0.00392, cv::Size(this->inference_width, this->inference_height), cv::Scalar(), true, false, CV_32F);
    return blob;
}

// Perform inference
std::vector<cv::Mat> YOLOv10Model::infer(const cv::Mat &blob)
{
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    auto output_names = net.getUnconnectedOutLayersNames();
    net.forward(outs, output_names);
    return outs;
}

// Postprocess the inference results
std::vector<ObjectDetection::Output> YOLOv10Model::postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
    // Latest YOLOv10 no need to use NMS
    // But, OpenCV not work with original YOLOv10
    // OpenCV need modified version of YOLOv10 until fix this issue
    /*
    // YOLOv10 output is [1, 300, 6] where 300 bbox with 6 = [left, top, right, bottom, conf, class id]
    const int num_boxes = outs[0].size[1]; // 300

    // Scale factor for resize BBox
    const float scale_x = static_cast<float>(frame.cols) / inference_width;
    const float scale_y = static_cast<float>(frame.rows) / inference_height;

    // Final Result output
    // YOLO v10 no need to do NMS
    std::vector<ObjectDetection::Output> final_results;
    final_results.reserve(20);
    for (int i = 0; i < num_boxes; i++)
    {
        const float* data = outs[0].ptr<float>(i);
        float left = data[i * 6 + 0];
        float top = data[i * 6 + 1];
        float right = data[i * 6 + 2];
        float bottom = data[i * 6 + 3];
        float confidence = data[i * 6 + 4];
        int class_id = static_cast<int>(data[i * 6 + 5]);

        if (confidence >= confidence_threshold)
        {
            // Create an Output object and populate it
            ObjectDetection::Output output;
            output.bbox_x = static_cast<int>(left * scale_x);
            output.bbox_y = static_cast<int>(top * scale_y);
            output.bbox_width = static_cast<int>((right - left) * scale_x);
            output.bbox_height = static_cast<int>((bottom - top) * scale_y);
            output.class_id = class_id;
            output.class_name = class_names[class_id];  // Assumes you have class_names vector available
            output.confidence = confidence;

            // Add the Output object to the outputs vector
            final_results.push_back(output);
        }
    }

    return final_results;
    */

    // Initialize vectors to store NMS results
    std::vector<std::vector<int>> indices(num_classes);
    std::vector<std::vector<cv::Rect>> boxes_per_class(num_classes);
    std::vector<std::vector<float>> confidences_per_class(num_classes);

    // modified YOLOv10 output is [1, 84, 8400] where 84 = 4 (bbox) + 80 (class_conf)
    const int num_boxes = outs[0].size[2]; // 8400
    const int num_classes = outs[0].size[1] - 4; // 80

    // Reshape to [84, 8400], Transpose to [8400, 84]
    cv::Mat output = outs[0].reshape(1, {outs[0].size[1], outs[0].size[2]});
    cv::Mat transposed_output = output.t(); 

    // Scale factor for resize BBox
    const float scale_x = static_cast<float>(frame.cols) / inference_width;
    const float scale_y = static_cast<float>(frame.rows) / inference_height;

    for (int i = 0; i < num_boxes; i++)
    {
        // Extract bounding box coordinates and confidence scores
        const float* data = transposed_output.ptr<float>(i);
        float left = data[0] * scale_x;
        float top = data[1] * scale_y;
        float right = data[2] * scale_x;
        float bottom = data[3] * scale_y;

        cv::Rect rect(
            static_cast<int>(left),
            static_cast<int>(top),
            static_cast<int>(right - left),
            static_cast<int>(bottom - top)
        );

        // Loop over the classes and gather the confidence scores
        for (int c = 0; c < num_classes; c++)
        {
            float confidence = data[4 + c];

            if (confidence >= confidence_threshold)
            {
                boxes_per_class[c].push_back(rect);
                confidences_per_class[c].push_back(confidence);
            }
        }
    }

    // Final Result output
    std::vector<ObjectDetection::Output> final_results;
    final_results.reserve(20);

    // Perform NMS and aggregate the results
    for (int c = 0; c < num_classes; c++)
    {
        cv::dnn::NMSBoxes(boxes_per_class[c], confidences_per_class[c], confidence_threshold, nms_threshold, indices[c]);

        for (auto idx : indices[c])
        {
            // Create an Output object and populate it
            ObjectDetection::Output output;
            output.bbox_x = boxes_per_class[c][idx].x;
            output.bbox_y = boxes_per_class[c][idx].y;
            output.bbox_width = boxes_per_class[c][idx].width;
            output.bbox_height = boxes_per_class[c][idx].height;
            output.class_id = c;
            output.class_name = class_names[c];  // Assumes you have class_names vector available
            output.confidence = confidences_per_class[c][idx];

            // Add the Output object to the outputs vector
            final_results.push_back(output);
        }
    }

    return final_results;

}
