#include "dnn_model_darknet.hpp"

#include <fstream>
#include <opencv2/opencv.hpp>

void DarknetModel::init(const InitParams &model_files
                        , float confidence_threshold
                        , float nms_threshold
                        , int inference_width
                        , int inference_height)
{
    // Set confidence threshold, NMS threshold, width, height
    ObjectDetection::init(model_files, confidence_threshold, nms_threshold, inference_width, inference_height);

    if (model_files.type() == typeid(std::tuple<std::string, std::string, std::string>))
    {
        auto [config_path, weights_path, names_file] = std::any_cast<std::tuple<std::string, std::string, std::string>>(model_files);
        net = cv::dnn::readNetFromDarknet(config_path, weights_path);
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

        // Set the number of classes based on the size of class_names
        num_classes = static_cast<int>(class_names.size());
    }
    else
    {
        throw std::logic_error("Darknet model requires config, weights, and class names paths");
    }
}

// Perform inference
std::vector<ObjectDetection::Output> DarknetModel::infer(ObjectDetection::Input& input)
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
cv::Mat DarknetModel::convert_to_Mat(const ObjectDetection::Input& input)
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
cv::Mat DarknetModel::preprocess(const cv::Mat &image)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 0.00392, cv::Size(this->inference_width, this->inference_height), cv::Scalar(), true, false, CV_32F);
    return blob;
}

// Perform inference
std::vector<cv::Mat> DarknetModel::infer(const cv::Mat &blob)
{
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    auto output_names = net.getUnconnectedOutLayersNames();
    net.forward(outs, output_names);
    return outs;
}

// Postprocess the inference results
std::vector<ObjectDetection::Output> DarknetModel::postprocess(const cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
    // Initialize vectors to store NMS results
    std::vector<std::vector<int>> indices(num_classes);
    std::vector<std::vector<cv::Rect>> boxes_per_class(num_classes);
    std::vector<std::vector<float>> confidences_per_class(num_classes);

    for (const auto& output : outs)
    {
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < num_classes; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= confidence_threshold)
                {
                    boxes_per_class[c].push_back(rect);
                    confidences_per_class[c].push_back(confidence);
                }
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
            output.class_name = class_names[c];
            output.confidence = confidences_per_class[c][idx];

            // Add the Output object to the outputs vector
            final_results.push_back(output);
        }
    }

    return final_results;
}

// Utility function to draw the results
void DarknetModel::draw(cv::Mat &frame, std::vector<ObjectDetection::Output> output)
{
    // colors for bounding boxes
    const int num_colors = 4;
    const cv::Scalar colors[4] = {
        {0, 255, 255},
        {255, 255, 0},
        {0, 255, 0},
        {255, 0, 0}
    };

    for (size_t i = 0; i < output.size(); ++i)
    {
        const auto color = colors[output[i].class_id % num_colors];
        const auto& rect = cv::Rect(output[i].bbox_x, output[i].bbox_y, output[i].bbox_width, output[i].bbox_height);
        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

        std::ostringstream label_ss;
        label_ss << output[i].class_name << ": " << std::fixed << std::setprecision(2) << output[i].confidence;
        auto label = label_ss.str();
        
        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
}