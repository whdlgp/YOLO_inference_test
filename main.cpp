#include <vector>
#include <fstream>
#include <filesystem>
#include <any>

#include <opencv2/opencv.hpp>

#include "module/make_models.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// Draw Detection with OpenCV
void draw(const std::vector<DetectedOutput>& detections, cv::Mat img, std::string window_name)
{
    // colors for bounding boxes
    const int num_colors = 4;
    const cv::Scalar colors[4] = {
        {0, 255, 255},
        {255, 255, 0},
        {0, 255, 0},
        {255, 0, 0}
    };

    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto color = colors[detections[i].class_id % num_colors];
        const auto& rect = cv::Rect(detections[i].bbox_x, detections[i].bbox_y, detections[i].bbox_width, detections[i].bbox_height);
        cv::rectangle(img, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

        std::ostringstream label_ss;
        label_ss << detections[i].class_name << ": " << std::fixed << std::setprecision(2) << detections[i].confidence;
        auto label = label_ss.str();
        
        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(img, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(img, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    
    // Display the output
    cv::namedWindow(window_name);
    cv::imshow(window_name, img);
    cv::waitKey(0);
}

void test_onnxruntime()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov8" / "yolov8l.onnx";
    fs::path names_file = models_dir / "yolov8" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_onnxruntime();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "ONNXRuntime");
}

void test_yolov4_darknet()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    fs::path config_file = models_dir / "yolov4" / "yolov4.cfg";
    fs::path weights_file = models_dir / "yolov4" / "yolov4.weights";
    fs::path names_file = models_dir / "yolov4" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["config_path"] = config_file.string();
    init_param["weights_path"] = weights_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_darknet();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v4");
}

void test_yolov5_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov5" / "yolov5l.onnx";
    fs::path names_file = models_dir / "yolov5" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov5();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v5");
}

void test_yolov6_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov6" / "yolov6l.onnx";
    fs::path names_file = models_dir / "yolov6" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov6();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v6");
}

void test_yolov7_darknet()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path config_file = models_dir / "yolov7" / "yolov7.cfg";
    fs::path weights_file = models_dir / "yolov7" / "yolov7.weights";
    fs::path names_file = models_dir / "yolov7" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["config_path"] = config_file.string();
    init_param["weights_path"] = weights_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_darknet();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v7");
}

void test_yolov8_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov8" / "yolov8l.onnx";
    fs::path names_file = models_dir / "yolov8" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov8();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v8");
}

void test_yolov9_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov9" / "yolov9-c-converted.onnx";
    fs::path names_file = models_dir / "yolov9" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov9();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v9");
}

void test_yolov10_onnx()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path onnx_file = models_dir / "yolov10" / "yolov10s.onnx";
    fs::path names_file = models_dir / "yolov10" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Load the image
    cv::Mat img = cv::imread(image_file.string());

    // Convert to Image class
    Image input;
    input.width = img.cols;
    input.height = img.rows;
    input.chan = img.channels();
    input.data.resize(input.total());
    std::memcpy(input.data.data(), img.data, input.data.size());

    // Initialize model
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;

    auto model = make_yolov10();
    model->init(init_param);
    
    // Inference and Post Process
    auto detections = model->run(input);

    // Print the outputs
    for (const auto& output : detections)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    // Draw and Show
    draw(detections, img, "YOLO v10");
}

int main()
{
    /*
    test_yolov4_darknet();
    test_yolov5_onnx();
    test_yolov6_onnx();
    test_yolov7_darknet();
    test_yolov8_onnx();
    test_yolov9_onnx();
    test_yolov10_onnx();
    */
    test_onnxruntime();
    return 0;
}