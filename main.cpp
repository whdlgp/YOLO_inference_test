#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "extra/json.hpp"

#include <vector>
#include <fstream>
#include <filesystem>
#include <any>

#include "module/dnn_model_darknet.hpp"
#include "module/dnn_model_yolov5.hpp"
#include "module/dnn_model_yolov6.hpp"
#include "module/dnn_model_yolov8.hpp"
#include "module/dnn_model_yolov9.hpp"
#include "module/dnn_model_yolov10.hpp"


namespace fs = std::filesystem;
using json = nlohmann::json;


void test_yolov4_darknet()
{
    // Model and Image directories
    fs::path models_dir = "models";
    fs::path images_dir = "images";

    // Model files
    fs::path config_file = models_dir / "yolov4" / "yolov4.cfg";
    fs::path weights_file = models_dir / "yolov4" / "yolov4.weights";
    fs::path names_file = models_dir / "yolov4" / "coco.names";

    // Image file
    fs::path image_file = images_dir / "dog.jpg";

    // Initialize model
    DarknetModel model;
    json init_param;
    init_param["config_path"] = config_file.string();
    init_param["weights_path"] = weights_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 416;
    init_param["inference_height"] = 416;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov4");
    cv::imshow("yolov4", img);
    cv::waitKey(0);
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

    // Initialize model
    YOLOv5Model model;
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov5");
    cv::imshow("yolov5", img);
    cv::waitKey(0);
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

    // Initialize model
    YOLOv6Model model;
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov6");
    cv::imshow("yolov6", img);
    cv::waitKey(0);
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

    // Initialize model
    DarknetModel model;
    json init_param;
    init_param["config_path"] = config_file.string();
    init_param["weights_path"] = weights_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 416;
    init_param["inference_height"] = 416;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov7");
    cv::imshow("yolov7", img);
    cv::waitKey(0);
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

    // Initialize model
    YOLOv8Model model;
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov8");
    cv::imshow("yolov8", img);
    cv::waitKey(0);
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

    // Initialize model
    YOLOv9Model model;
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov9");
    cv::imshow("yolov9", img);
    cv::waitKey(0);
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

    // Initialize model
    YOLOv10Model model;
    json init_param;
    init_param["onnx_path"] = onnx_file.string();
    init_param["names_file"] = names_file.string();
    init_param["confidence_threshold"] = 0.5f;
    init_param["nms_threshold"] = 0.4f;
    init_param["inference_width"] = 640;
    init_param["inference_height"] = 640;
    model.init(init_param);

    // Load the image
    cv::Mat img = cv::imread(image_file.string());
    ObjectDetection::Input input;
    input.chan = img.channels();
    input.height = img.rows;
    input.width = img.cols;
    input.data.resize(input.chan*input.height*input.width);
    std::memcpy(input.data.data(), img.data, input.data.size());

    auto outputs = model.infer(input);

    // Print the outputs
    for (const auto& output : outputs)
    {
        std::cout << "Bounding Box: (" << output.bbox_x << ", " << output.bbox_y << "), "
                  << "Width: " << output.bbox_width << ", Height: " << output.bbox_height << ", "
                  << "Class Name: " << output.class_name << ", "
                  << "Confidence: " << output.confidence << std::endl;
    }

    DarknetModel::draw(img, outputs);
    
    // Display the output
    cv::namedWindow("yolov10");
    cv::imshow("yolov10", img);
    cv::waitKey(0);
}

int main()
{
    test_yolov4_darknet();
    test_yolov5_onnx();
    test_yolov6_onnx();
    test_yolov7_darknet();
    test_yolov8_onnx();
    test_yolov9_onnx();
    test_yolov10_onnx();

    return 0;
}