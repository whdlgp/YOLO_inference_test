#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <fstream>
#include <filesystem>
#include <any>

#include "module/dnn_model_darknet.hpp"
#include "module/dnn_model_yolov5.hpp"
#include "module/dnn_model_yolov6.hpp"
#include "module/dnn_model_yolov8.hpp"

namespace fs = std::filesystem;


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
    model.init(ObjectDetection::InitParams(std::make_tuple(config_file.string(), weights_file.string(), names_file.string())), 0.5f, 0.4f, 416, 416);

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
    cv::namedWindow("output");
    cv::imshow("output", img);
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
    model.init(ObjectDetection::InitParams(std::make_tuple(onnx_file.string(), names_file.string())), 0.5f, 0.4f, 640, 640);

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
    cv::namedWindow("output");
    cv::imshow("output", img);
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
    model.init(ObjectDetection::InitParams(std::make_tuple(onnx_file.string(), names_file.string())), 0.5f, 0.4f, 640, 640);

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
    cv::namedWindow("output");
    cv::imshow("output", img);
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
    model.init(ObjectDetection::InitParams(std::make_tuple(config_file.string(), weights_file.string(), names_file.string())), 0.5f, 0.4f, 416, 416);

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
    cv::namedWindow("output");
    cv::imshow("output", img);
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
    model.init(ObjectDetection::InitParams(std::make_tuple(onnx_file.string(), names_file.string())), 0.5f, 0.4f, 640, 640);

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
    cv::namedWindow("output");
    cv::imshow("output", img);
    cv::waitKey(0);
}


int main()
{
    //test_yolov4_darknet();
    //test_yolov5_onnx();
    //test_yolov6_onnx();
    //test_yolov7_darknet();
    test_yolov8_onnx();

    return 0;
}