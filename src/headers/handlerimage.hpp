#ifndef HANDLERIMAGE_HPP
#define HANDLERIMAGE_HPP

#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <tesseract/baseapi.h>
#include <onnxruntime_cxx_api.h> 
#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <regex>
#include <algorithm>
#include <fstream>
#include <opencv2/dnn.hpp>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace cv::dnn;

struct PlateResult {
    string filename;
    string plateText;
    string vehicleName;
    string vehicleBrand;
    float confidence;
    cv::Rect boundingBox;
    bool detected;
    string plateType;
};

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class ReadCar {
private:
    vector<string> listImage;
    vector<PlateResult> results;
    tesseract::TessBaseAPI* ocr;

    unique_ptr<Ort::Env> onnxEnv;
    unique_ptr<Ort::Session> onnxSession;
    Ort::SessionOptions sessionOptions;
    vector<int64_t> inputShape;
    string modelPath = "models/license-plate-finetune-v1m.onnx";
    bool modelLoaded;
    
    // Helper functions
    cv::Mat preprocessImage(const cv::Mat& image, const string& nameImage);
    cv::Mat preprocessPlateForOCR(const cv::Mat& plateROI);
    vector<cv::Rect> detectPlateCandidates(const cv::Mat& image, string nameImage);
    cv::Rect filterBestPlate(const vector<cv::Rect>& candidates, const cv::Mat& image);
    bool validatePlateRegion(const cv::Rect& region, const cv::Mat& image);

    // Black plate function
    bool isBlackPlate(const cv::Rect& region, const cv::Mat& image);
    cv::Mat preprocessForBlackPlate(const cv::Mat& image, string nameImage);
    vector<cv::Rect> detectBlackPlateCandidates(const cv::Mat& image, string nameImage);
    
    // ONNX Model functions
    bool loadONNXModel(const string& path);
    vector<Detection> detectWithONNX(const cv::Mat& image, const string& currentNameImage);
    vector<Detection> postprocessYOLO(const string& currentNameImage, const vector<Ort::Value>& outputs, const Mat& originalImage, const cv::Size& imgSize, float confThreshold = 0.25, float iouThreshold = 0.45);
    void drawDetections(const cv::Mat& image, const vector<Detection>& detections);

    // OCR functions
    string performOCR(const cv::Mat& plateROI, float& confidence);
    string cleanPlateText(const string& rawText);
    bool validateIndonesianPlate(const string& text);
    Mat repairCharacters(const cv::Mat& binaryInput, const std::string& currentNameImage);

    
public:
    ReadCar();
    ~ReadCar();
    void initModel(const string& nameModel);

    void readImage(string path);
    void handleImage();
    void analyzeImage(string path);
    bool isRectangle(const string& path);
    void detectAndSavePlate(const string& path);
    PlateResult detectPlateWithOCR(const string& path);

    // New OCR methods
    void exportResultsToCSV(const string& outputPath = "output/results.csv");
    void printResults();

    void showSpecificationModel();

};

#endif