#include <handlerimage.hpp>

void ReadCar::initModel(const string& nameModel){
    modelPath = nameModel;
    modelLoaded = loadONNXModel(modelPath);
}

void ReadCar::showSpecificationModel(){
     try {
        // Load the network
        dnn::Net net = dnn::readNetFromONNX(modelPath);

        // --- Get Input/Output Layer Names ---
        vector<string> inputNames = net.getLayerNames(); // This might return all layers, not just inputs
        // A better approach for input name might involve documentation or Netron

        vector<string> outputNames = net.getUnconnectedOutLayersNames();
        cout << "Output Layer Names:" << endl;
        for (const string& name : outputNames) {
            cout << "- " << name << endl;
        }

        // --- Get Layer IDs and Names (all layers) ---
        vector<string> allLayerNames = net.getLayerNames();
        cout << "\nAll Layer Names and IDs:" << endl;
        for (const string& name : allLayerNames) {
            int layerId = net.getLayerId(name);
            // Using getLayer() might give some info on the type, but generally limited within C++
            cout << "- ID: " << layerId << ", Name: " << name << endl;
        }
        
        // You can dump the network configuration to a text file
        net.dumpToFile("network_structure.txt");
        cout << "\nNetwork structure dumped to 'network_structure.txt'" << endl;

    } catch (const Exception& e) {
        cerr << "Error loading the model: " << e.what() << endl;
    }
}

bool ReadCar::loadONNXModel(const string& path) {
    try {
        cout << "Loading ONNX model: " << path << endl;
        
        // Check if file exists
        if (!fs::exists(path)) {
            cerr << "✗ Model file not found: " << path << endl;
            return false;
        }
        
        // Create ONNX Runtime environment
        onnxEnv = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LicensePlateDetection");
        
        // Configure session options
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        #ifdef _WIN32
            wstring wModelPath(path.begin(), path.end());
            onnxSession = make_unique<Ort::Session>(*onnxEnv, wModelPath.c_str(), sessionOptions);
        #else
            onnxSession = make_unique<Ort::Session>(*onnxEnv, path.c_str(), sessionOptions);
        #endif
        
        // Get input shape
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = onnxSession->GetInputNameAllocated(0, allocator);
        auto inputTypeInfo = onnxSession->GetInputTypeInfo(0);
        auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape = tensorInfo.GetShape();
        
        cout << "✓ ONNX Model loaded successfully!" << endl;
        cout << "  Input name: " << inputName.get() << endl;
        cout << "  Input shape: [";
        for (size_t i = 0; i < inputShape.size(); i++) {
            if (inputShape[i] == -1) {
                cout << "dynamic";
            } else {
                cout << inputShape[i];
            }
            if (i < inputShape.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Get output info
        size_t numOutputs = onnxSession->GetOutputCount();
        cout << "  Number of outputs: " << numOutputs << endl;
        
        for (size_t i = 0; i < numOutputs; i++) {
            auto outputName = onnxSession->GetOutputNameAllocated(i, allocator);
            auto outputTypeInfo = onnxSession->GetOutputTypeInfo(i);
            auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            auto outputShape = outputTensorInfo.GetShape();
            
            cout << "  Output " << i << " name: " << outputName.get() << endl;
            cout << "  Output " << i << " shape: [";
            for (size_t j = 0; j < outputShape.size(); j++) {
                if (outputShape[j] == -1) {
                    cout << "dynamic";
                } else {
                    cout << outputShape[j];
                }
                if (j < outputShape.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        cerr << "✗ ONNX Runtime error: " << e.what() << endl;
        return false;
    } catch (const exception& e) {
        cerr << "✗ Error loading model: " << e.what() << endl;
        return false;
    }
}

vector<Detection> ReadCar::detectWithONNX(const Mat& image, const string& currentNameImage) {
    vector<Detection> detections;
    
    if (!modelLoaded || !onnxSession) {
        cerr << "  ✗ Model not loaded!" << endl;
        return detections;
    }
    
    // Validate input image
    if (image.empty()) {
        cerr << "  ✗ Empty image for ONNX inference!" << endl;
        return detections;
    } else {
        cout << "  → ONNX input image size: " << image.cols << "x" << image.rows << endl;
    }
    
    try {
        // Get input dimensions with proper validation
        int inputHeight = 640;  // Default YOLO size
        int inputWidth = 640;
        
        // Check if we have valid shape from model
        if (inputShape.size() >= 4) {
            // inputShape format: [batch, channels, height, width]
            int64_t h = inputShape[2];
            int64_t w = inputShape[3];
            
            // Use model shape only if valid (not -1 or dynamic)
            if (h > 0 && h <= 4096) inputHeight = static_cast<int>(h);
            if (w > 0 && w <= 4096) inputWidth = static_cast<int>(w);
        }
        
        cout << "  → ONNX input size: " << inputWidth << "x" << inputHeight << endl;
        
        // Validate resize dimensions
        if (inputWidth <= 0 || inputHeight <= 0) {
            cerr << "  ✗ Invalid input dimensions!" << endl;
            return detections;
        }
        
        // Preprocess image
        Mat resized, floatImg;
        
        // Safety check before resize
        if (image.cols <= 0 || image.rows <= 0) {
            cerr << "  ✗ Invalid source image size!" << endl;
            return detections;
        }
        
        // Resize dengan interpolasi yang tepat
        if (image.cols > inputWidth || image.rows > inputHeight) {
            resize(image, resized, Size(inputWidth, inputHeight), 0, 0, INTER_AREA);
        } else {
            resize(image, resized, Size(inputWidth, inputHeight), 0, 0, INTER_LINEAR);
        }
        
        // Convert to float [0, 1]
        resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
        
        // Convert BGR to RGB if needed (YOLO expects RGB)
        Mat rgbImg;
        cvtColor(floatImg, rgbImg, COLOR_BGR2RGB);
        
        // Convert HWC to CHW format
        vector<Mat> channels(3);
        split(rgbImg, channels);
        
        // Create input tensor
        vector<float> inputData;
        inputData.reserve(3 * inputHeight * inputWidth);
        
        // CHW format: all R, then all G, then all B
        for (int c = 0; c < 3; c++) {
            inputData.insert(inputData.end(), 
                           (float*)channels[c].data, 
                           (float*)channels[c].data + inputHeight * inputWidth);
        }
        
        cout << "  → Input tensor size: " << inputData.size() << endl;
        
        // Create input tensor object
        vector<int64_t> inputShapeVec = {1, 3, inputHeight, inputWidth};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, 
            inputData.data(), 
            inputData.size(),
            inputShapeVec.data(), 
            inputShapeVec.size()
        );
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamePtr = onnxSession->GetInputNameAllocated(0, allocator);
        auto outputNamePtr = onnxSession->GetOutputNameAllocated(0, allocator);
        
        const char* inputNames[] = {inputNamePtr.get()};
        const char* outputNames[] = {outputNamePtr.get()};
        
        cout << "  → Running inference..." << endl;
        
        // Run inference
        auto outputTensors = onnxSession->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1
        );
        
        cout << "  → Inference complete, post-processing..." << endl;
        
        // Post-process outputs
        detections = postprocessYOLO(currentNameImage, outputTensors, image , image.size());
        drawDetections(image, detections);

        cout << "  → Found " << detections.size() << " detection(s)" << endl;
        
    } catch (const Exception& e) {
        cerr << "  ✗ OpenCV Error: " << e.what() << endl;
    } catch (const Ort::Exception& e) {
        cerr << "  ✗ ONNX Runtime Error: " << e.what() << endl;
    } catch (const exception& e) {
        cerr << "  ✗ Standard Error: " << e.what() << endl;
    }
    
    return detections;
}

vector<Detection> ReadCar::postprocessYOLO(
    const string& currentNameImage, 
    const vector<Ort::Value>& outputs, 
    const Mat& originalImage,
    const Size& imgSize, 
    float confThreshold, 
    float iouThreshold
){
    vector<Detection> detections;
    vector<int> size;
    
    if (outputs.empty()) {
        cerr << "  ✗ Empty output from model!" << endl;
        return detections;
    }
    
    try {
        // Get output tensor
        const float* rawOutput = outputs[0].GetTensorData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        cout << "  → Output shape: [";
        for (size_t i = 0; i < shape.size(); i++) {
            cout << shape[i];
            if (i < shape.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // YOLOv11 output format bisa:
        // Option 1: [batch, num_predictions, 84] -> [cx, cy, w, h, class_scores...]
        // Option 2: [batch, 84, num_predictions] -> transposed
        
        int batchSize = shape.size() > 0 ? shape[0] : 1;
        int dim1 = shape.size() > 1 ? shape[1] : 0;
        int dim2 = shape.size() > 2 ? shape[2] : 0;
        
        int numPredictions, numData;
        bool transposed = false;
        
        // Detect format
        if (dim1 > dim2 && dim2 >= 4) {
            // Format: [batch, num_predictions, data]
            numPredictions = dim1;
            numData = dim2;
            transposed = false;
        } else if (dim2 > dim1 && dim1 >= 4) {
            // Format: [batch, data, num_predictions] - need transpose
            numPredictions = dim2;
            numData = dim1;
            transposed = true;
        } else {
            cerr << "  ✗ Unexpected output format!" << endl;
            return detections;
        }
        
        cout << "  → Predictions: " << numPredictions << ", Data per prediction: " << numData << endl;
        cout << "  → Format: " << (transposed ? "transposed" : "normal") << endl;
        
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;
        
        // Scale factors
        float scaleX = static_cast<float>(imgSize.width) / 640.0f;
        float scaleY = static_cast<float>(imgSize.height) / 640.0f;
        
        // Parse predictions
        for (int i = 0; i < numPredictions; i++) {
            const float* prediction;
            
            if (transposed) {
                // Access data for transposed format
                prediction = new float[numData];
                for (int j = 0; j < numData; j++) {
                    ((float*)prediction)[j] = rawOutput[j * numPredictions + i];
                }
            } else {
                prediction = rawOutput + i * numData;
            }
            
            // YOLOv11 format: [cx, cy, w, h, class_scores...]
            // No explicit objectness score, use max class score
            float maxClassScore = 0.0f;
            int maxClassId = 0;
            
            // Find max class score (starting from index 4)
            for (int j = 4; j < numData; j++) {
                if (prediction[j] > maxClassScore) {
                    maxClassScore = prediction[j];
                    maxClassId = j - 4;
                }
            }
            
            if (maxClassScore > confThreshold) {
                // Get box coordinates (normalized to 640x640)
                float cx = prediction[0];
                float cy = prediction[1];
                float w = prediction[2];
                float h = prediction[3];
                
                // Convert to pixel coordinates
                int x = static_cast<int>((cx - w / 2) * scaleX);
                int y = static_cast<int>((cy - h / 2) * scaleY);
                int width = static_cast<int>(w * scaleX);
                int height = static_cast<int>(h * scaleY);
                
                // Clamp to image boundaries
                x = max(0, min(x, imgSize.width - 1));
                y = max(0, min(y, imgSize.height - 1));
                width = min(width, imgSize.width - x);
                height = min(height, imgSize.height - y);
                
                if (width > 0 && height > 0) {
                    size.push_back(width);
                    size.push_back(height);
                    boxes.push_back(Rect(x, y, width, height));
                    confidences.push_back(maxClassScore);
                    classIds.push_back(maxClassId);
                }
            }
            
            if (transposed) {
                delete[] prediction;
            }
        }
        
        cout << "  → Found " << boxes.size() << " raw detections before NMS" << endl;
        
        // Apply Non-Maximum Suppression
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

        // Create final detections
        for (int idx : indices) {
            Detection det;
            det.box = boxes[idx];
            det.confidence = confidences[idx];
            det.classId = classIds[idx];
            detections.push_back(det);
        }

        cout << "  → " << indices.size() << " detections after NMS" << endl;
        
        // ========== NEW: Crop last detection and draw rectangle ==========
        if (!detections.empty()) {
            const Detection& lastDet = detections.back();

            // Draw rectangle on original image
            Mat rectImage = originalImage.clone();
            rectangle(rectImage, lastDet.box, Scalar(0, 255, 0), 2);

            // Crop region safely
            Rect safeRect = lastDet.box & Rect(0, 0, originalImage.cols, originalImage.rows);
            Mat cropped = originalImage(safeRect).clone();

            // Save the cropped region and annotated image
            string croppedFile = "rectangled_last_" + currentNameImage + ".jpg";
            string visualizedFile = "rectmarked_" + currentNameImage + ".jpg";

            imwrite(croppedFile, cropped);
            imwrite(visualizedFile, rectImage);

            cout << "  ✓ Saved cropped region: " << croppedFile << endl;
            cout << "  ✓ Saved visualization:  " << visualizedFile << endl;
        }       
    } catch (const exception& e) {
        cerr << "  ✗ Post-processing error: " << e.what() << endl;
    }
    
    return detections;
}

void ReadCar::drawDetections(const Mat& image, const vector<Detection>& detections) {
    cout << "Draw detections..." << endl;
    for (const auto& det : detections) {
        rectangle(image, det.box, Scalar(0, 255, 0), 2);
        
        string label = "Plate: " + to_string(int(det.confidence * 100)) + "%";
        cout << label << endl;
        int baseline;
        Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        rectangle(image, 
                     Point(det.box.x, det.box.y - textSize.height - 5),
                     Point(det.box.x + textSize.width, det.box.y),
                     Scalar(0, 255, 0), FILLED);
        
        putText(image, label, Point(det.box.x, det.box.y - 5),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

        imshow("output/Detections.png", image);
    }
}
