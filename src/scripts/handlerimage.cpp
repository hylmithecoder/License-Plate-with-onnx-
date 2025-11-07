#include <handlerimage.hpp>

ReadCar::ReadCar() {
    // Initialize Tesseract
    ocr = new tesseract::TessBaseAPI();
    
    // Init dengan bahasa Inggris (untuk karakter alfanumerik)
    if (ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
        cerr << "Could not initialize tesseract!" << endl;
        exit(1);
    }
    
    // Set Page Segmentation Mode
    // PSM_7 = Treat image as single text line (cocok untuk plat nomor)
    ocr->SetPageSegMode(tesseract::PSM_RAW_LINE);
    
    // Whitelist karakter (hanya huruf kapital dan angka)
    ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    
    cout << "✓ Tesseract OCR initialized successfully" << endl;
}

ReadCar::~ReadCar() {
    if (ocr) {
        ocr->End();
        delete ocr;
    }
}

void ReadCar::readImage(string path){
    listImage.clear(); // Clear previous list
    
    if (!fs::exists(path)) {
        cerr << "Error: Directory '" << path << "' does not exist!" << endl;
        return;
    }
    
    for (const auto& entry : fs::directory_iterator(path)){
        string ext = entry.path().extension().string();
        // Filter hanya file gambar
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp"){
            listImage.push_back(entry.path().string());
            cout << "Loaded: " << entry.path().filename() << endl;
        }
    }
    
    cout << "Total images loaded: " << listImage.size() << endl;
}

void ReadCar::handleImage(){
    readImage("images");
    
    if (listImage.empty()) {
        cout << "No images found in 'images' directory!" << endl;
        return;
    }
    
    if (!fs::exists("output")) {
        fs::create_directory("output");
    }
    
    int count = 0;
    for (const string& nameImage : listImage){
        cout << "\n=== Processing [" << ++count << "/" << listImage.size() << "]: " 
             << fs::path(nameImage).filename() << " ===" << endl;
        
        PlateResult result = detectPlateWithOCR(nameImage);
        results.push_back(result);
    }
    
    cout << "\n" << string(60, '=') << endl;
    printResults();
    exportResultsToCSV();
    cout << "\n✓ All images processed! Check 'output' folder." << endl;
}

#pragma region Handle white plate
Mat ReadCar::preprocessImage(const Mat& image, const std::string& nameImage) {
    Mat gray, blurred, thresh, processed;

    // 1. Convert to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imwrite("output/step1_gray_" + nameImage, gray);

    // 2. Apply bilateral filter (mengurangi noise tapi tetap jaga edge)
    bilateralFilter(gray, blurred, 11, 17, 17);
    imwrite("output/step2_blurred_" + nameImage, blurred);

    // 3. Apply adaptive threshold
    adaptiveThreshold(blurred, thresh, 255,
                          ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY_INV, 21, 10);
    imwrite("output/step3_thresh_" + nameImage, thresh);

    // 4. Morphological operations untuk cleanup
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresh, processed, MORPH_CLOSE, kernel);
    imwrite("output/step4_morph_" + nameImage, processed);

    return processed;
}

#pragma endregion

string ReadCar::cleanPlateText(const string& rawText) {
    string cleaned = rawText;
    
    // Remove whitespace dan newlines
    cleaned.erase(remove_if(cleaned.begin(), cleaned.end(), 
                  [](char c) { return isspace(c); }), cleaned.end());
    
    // Convert to uppercase
    transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::toupper);
    
    // Common OCR mistakes correction
    replace(cleaned.begin(), cleaned.end(), 'O', '0'); // O -> 0
    replace(cleaned.begin(), cleaned.end(), 'I', '1'); // I -> 1
    // replace(cleaned.begin(), cleaned.end(), 'S', '5'); // S -> 5 (kadang)
    // replace(cleaned.begin(), cleaned.end(), 'Z', '2'); // Z -> 2 (kadang)
    
    // Remove special characters
    cleaned.erase(remove_if(cleaned.begin(), cleaned.end(),
                  [](char c) { return !isalnum(c); }), cleaned.end());
    
    return cleaned;
}

bool ReadCar::validateIndonesianPlate(const string& text) {
    if (text.length() < 5 || text.length() > 11) return false;
    
    // Format plat Indonesia: 
    // - 1-2 huruf (kode wilayah, misal: B, DK)
    // - 1-4 angka
    // - 1-3 huruf (seri, misal: ABC)
    // Contoh: B1234XYZ, DK5678AB
    
    regex platePattern("^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$");
    return regex_match(text, platePattern);
}

string ReadCar::performOCR(const Mat& plateROI, float& confidence) {
    if (plateROI.empty()) {
        confidence = 0.0f;
        return "UNRECOGNIZED";
    }

    // Generate unique filename suffix using timestamp
    auto now = chrono::system_clock::now();
    auto timestamp = chrono::duration_cast<chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    string debug_suffix = to_string(timestamp);

    // Save original ROI
    imwrite("output/debug_original_" + debug_suffix + ".jpg", plateROI);
    
    // Preprocess plate untuk OCR
    Mat processed = preprocessPlateForOCR(plateROI);
    imwrite("output/debug_processed_" + debug_suffix + ".jpg", processed);
    
    // Method 1: Default preprocessing
    ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr->SetImage(processed.data, processed.cols, processed.rows, 
                 processed.channels(), processed.step);
    
    char* outText = ocr->GetUTF8Text();
    string rawText(outText);
    delete[] outText;
    
    confidence = ocr->MeanTextConf();
    string cleanedText = cleanPlateText(rawText);
    
    cout << "\n=== Method 1 (Default) ===" << endl;
    cout << "Raw text: " << rawText << endl;
    cout << "Cleaned: " << cleanedText << endl;
    cout << "Confidence: " << confidence << "%" << endl;
    
    // Try alternative methods if confidence is low
    if (confidence < 70.0f) {
        cout << "\n=== Trying Alternative Methods ===" << endl;
        
        // Method 2: Otsu threshold
        Mat binary2;
        threshold(processed, binary2, 0, 255, THRESH_BINARY | THRESH_OTSU);
        imwrite("output/debug_otsu_" + debug_suffix + ".jpg", binary2);
        
        ocr->SetImage(binary2.data, binary2.cols, binary2.rows, 1, binary2.step);
        outText = ocr->GetUTF8Text();
        string rawText2(outText);
        delete[] outText;
        
        float conf2 = ocr->MeanTextConf();
        string cleaned2 = cleanPlateText(rawText2);
        
        cout << "\n=== Method 2 (Otsu) ===" << endl;
        cout << "Raw text: " << rawText2 << endl;
        cout << "Cleaned: " << cleaned2 << endl;
        cout << "Confidence: " << conf2 << "%" << endl;
        
        if (conf2 > confidence) {
            confidence = conf2;
            cleanedText = cleaned2;
            cout << "✓ Method 2 selected (better confidence)" << endl;
        }
        
        // Method 3: Inverted threshold if still low confidence
        if (confidence < 65.0f) {
            Mat inverted;
            bitwise_not(processed, inverted);
            threshold(inverted, inverted, 0, 255, THRESH_BINARY | THRESH_OTSU);
            imwrite("output/debug_inverted_" + debug_suffix + ".jpg", inverted);
            
            ocr->SetImage(inverted.data, inverted.cols, inverted.rows, 1, inverted.step);
            outText = ocr->GetUTF8Text();
            string rawText3(outText);
            delete[] outText;
            
            float conf3 = ocr->MeanTextConf();
            string cleaned3 = cleanPlateText(rawText3);
            
            cout << "\n=== Method 3 (Inverted) ===" << endl;
            cout << "Raw text: " << rawText3 << endl;
            cout << "Cleaned: " << cleaned3 << endl;
            cout << "Confidence: " << conf3 << "%" << endl;
            
            if (conf3 > confidence) {
                confidence = conf3;
                cleanedText = cleaned3;
                cout << "✓ Method 3 selected (best confidence)" << endl;
            }
        }
    }
    
    cout << "\n=== Final Result ===" << endl;
    cout << "Text: \"" << cleanedText << "\"" << endl;
    cout << "Confidence: " << confidence << "%" << endl;
    cout << "Debug images saved with suffix: " << debug_suffix << endl;
    
    return cleanedText;
}

bool ReadCar::validatePlateRegion(const Rect& region, const Mat& image) {
    if (region.width <= 0 || region.height <= 0) return false;
    if (region.x < 0 || region.y < 0) return false;
    if (region.x + region.width > image.cols) return false;
    if (region.y + region.height > image.rows) return false;
    
    Mat plateROI = image(region);
    Mat grayPlate;
    cvtColor(plateROI, grayPlate, COLOR_BGR2GRAY);
    
    // Hitung edge density (plat nomor harusnya punya banyak edge karena ada karakter)
    Mat edges;
    Canny(grayPlate, edges, 100, 200);
    int edgeCount = countNonZero(edges);
    double edgeDensity = (double)edgeCount / (region.width * region.height);
    
    // Plat nomor biasanya punya edge density 0.1 - 0.4
    return (edgeDensity > 0.08 && edgeDensity < 0.5);
}

PlateResult ReadCar::detectPlateWithOCR(const string& path) {
    string plateText;

    cout << "\n=== Processing " << fs::path(path).filename() << " ===" << endl;
    if (!fs::exists("output")) {
        fs::create_directory("output");
    }

    PlateResult result;
    result.filename = fs::path(path).filename().string();
    result.vehicleName = "Tidak Ter Identifikasi";
    result.detected = false;
    result.confidence = 0.0f;
    
    Mat image = imread(path, IMREAD_COLOR);
    
    if (image.empty()) {
        cerr << "  ✗ Error: Could not load image!" << endl;
        return result;
    }
    
    // Resize jika terlalu besar
    Mat resized = image.clone();
    if (image.cols > 1280) {
        double scale = 1280.0 / image.cols;
        resize(image, resized, Size(), scale, scale);
    }

    // Deteksi mobil dengan save debug image
    float carProbability = detectCar(image, true);
    
    if (carProbability > 50.0f) {
        cout << "HIGH CONFIDENCE: Vehicle detected with " 
             << carProbability << "% probability" << endl;
        
        // Lanjut ke deteksi plat nomor
        // ... your existing plate detection code ...
    } else if (carProbability > 0.0f) {
        cout << "LOW CONFIDENCE: Possible vehicle with " 
             << carProbability << "% probability" << endl;
    } else {
        cout << "NO VEHICLE DETECTED in the image" << endl;
    }

    // Detect candidates
    vector<Rect> candidates = detectPlateCandidates(resized, result.filename);
    cout << "  → Found " << candidates.size() << " plate candidate(s)" << endl;
    
    if (candidates.empty()) {
        cout << "  ✗ No plate detected" << endl;
        
        string filename = fs::path(path).stem().string();
        string outputPath = "output/no_plate_" + filename + ".jpg";
        imwrite(outputPath, image);
        return result;
    }
    
    // Get best plate
    Rect bestPlate = filterBestPlate(candidates, resized);
    
    if (bestPlate.width == 0) {
        cout << "  ✗ No valid plate found" << endl;
        plateText = "UNRECOGNIZED";
        string filename = fs::path(path).stem().string();
        string outputPath = "output/no_plate_" + filename + ".jpg";
        imwrite(outputPath, image);
        return result;
    }
    
    // Scale back
    if (image.cols != resized.cols) {
        double scale = (double)image.cols / resized.cols;
        bestPlate.x *= scale;
        bestPlate.y *= scale;
        bestPlate.width *= scale;
        bestPlate.height *= scale;
    }
    
    // Crop plate
    Mat plateROI = image(bestPlate);
    
    // Perform OCR
    float confidence;
    if (performOCR(plateROI, confidence).empty()) {
        cout << "  ✗ OCR failed to read plate" << endl;
        plateText = "UNRECOGNIZED";
    } else {
        plateText = performOCR(plateROI, confidence);
    }

    bool isBlack = isBlackPlate(bestPlate, image);
    string plateType = isBlack ? "BLACK" : "WHITE";

    // Update result
    result.detected = true;
    result.plateText = plateText;
    result.confidence = confidence;
    result.boundingBox = bestPlate;
    
    // Validate format
    bool isValid = validateIndonesianPlate(plateText);
    
    // Draw result
    Mat output = image.clone();
    Scalar color = isValid ? Scalar(0, 255, 0) : Scalar(0, 165, 255); // Green or Orange
    rectangle(output, bestPlate, color, 3);
    
    // Add text dengan background
    string displayText = plateText.empty() ? "UNREADABLE" : plateText;
    string confText = " (" + to_string((int)confidence) + "%)";
    displayText += confText;
    
    int baseline = 0;
    double fontScale = 0.8;
    int thickness = 2;
    Size textSize = getTextSize(displayText, FONT_HERSHEY_SIMPLEX, 
                                         fontScale, thickness, &baseline);
    
    Point textOrg(bestPlate.x, bestPlate.y - 15);
    
    // Background
    rectangle(output,
                  textOrg + Point(0, baseline + 5),
                  textOrg + Point(textSize.width, -textSize.height - 5),
                  color, FILLED);
    
    putText(output, displayText, textOrg, FONT_HERSHEY_SIMPLEX, 
                fontScale, Scalar(0, 0, 0), thickness);
    
    // Save hasil
    string filename = fs::path(path).stem().string();
    string outputFull = "output/detected_" + filename + ".jpg";
    string outputCrop = "output/plate_" + filename + ".jpg";
    string outputProcessed = "output/processed_" + filename + ".jpg";
    
    imwrite(outputFull, output);
    imwrite(outputCrop, plateROI);
    
    // Save preprocessed version
    Mat processedPlate = preprocessPlateForOCR(plateROI);
    imwrite(outputProcessed, processedPlate);
    
    cout << "  ✓ Plate detected!" << endl;
    cout << "    Type: " << plateType << " PLATE" << endl;
    cout << "    Text: " << (plateText.empty() ? "UNREADABLE" : plateText) << endl;
    cout << "    Confidence: " << fixed << setprecision(1) << confidence << "%" << endl;
    cout << "    Valid Format: " << (isValid ? "YES" : "NO") << endl;
    cout << "    Saved:" << endl;
    cout << "      - " << outputFull << endl;
    cout << "      - " << outputCrop << endl;
    cout << "      - " << outputProcessed << endl;
    
    return result;
}

void ReadCar::printResults() {
    cout << "\n" << string(80, '=') << endl;
    cout << "                           DETECTION SUMMARY" << endl;
    cout << string(80, '=') << endl;
    
    int detected = 0, readable = 0, valid = 0;
    
    for (const auto& r : results) {
        if (r.detected) detected++;
        if (!r.plateText.empty()) readable++;
        if (validateIndonesianPlate(r.plateText)) valid++;
    }
    
    cout << "Total Images:     " << results.size() << endl;
    cout << "Plates Detected:  " << detected << " (" 
         << fixed << setprecision(1) << (100.0 * detected / results.size()) << "%)" << endl;
    cout << "Readable:         " << readable << " (" 
         << (detected > 0 ? 100.0 * readable / detected : 0.0) << "%)" << endl;
    cout << "Valid Format:     " << valid << " (" 
         << (readable > 0 ? 100.0 * valid / readable : 0.0) << "%)" << endl;
    
    cout << "\n" << string(80, '-') << endl;
    cout << left << setw(30) << "Filename" 
         << setw(20) << "Plate Number" 
         << setw(15) << "Confidence"
         << setw(15) << "Valid" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& r : results) {
        string plateDisplay = r.plateText.empty() ? "UNREADABLE" : r.plateText;
        string confDisplay = r.detected ? to_string((int)r.confidence) + "%" : "N/A";
        string validDisplay = validateIndonesianPlate(r.plateText) ? "YES" : "NO";
        
        cout << left << setw(30) << r.filename.substr(0, 28)
             << setw(20) << plateDisplay
             << setw(15) << confDisplay
             << setw(15) << (r.plateText.empty() ? "-" : validDisplay) << endl;
    }
    
    cout << string(80, '=') << endl;
}

void ReadCar::exportResultsToCSV(const string& outputPath) {
    ofstream csvFile(outputPath);
    
    if (!csvFile.is_open()) {
        cerr << "Error: Could not create CSV file!" << endl;
        return;
    }
    
    // Header
    csvFile << "Filename,Plate Number,Confidence (%),Valid Format,Bounding Box (x y w h)" << endl;
    
    // Data
    for (const auto& r : results) {
        csvFile << r.filename << ",";
        csvFile << (r.plateText.empty() ? "UNREADABLE" : r.plateText) << ",";
        csvFile << fixed << setprecision(1) << r.confidence << ",";
        csvFile << (validateIndonesianPlate(r.plateText) ? "YES" : "NO") << ",";
        csvFile << r.boundingBox.x << " " << r.boundingBox.y << " " 
                << r.boundingBox.width << " " << r.boundingBox.height << endl;
    }
    
    csvFile.close();
    cout << "\n✓ Results exported to: " << outputPath << endl;
}

void ReadCar::detectAndSavePlate(const string& path) {
    detectPlateWithOCR(path);
}

void ReadCar::analyzeImage(string path){
    Mat image = imread(path, IMREAD_COLOR);

    if (image.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return;
    }

    Mat gray_image, blurred_image, edges;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    GaussianBlur(gray_image, blurred_image, Size(5, 5), 0);
    Canny(blurred_image, edges, 100, 200);

    imshow("Grayscale Image", gray_image);
    imshow("Canny Edges", edges);
    waitKey(0);
}

bool ReadCar::isRectangle(const string& path) {
    // Legacy function - sekarang gunakan detectAndSavePlate()
    detectAndSavePlate(path);
    return true;
}

#pragma region Handle Black plate
bool ReadCar::isBlackPlate(const Rect& region, const Mat& image) {
    // Safety check
    if (region.x < 0 || region.y < 0 || 
        region.x + region.width > image.cols || 
        region.y + region.height > image.rows) {
        return false;
    }
    
    Mat plateROI = image(region);
    Mat grayPlate;
    
    if (plateROI.channels() == 3) {
        cvtColor(plateROI, grayPlate, COLOR_BGR2GRAY);
    } else {
        grayPlate = plateROI.clone();
    }
    
    // Hitung mean brightness
    Scalar meanBrightness = mean(grayPlate);
    double avgBrightness = meanBrightness[0];
    
    // Plat hitam biasanya punya average brightness < 80
    // Plat putih biasanya > 120
    bool isDark = avgBrightness < 100;
    
    if (isDark) {
        // Double check: plat hitam harus punya text putih/terang
        // Cek histogram untuk memastikan ada pixel terang
        Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        calcHist(&grayPlate, 1, 0, Mat(), hist, 1, &histSize, &histRange);
        
        // Hitung jumlah pixel terang (> 180)
        int brightPixels = 0;
        for (int i = 180; i < 256; i++) {
            brightPixels += hist.at<float>(i);
        }
        
        double brightRatio = (double)brightPixels / (grayPlate.rows * grayPlate.cols);
        
        // Plat hitam harus punya setidaknya 5-30% pixel terang (untuk text)
        return (brightRatio > 0.05 && brightRatio < 0.4);
    }
    
    return false;
}

Mat ReadCar::preprocessForBlackPlate(const Mat& image, string nameImage) {
    Mat gray, blurred, processed;
    
    // Convert to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imwrite("output/step1_gray_black_" + nameImage, gray);
    
    // Bilateral filter
    bilateralFilter(gray, blurred, 11, 17, 17);    
    imwrite("output/step2_blurred_black_" + nameImage, blurred);

    // Untuk plat HITAM: gunakan THRESH_BINARY (bukan INV)
    // Karena kita cari text PUTIH di background HITAM
    adaptiveThreshold(blurred, processed, 255,
                          ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY, 21, 10); // BINARY, bukan BINARY_INV    
    // imwrite("output/step3_thresh_black_" + nameImage, blurred);

    // Morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(processed, processed, MORPH_CLOSE, kernel);
    imwrite("output/step3_morph_black_" + nameImage, processed);
    
    return processed;
}

vector<Rect> ReadCar::detectBlackPlateCandidates(const Mat& image, string nameImage) {
    Mat processed = preprocessForBlackPlate(image, nameImage);
    
    vector<vector<Point>> contours;
    findContours(processed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    vector<Rect> candidates;
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < 500) continue;
        
        double perimeter = arcLength(contour, true);
        vector<Point> approx;
        approxPolyDP(contour, approx, 0.018 * perimeter, true);
        
        if (approx.size() >= 4 && approx.size() <= 6) {
            Rect boundRect = boundingRect(approx);
            double aspectRatio = (double)boundRect.width / boundRect.height;
            
            if (aspectRatio > 2.0 && aspectRatio < 6.5) {
                double relativeArea = area / (image.rows * image.cols);
                if (relativeArea > 0.005 && relativeArea < 0.3) {
                    // Validasi bahwa ini plat hitam
                    if (isBlackPlate(boundRect, image)) {
                        cout << "  → Found BLACK PLATE candidate!" << endl;
                        candidates.push_back(boundRect);
                    }
                }
            }
        }
    }
    
    return candidates;
}

#pragma endregion

#pragma region Handle Car
float ReadCar::detectCar(const Mat& frame, bool saveDebug) {
    // Muat model MobileNet SSD (COCO trained)
    static dnn::Net net;
    static bool modelLoaded = false;
    
    if (!modelLoaded) {
        cout << "Loading car detection model..." << endl;
        net = dnn::readNetFromONNX("models/object_detection_yolox_2022nov.onnx");
        
        if (net.empty()) {
            cerr << "ERROR: Failed to load model!" << endl;
            return 0.0f;
        }
        
        // Set backend dan target untuk performa optimal
        net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        
        modelLoaded = true;
        cout << "Model loaded successfully!" << endl;
    }

    if (frame.empty()) {
        cerr << "ERROR: Empty frame for car detection!" << endl;
        return 0.0f;
    }

    // Class names untuk COCO dataset
    map<int, string> classNames = {
        {2, "car"},
        {3, "motorcycle"},
        {5, "bus"},
        {7, "truck"}
    };

    // Preprocess image
    Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(640, 640),
                              Scalar(0, 0, 0), true, false);

    net.setInput(blob);

    // Forward pass
    auto startTime = chrono::high_resolution_clock::now();
    Mat output = net.forward();
    auto endTime = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    cout << "Detection inference time: " << duration.count() << " ms" << endl;

    // Debug output shape
    cout << "Output dimensions: ";
    for (int i = 0; i < output.dims; i++) {
        cout << output.size[i] << " ";
    }
    cout << endl;

    // Parse detections - PERBAIKAN DI SINI
    // Output biasanya berbentuk [1, 1, N, 7] atau [1, N, 7]
    int numDetections = output.size[2];
    float* data = (float*)output.data;
    
    vector<CarDetection> carDetections;
    float highestConf = 0.0f;
    CarDetection bestDetection;

    int frameHeight = frame.rows;
    int frameWidth = frame.cols;

    cout << "\n=== CAR DETECTION RESULTS ===" << endl;
    cout << "Total detections found: " << numDetections << endl;

    for (int i = 0; i < numDetections; i++) {
        // Setiap deteksi punya 7 elemen: [batchId, classId, confidence, x1, y1, x2, y2]
        int offset = i * 7;
        
        float batchId = data[offset + 0];
        int classId = (int)data[offset + 1];
        float confidence = data[offset + 2];
        float x1 = data[offset + 3];
        float y1 = data[offset + 4];
        float x2 = data[offset + 5];
        float y2 = data[offset + 6];

        // Filter: hanya vehicle classes dan confidence > threshold
        if (classNames.find(classId) != classNames.end() && confidence > 0.3f) {
            // Convert normalized coordinates ke pixel coordinates
            x1 *= frameWidth;
            y1 *= frameHeight;
            x2 *= frameWidth;
            y2 *= frameHeight;

            // Pastikan koordinat dalam bounds
            x1 = max(0.0f, min(x1, (float)frameWidth - 1));
            y1 = max(0.0f, min(y1, (float)frameHeight - 1));
            x2 = max(0.0f, min(x2, (float)frameWidth));
            y2 = max(0.0f, min(y2, (float)frameHeight));

            // Pastikan x2 > x1 dan y2 > y1
            if (x2 <= x1 || y2 <= y1) continue;

            Rect bbox(Point((int)x1, (int)y1), Point((int)x2, (int)y2));

            CarDetection detection;
            detection.classId = classId;
            detection.className = classNames[classId];
            detection.confidence = confidence;
            detection.bbox = bbox;

            carDetections.push_back(detection);

            cout << "  [" << i << "] " << detection.className 
                 << " | Confidence: " << fixed << setprecision(2) 
                 << (confidence * 100.0f) << "%" 
                 << " | BBox: (" << bbox.x << "," << bbox.y << "," 
                 << bbox.width << "x" << bbox.height << ")" << endl;

            // Track detection dengan confidence tertinggi
            if (confidence > highestConf) {
                highestConf = confidence;
                bestDetection = detection;
            }
        }
    }

    // Output hasil
    if (highestConf > 0.0f) {
        cout << "\n✓ VEHICLE DETECTED!" << endl;
        cout << "  Type: " << bestDetection.className << endl;
        cout << "  Confidence: " << fixed << setprecision(2) 
             << (highestConf * 100.0f) << "%" << endl;
        cout << "  Total vehicles found: " << carDetections.size() << endl;
    } else {
        cout << "\n✗ NO VEHICLE DETECTED" << endl;
    }

    // Save debug image jika diminta
    if (saveDebug && !carDetections.empty()) {
        Mat debugFrame = frame.clone();
        
        // Gambar semua deteksi
        for (const auto& det : carDetections) {
            // Warna berbeda untuk setiap class
            Scalar color;
            if (det.classId == 2) color = Scalar(0, 255, 0);      // car - hijau
            else if (det.classId == 3) color = Scalar(255, 0, 0); // motorcycle - biru
            else if (det.classId == 5) color = Scalar(0, 165, 255); // bus - orange
            else if (det.classId == 7) color = Scalar(0, 0, 255); // truck - merah

            // Gambar bounding box
            rectangle(debugFrame, det.bbox, color, 3);

            // Label background
            string label = det.className + " " + 
                          to_string((int)(det.confidence * 100)) + "%";
            int baseline;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 
                                         0.8, 2, &baseline);
            
            Point labelPos(det.bbox.x, det.bbox.y - 10);
            if (labelPos.y < labelSize.height) labelPos.y = det.bbox.y + labelSize.height;

            // Background untuk text
            rectangle(debugFrame, 
                     Point(labelPos.x, labelPos.y - labelSize.height - baseline),
                     Point(labelPos.x + labelSize.width, labelPos.y + baseline),
                     color, FILLED);

            // Text
            putText(debugFrame, label, labelPos, 
                   FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        }

        // Tambahkan info summary di pojok kiri atas
        string summaryText = "Vehicles: " + to_string(carDetections.size()) + 
                            " | Best: " + bestDetection.className + " " +
                            to_string((int)(highestConf * 100)) + "%";
        
        rectangle(debugFrame, Point(5, 5), Point(600, 45), 
                 Scalar(0, 0, 0), FILLED);
        putText(debugFrame, summaryText, Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        // Save dengan timestamp
        time_t now = time(0);
        tm* ltm = localtime(&now);
        char timestamp[50];
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", ltm);

        string filename = "found_car_" + string(timestamp) + "_" + 
                         bestDetection.className + "_" + 
                         to_string((int)(highestConf * 100)) + "pct.jpg";
        
        bool success = imwrite(filename, debugFrame);
        if (success) {
            cout << "\n→ Debug image saved: " << filename << endl;
        } else {
            cerr << "ERROR: Failed to save debug image!" << endl;
        }
    }

    cout << "==============================\n" << endl;

    return highestConf * 100.0f; // return probability dalam persen
}
#pragma endregion