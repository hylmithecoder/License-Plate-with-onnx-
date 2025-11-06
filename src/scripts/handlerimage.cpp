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
cv::Mat ReadCar::preprocessImage(const cv::Mat& image, const std::string& nameImage) {
    cv::Mat gray, blurred, thresh, processed;

    // 1. Convert to grayscale
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("output/step1_gray_" + nameImage, gray);

    // 2. Apply bilateral filter (mengurangi noise tapi tetap jaga edge)
    cv::bilateralFilter(gray, blurred, 11, 17, 17);
    cv::imwrite("output/step2_blurred_" + nameImage, blurred);

    // 3. Apply adaptive threshold
    cv::adaptiveThreshold(blurred, thresh, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 21, 10);
    cv::imwrite("output/step3_thresh_" + nameImage, thresh);

    // 4. Morphological operations untuk cleanup
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, processed, cv::MORPH_CLOSE, kernel);
    cv::imwrite("output/step4_morph_" + nameImage, processed);

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

string ReadCar::performOCR(const cv::Mat& plateROI, float& confidence) {
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
    cv::imwrite("output/debug_original_" + debug_suffix + ".jpg", plateROI);
    
    // Preprocess plate untuk OCR
    cv::Mat processed = preprocessPlateForOCR(plateROI);
    cv::imwrite("output/debug_processed_" + debug_suffix + ".jpg", processed);
    
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
        cv::Mat binary2;
        cv::threshold(processed, binary2, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::imwrite("output/debug_otsu_" + debug_suffix + ".jpg", binary2);
        
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
            cv::Mat inverted;
            cv::bitwise_not(processed, inverted);
            cv::threshold(inverted, inverted, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            cv::imwrite("output/debug_inverted_" + debug_suffix + ".jpg", inverted);
            
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

bool ReadCar::validatePlateRegion(const cv::Rect& region, const cv::Mat& image) {
    if (region.width <= 0 || region.height <= 0) return false;
    if (region.x < 0 || region.y < 0) return false;
    if (region.x + region.width > image.cols) return false;
    if (region.y + region.height > image.rows) return false;
    
    cv::Mat plateROI = image(region);
    cv::Mat grayPlate;
    cv::cvtColor(plateROI, grayPlate, cv::COLOR_BGR2GRAY);
    
    // Hitung edge density (plat nomor harusnya punya banyak edge karena ada karakter)
    cv::Mat edges;
    cv::Canny(grayPlate, edges, 100, 200);
    int edgeCount = cv::countNonZero(edges);
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
    
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        cerr << "  ✗ Error: Could not load image!" << endl;
        return result;
    }
    
    // Resize jika terlalu besar
    cv::Mat resized = image.clone();
    if (image.cols > 1280) {
        double scale = 1280.0 / image.cols;
        cv::resize(image, resized, cv::Size(), scale, scale);
    }
    
    // Detect candidates
    vector<cv::Rect> candidates = detectPlateCandidates(resized, result.filename);
    cout << "  → Found " << candidates.size() << " plate candidate(s)" << endl;
    
    if (candidates.empty()) {
        cout << "  ✗ No plate detected" << endl;
        
        string filename = fs::path(path).stem().string();
        string outputPath = "output/no_plate_" + filename + ".jpg";
        cv::imwrite(outputPath, image);
        return result;
    }
    
    // Get best plate
    cv::Rect bestPlate = filterBestPlate(candidates, resized);
    
    if (bestPlate.width == 0) {
        cout << "  ✗ No valid plate found" << endl;
        plateText = "UNRECOGNIZED";
        string filename = fs::path(path).stem().string();
        string outputPath = "output/no_plate_" + filename + ".jpg";
        cv::imwrite(outputPath, image);
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
    cv::Mat plateROI = image(bestPlate);
    
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
    cv::Mat output = image.clone();
    cv::Scalar color = isValid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255); // Green or Orange
    cv::rectangle(output, bestPlate, color, 3);
    
    // Add text dengan background
    string displayText = plateText.empty() ? "UNREADABLE" : plateText;
    string confText = " (" + to_string((int)confidence) + "%)";
    displayText += confText;
    
    int baseline = 0;
    double fontScale = 0.8;
    int thickness = 2;
    cv::Size textSize = cv::getTextSize(displayText, cv::FONT_HERSHEY_SIMPLEX, 
                                         fontScale, thickness, &baseline);
    
    cv::Point textOrg(bestPlate.x, bestPlate.y - 15);
    
    // Background
    cv::rectangle(output,
                  textOrg + cv::Point(0, baseline + 5),
                  textOrg + cv::Point(textSize.width, -textSize.height - 5),
                  color, cv::FILLED);
    
    cv::putText(output, displayText, textOrg, cv::FONT_HERSHEY_SIMPLEX, 
                fontScale, cv::Scalar(0, 0, 0), thickness);
    
    // Save hasil
    string filename = fs::path(path).stem().string();
    string outputFull = "output/detected_" + filename + ".jpg";
    string outputCrop = "output/plate_" + filename + ".jpg";
    string outputProcessed = "output/processed_" + filename + ".jpg";
    
    cv::imwrite(outputFull, output);
    cv::imwrite(outputCrop, plateROI);
    
    // Save preprocessed version
    cv::Mat processedPlate = preprocessPlateForOCR(plateROI);
    cv::imwrite(outputProcessed, processedPlate);
    
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
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

    if (image.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return;
    }

    cv::Mat gray_image, blurred_image, edges;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_image, blurred_image, cv::Size(5, 5), 0);
    cv::Canny(blurred_image, edges, 100, 200);

    cv::imshow("Grayscale Image", gray_image);
    cv::imshow("Canny Edges", edges);
    cv::waitKey(0);
}

bool ReadCar::isRectangle(const string& path) {
    // Legacy function - sekarang gunakan detectAndSavePlate()
    detectAndSavePlate(path);
    return true;
}

#pragma region Handle Black plate
bool ReadCar::isBlackPlate(const cv::Rect& region, const cv::Mat& image) {
    // Safety check
    if (region.x < 0 || region.y < 0 || 
        region.x + region.width > image.cols || 
        region.y + region.height > image.rows) {
        return false;
    }
    
    cv::Mat plateROI = image(region);
    cv::Mat grayPlate;
    
    if (plateROI.channels() == 3) {
        cv::cvtColor(plateROI, grayPlate, cv::COLOR_BGR2GRAY);
    } else {
        grayPlate = plateROI.clone();
    }
    
    // Hitung mean brightness
    cv::Scalar meanBrightness = cv::mean(grayPlate);
    double avgBrightness = meanBrightness[0];
    
    // Plat hitam biasanya punya average brightness < 80
    // Plat putih biasanya > 120
    bool isDark = avgBrightness < 100;
    
    if (isDark) {
        // Double check: plat hitam harus punya text putih/terang
        // Cek histogram untuk memastikan ada pixel terang
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&grayPlate, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
        
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

cv::Mat ReadCar::preprocessForBlackPlate(const cv::Mat& image, string nameImage) {
    cv::Mat gray, blurred, processed;
    
    // Convert to grayscale
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("output/step1_gray_black_" + nameImage, gray);
    
    // Bilateral filter
    cv::bilateralFilter(gray, blurred, 11, 17, 17);    
    cv::imwrite("output/step2_blurred_black_" + nameImage, blurred);

    // Untuk plat HITAM: gunakan THRESH_BINARY (bukan INV)
    // Karena kita cari text PUTIH di background HITAM
    cv::adaptiveThreshold(blurred, processed, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 21, 10); // BINARY, bukan BINARY_INV    
    // cv::imwrite("output/step3_thresh_black_" + nameImage, blurred);

    // Morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);
    cv::imwrite("output/step3_morph_black_" + nameImage, processed);
    
    return processed;
}

vector<cv::Rect> ReadCar::detectBlackPlateCandidates(const cv::Mat& image, string nameImage) {
    cv::Mat processed = preprocessForBlackPlate(image, nameImage);
    
    vector<vector<cv::Point>> contours;
    cv::findContours(processed, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    vector<cv::Rect> candidates;
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 500) continue;
        
        double perimeter = cv::arcLength(contour, true);
        vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.018 * perimeter, true);
        
        if (approx.size() >= 4 && approx.size() <= 6) {
            cv::Rect boundRect = cv::boundingRect(approx);
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