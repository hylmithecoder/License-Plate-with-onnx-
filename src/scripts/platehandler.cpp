#include <handlerimage.hpp>

Rect ReadCar::filterBestPlate(const vector<Rect>& candidates, const Mat& image) {
    if (candidates.empty()) return Rect();
    
    Rect bestPlate;
    double bestScore = -1.0;
    bool foundBlackPlate = false;
    
    for (const auto& candidate : candidates) {
        if (!validatePlateRegion(candidate, image)) continue;
        
        bool isBlack = isBlackPlate(candidate, image);
        
        // Scoring
        double aspectRatio = (double)candidate.width / candidate.height;
        double aspectScore = 1.0 - abs(aspectRatio - 3.33) / 3.33;
        
        double positionY = (double)candidate.y / image.rows;
        double positionScore = (positionY > 0.4) ? 1.0 : positionY / 0.4;
        
        double area = candidate.width * candidate.height;
        double imageArea = image.rows * image.cols;
        double sizeRatio = area / imageArea;
        double sizeScore = (sizeRatio > 0.01 && sizeRatio < 0.15) ? 1.0 : 0.5;
        
        double totalScore = (aspectScore * 0.5) + (positionScore * 0.3) + (sizeScore * 0.2);
        
        // BONUS: Black plate dapat prioritas lebih tinggi
        if (isBlack) {
            totalScore *= 1.2; // 20% bonus untuk plat hitam
            cout << "  → BLACK PLATE candidate boosted! Score: " << totalScore << endl;
        }
        
        if (totalScore > bestScore) {
            bestScore = totalScore;
            bestPlate = candidate;
            foundBlackPlate = isBlack;
        }
    }
    
    if (foundBlackPlate) {
        cout << "  ✓ Best candidate is BLACK PLATE" << endl;
    } else {
        cout << "  ✓ Best candidate is WHITE PLATE" << endl;
    }
    
    return bestPlate;
}


vector<Rect> ReadCar::detectPlateCandidates(const Mat& image, string nameImage) {
    vector<Rect> allCandidates;
    
    // ========== 1. DETECT VIA ONNX MODEL (PRIORITY) ==========
    vector<Detection> modelDetections;
    if (modelLoaded) {
        cout << "  → Running ONNX inference..." << endl;
        modelDetections = detectWithONNX(image, nameImage);
        cout << "  → Found " << modelDetections.size() << " ONNX detection(s)" << endl;
        
        for (const auto& det : modelDetections) {
            allCandidates.push_back(det.box);
        }
    }
    
    // ========== 2. FALLBACK: CONTOUR-BASED DETECTION ==========
    if (allCandidates.empty()) {
        cout << "  → Falling back to contour detection..." << endl;
        
        Mat processed = preprocessImage(image, nameImage);
        vector<vector<Point>> contours;
        findContours(processed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        
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
                        if (!isBlackPlate(boundRect, image)) {
                            allCandidates.push_back(boundRect);
                        }
                    }
                }
            }
        }
        
        cout << "  → Found " << allCandidates.size() << " WHITE plate candidate(s)" << endl;
    }
    
    // ========== 3. BLACK PLATE DETECTION ==========
    vector<Rect> blackCandidates = detectBlackPlateCandidates(image, nameImage);
    cout << "  → Found " << blackCandidates.size() << " BLACK plate candidate(s)" << endl;
    
    allCandidates.insert(allCandidates.end(), blackCandidates.begin(), blackCandidates.end());
    
    // Debug visualization
    Mat debug = image.clone();
    for (size_t i = 0; i < allCandidates.size(); i++) {
        Scalar color = (i < modelDetections.size()) ? 
                           Scalar(255, 0, 0) : Scalar(0, 255, 0); // Blue for ONNX, Green for contour
        rectangle(debug, allCandidates[i], color, 2);
    }
    imwrite("output/debug_candidates_" + nameImage, debug);
    
    return allCandidates;
}

Mat ReadCar::preprocessPlateForOCR(const Mat& plateROI) {
    Mat processed;
    
    // Resize untuk OCR lebih baik
    if (plateROI.rows < 100) {
        double scale = 100.0 / plateROI.rows;
        resize(plateROI, processed, Size(), scale, scale, INTER_CUBIC);
    } else {
        processed = plateROI.clone();
    }
    
    // Convert to grayscale
    Mat gray;
    if (processed.channels() == 3) {
        cvtColor(processed, gray, COLOR_BGR2GRAY);
    } else {
        gray = processed.clone();
    }
    
    // Cek apakah plat hitam atau putih
    Scalar meanBrightness = mean(gray);
    bool isBlack = meanBrightness[0] < 100;
    
    if (isBlack) {
        cout << "  → Preprocessing for BLACK PLATE" << endl;
        Mat inverted;
        bitwise_not(gray, inverted);
        gray = inverted;
    } else {
        cout << "  → Preprocessing for WHITE PLATE" << endl;
    }
    
    // Enhance contrast dengan CLAHE
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat enhanced;
    clahe->apply(gray, enhanced);
    
    // Bilateral filter
    Mat filtered;
    bilateralFilter(enhanced, filtered, 11, 17, 17);
    
    // Adaptive threshold
    Mat binary;
    adaptiveThreshold(filtered, binary, 255,
                      ADAPTIVE_THRESH_GAUSSIAN_C,
                      THRESH_BINARY, 21, 10);
    
    // Morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);
    
    // Denoise
    Mat denoised;
    fastNlMeansDenoising(binary, denoised, 10, 7, 21);

    // --- Tambahan: buat rectangle tebal ---
    Mat debugRect = plateROI.clone();
    
    // Buat persegi sedikit lebih kecil (biar border-nya ga keluar frame)
    int pad = 5;
    Rect safeRect(pad, pad, debugRect.cols - pad * 2, debugRect.rows - pad * 2);
    
    // Gambar rectangle agak tebal warna hijau
    rectangle(debugRect, safeRect, Scalar(0, 255, 0), 3);

    // Simpan hasil rectangled image
    string outPath = "rectangled_last_" + to_string(rand() % 99999) + ".jpg";
    imwrite(outPath, debugRect);
    cout << "  → Saved debug rect image: " << outPath << endl;

    // Mat repaired = repairCharacters(denoised, "last_" + outPath);
    // return repaired;
    return denoised;

}

cv::Mat ReadCar::repairCharacters(const cv::Mat& binaryInput, const std::string& currentNameImage) {
    if (binaryInput.empty()) {
        std::cerr << "  ✗ repairCharacters: input kosong!" << std::endl;
        return binaryInput;
    }

    cv::Mat repaired = binaryInput.clone();

    try {
        // 🔹 1. Pastikan grayscale
        if (repaired.channels() == 3) {
            cv::cvtColor(repaired, repaired, cv::COLOR_BGR2GRAY);
        }

        // 🔹 2. Nyambungin bagian huruf yang putus halus (morphological close)
        cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(repaired, repaired, cv::MORPH_CLOSE, kernelClose);

        // 🔹 3. Hapus noise kecil (morphological open)
        cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(repaired, repaired, cv::MORPH_OPEN, kernelOpen);

        // 🔹 4. Sedikit “inpaint” gap putih (pakai mask dari area hitam tipis)
        cv::Mat mask;
        cv::threshold(repaired, mask, 200, 255, cv::THRESH_BINARY_INV);
        cv::inpaint(repaired, mask, repaired, 1.5, cv::INPAINT_TELEA);

        // 🔹 5. Gambar border tebal buat debug
        cv::Mat debugImage;
        cv::cvtColor(repaired, debugImage, cv::COLOR_GRAY2BGR);

        int borderThickness = 3;
        cv::rectangle(debugImage, cv::Rect(0, 0, repaired.cols, repaired.rows),
                      cv::Scalar(0, 255, 0), borderThickness);

        // 🔹 6. Simpan debug output
        std::string outputPath = "images/rectangled_last_" + currentNameImage;
        cv::imwrite(outputPath, debugImage);

        std::cout << "  ✓ repairCharacters selesai, disimpan di: " << outputPath << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "  ✗ repairCharacters error: " << e.what() << std::endl;
    }

    return repaired;
}
