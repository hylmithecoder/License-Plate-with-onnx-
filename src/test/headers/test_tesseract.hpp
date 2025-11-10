#pragma once
#include <iostream>
#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <leptonica/allheaders.h>
#include <onnxruntime_cxx_api.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <nfd.hpp>
#include <render_image.hpp>

namespace fs = std::filesystem;
using namespace std;
using namespace cv::dnn;
using namespace ImGui;
using namespace cv;
using namespace tesseract;

class HandlerTesseract {
    public:
        HandlerTesseract() {
            // Initialize Tesseract OCR
            ocr = new TessBaseAPI();
    
            // Init dengan bahasa Inggris (untuk karakter alfanumerik)
            if (ocr->Init(NULL, "eng", OEM_LSTM_ONLY)) {
                cerr << "Could not initialize tesseract!" << endl;
                exit(1);
            }
            
            // Set Page Segmentation Mode
            ocr->SetPageSegMode(PSM_SINGLE_LINE);
            
            // Whitelist karakter (hanya huruf kapital dan angka)
            ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
            
            cout << "✓ Tesseract OCR initialized successfully" << endl;
        }

        ~HandlerTesseract() {
            if (ocr) {
                ocr->End();
                delete ocr;
            }
        }

        void analyzeImage();
        void window();        
        void openFile();
        GLuint cvMatToGLTexture(const Mat& image);
        void imageRender();
        string tesseractWithThreshImage(const Mat& inputImg);

    private:
        HandlerImage imgHandler;
        TessBaseAPI* ocr;
        string recognizedText;
        float textConfidence = 0.0f;
        string currentFile;
        vector<GLuint> imageTextures;

        vector<string> loadDict(const string& dictPath) {
            vector<string> dict;
            ifstream file(dictPath);
            string line;
            while (getline(file, line))
                dict.push_back(line);
            return dict;
        }

        string ocrPaddle(const Mat& input, const string& detPath, const string& recPath, const string& dictPath) {
            // Load dict
            vector<string> dict = loadDict(dictPath);
            dict.insert(dict.begin(), " "); // add space at index 0

            // Load detection model
            Net detNet = readNetFromONNX(detPath);
            Net recNet = readNetFromONNX(recPath);

            // Step 1: Text detection
            Mat blob = blobFromImage(input, 1.0/255.0, Size(960, 960), Scalar(0,0,0), true, false);
            detNet.setInput(blob);
            Mat detOut = detNet.forward();

            // Note: PaddleOCR det output adalah map confidence untuk text regions
            // kita simplify aja — threshold lalu cari contour text region
            Mat probMap(detOut.size[2], detOut.size[3], CV_32F, detOut.ptr<float>());
            Mat binMap;
            threshold(probMap, binMap, 0.3, 255, THRESH_BINARY);
            binMap.convertTo(binMap, CV_8U);

            vector<vector<Point>> contours;
            findContours(binMap, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            // Step 2: Loop tiap region -> crop -> recognize
            string finalText;
            for (auto& c : contours) {
                Rect box = boundingRect(c);
                if (box.area() < 200) continue;
                Mat roi = input(box);

                // Resize to PaddleOCR rec input (100x32 typical)
                Mat recBlob = blobFromImage(roi, 1.0/255.0, Size(100, 32), Scalar(0,0,0), true, false);
                recNet.setInput(recBlob);
                Mat recOut = recNet.forward();

                // Decode
                Mat scores(recOut.size[1], recOut.size[2], CV_32F, recOut.ptr<float>());
                string text;
                int lastIndex = -1;
                for (int t = 0; t < scores.cols; t++) {
                    Point classIdPoint;
                    double confidence;
                    minMaxLoc(scores.col(t), 0, &confidence, 0, &classIdPoint);
                    int index = classIdPoint.y;
                    if (index > 0 && index < dict.size() && index != lastIndex) {
                        text += dict[index];
                    }
                    lastIndex = index;
                }

                if (!text.empty()) {
                    finalText += text + " ";
                }
            }

            return finalText;
        }

        void detectCharacters(const Mat& plateImg) {
            Mat gray, binary, morph;

            // 1️⃣ Konversi ke grayscale
            cvtColor(plateImg, gray, COLOR_BGR2GRAY);

            // 2️⃣ Threshold biar kontras tinggi (huruf putih, background hitam)
            threshold(gray, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

            // 3️⃣ Sedikit morphological opening buat hapus noise kecil
            Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
            morphologyEx(binary, morph, MORPH_OPEN, kernel);

            // 4️⃣ Cari contour (huruf/angka biasanya kecil dan panjang ke atas)
            vector<vector<Point>> contours;
            findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            // 5️⃣ Filter karakter berdasarkan area dan aspect ratio
            vector<Rect> charBoxes;
            for (const auto& contour : contours) {
                Rect box = boundingRect(contour);
                float aspectRatio = (float)box.width / (float)box.height;
                int area = box.area();

                // Filter kasar buat karakter plat (huruf/angka)
                if (area > 50 && area < 5000 && aspectRatio > 0.2 && aspectRatio < 1.0) {
                    charBoxes.push_back(box);
                }
            }

            // 6️⃣ Urutkan dari kiri ke kanan (karakter plat sebaris)
            sort(charBoxes.begin(), charBoxes.end(), [](const Rect& a, const Rect& b) {
                return a.x < b.x;
            });

            // 7️⃣ Gambar hasil deteksi
            Mat debug = plateImg.clone();
            for (size_t i = 0; i < charBoxes.size(); ++i) {
                rectangle(debug, charBoxes[i], Scalar(0, 255, 0), 2);
                putText(debug, to_string(i + 1), Point(charBoxes[i].x, charBoxes[i].y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            }

            // Simpan hasil debug
            imwrite("debug_detected_chars.jpg", debug);
            GLuint createTexture = imgHandler.loadTextureFromFile("debug_detected_chars.jpg");
            imageTextures.push_back(createTexture);
            cout << "Detected " << charBoxes.size() << " potential characters." << endl;
        }

        // Function untuk OCR dengan Tesseract yang sudah optimized
        string recognizePlateWithTesseract(const Mat& plateImage) {
            // 1. Straighten and preprocess
            Mat straightened = straightenPlate(plateImage);
            
            if (straightened.empty()) {
                return "";
            }
            
            // 2. Run Tesseract
            TessBaseAPI *api = new TessBaseAPI();
            
            // Initialize with English (for Indonesian plates)
            
            OcrEngineMode typeTesseract = OEM_LSTM_ONLY;
            if (api->Init(NULL, "eng", typeTesseract)) {
                cerr << "Could not initialize tesseract." << endl;
                delete api;
                return "";
            }
            
            // Set page segmentation mode untuk single line text
            api->SetPageSegMode(PSM_SINGLE_LINE);
            
            // Whitelist characters (Indonesian plates: A-Z, 0-9)
            api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
            
            // Set image
            // cout << "Straightend data:  " << straightened.data << endl;
            cout << "Straightend cols:   " << straightened.cols << endl;
            cout << "Straightend rows:   " << straightened.rows << endl;
            cout << "Straightend channels: " << straightened.channels() << endl;
            cout << "Straightend step:   " << straightened.step1() << endl;
            api->SetImage(straightened.data, straightened.cols, straightened.rows, 
                        straightened.channels(), straightened.step1());
            
            // Get OCR result
            char* outText = api->GetUTF8Text();
            cout << "OCR result: " << outText << endl;
            string result = string(outText);
            
            // Clean up
            delete[] outText;
            api->End();
            delete api;
            
            // Post-processing
            // Remove whitespace and non-alphanumeric
            result.erase(remove_if(result.begin(), result.end(), 
                                [](char c) { return !isalnum(c); }), 
                        result.end());
            
            // Convert to uppercase
            transform(result.begin(), result.end(), result.begin(), ::toupper);
            
            return result;
        }

        Mat straightenPlate(const Mat& input) {
            Mat gray, thresh;
            
            // Convert to grayscale
            cvtColor(input, gray, COLOR_BGR2GRAY);
            
            // Apply adaptive thresholding
            adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
            string currentNamefile = "debug_thresh.jpg";
            imwrite(currentNamefile, thresh);
            GLuint createTexture = imgHandler.loadTextureFromFile(currentNamefile.c_str());
            imageTextures.push_back(createTexture);
            string readThrestImage = tesseractWithThreshImage(thresh);
            cout << "Read thrested image: " << readThrestImage << endl;
            // Find contours
            vector<vector<Point>> contours;
            findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            cout << "Countours size: " << contours.size() << endl;
            // Find the largest contour that could be a license plate
            vector<Point> plateContour;
            double maxArea = 0;
            
            // for (const auto& contour : contours) {
            //     double area = contourArea(contour);
            //     if (area > 200) { // Minimal area threshold
            //         RotatedRect rect = minAreaRect(contour);
            //         double aspectRatio = rect.size.width / rect.size.height;
                    
            //         // Typical license plate aspect ratio is between 2 and 5
            //         if (aspectRatio > 2 && aspectRatio < 5) {
            //             if (area > maxArea) {
            //                 maxArea = area;
            //                 plateContour = contour;
            //             }
            //         }
            //     }
            // }

            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area > maxArea) {
                    maxArea = area;
                    plateContour = contour;
                }
            }
            
            if (plateContour.empty()) {
                return input;
            }

            cout << "PLate contour size: " << plateContour.size() << endl;
            
            // Get rotated rectangle
            RotatedRect rotatedRect = minAreaRect(plateContour);
            Point2f vertices[4];
            rotatedRect.points(vertices);

            // Sort points to get proper order: TL, TR, BR, BL
            vector<Point2f> srcPts(vertices, vertices + 4);
            vector<Point2f> dstPts = {
                Point2f(0, 0),                          // Top-Left
                Point2f(rotatedRect.size.width, 0),     // Top-Right
                Point2f(rotatedRect.size.width, rotatedRect.size.height), // Bottom-Right
                Point2f(0, rotatedRect.size.height)     // Bottom-Left
            };

            // Sort source points
            sort(srcPts.begin(), srcPts.end(), [](const Point2f& a, const Point2f& b) {
                return a.y < b.y;
            });

            // Split top and bottom points
            vector<Point2f> top(srcPts.begin(), srcPts.begin() + 2);
            vector<Point2f> bottom(srcPts.begin() + 2, srcPts.end());

            // Sort top points by x
            sort(top.begin(), top.end(), [](const Point2f& a, const Point2f& b) {
                return a.x < b.x;
            });

            // Sort bottom points by x
            sort(bottom.begin(), bottom.end(), [](const Point2f& a, const Point2f& b) {
                return a.x < b.x;
            });

            // Combine points in correct order
            srcPts = {top[0], top[1], bottom[1], bottom[0]};

            // Draw debug visualization
            Mat debugDraw = input.clone();
            for (int i = 0; i < 4; i++) {
                circle(debugDraw, srcPts[i], 5, Scalar(0, 0, 255), -1);
                line(debugDraw, srcPts[i], srcPts[(i+1)%4], Scalar(0, 255, 0), 2);
            }
            imwrite("debug_corners.jpg", debugDraw);

            // Calculate perspective transform
            Mat perspectiveMatrix = getPerspectiveTransform(srcPts, dstPts);
            
            // Apply perspective transform
            Mat warped;
            warpPerspective(input, warped, perspectiveMatrix, 
                        Size(rotatedRect.size.width, rotatedRect.size.height));

            resize(warped, warped, Size(warped.cols * 5.6, warped.rows));
            // Save debug output
            imwrite("debug_warped.jpg", warped);

            // --- [Tambahan untuk lurusin jajar genjang] ---
            Mat graypar, threshpar;
            cvtColor(warped, graypar, COLOR_BGR2GRAY);
            threshold(graypar, threshpar, 0, 255, THRESH_BINARY | THRESH_OTSU);

            vector<vector<Point>> contourspar;
            findContours(threshpar, contourspar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!contourspar.empty()) {
                // Ambil contour terbesar
                size_t largestIdx = 0;
                double maxArea = 0;
                for (size_t i = 0; i < contourspar.size(); i++) {
                    double area = contourArea(contourspar[i]);
                    if (area > maxArea) {
                        maxArea = area;
                        largestIdx = i;
                    }
                }

                RotatedRect rect = minAreaRect(contourspar[largestIdx]);

                // Buat matrix transform baru untuk ngeratain plat (deskew)
                Mat M = getRotationMatrix2D(rect.center, rect.angle, 1.0);
                Mat deskewed;
                warpAffine(warped, deskewed, M, warped.size(), INTER_CUBIC);

                // Simpan hasil "jajar genjang" dan hasil lurusannya
                imwrite("debug_warped_parallelogram.jpg", warped);
                imwrite("debug_warped_deskewed.jpg", deskewed);

                warped = deskewed; // opsional, biar lanjut ke OCR pakai hasil lurus
            } else {
                imwrite("debug_warped_parallelogram.jpg", warped);
            }

            // return warped;
            
            // Get angle and size
            float angle = rotatedRect.angle;
            Size rect_size = rotatedRect.size;
            
            // Ensure width is larger than height
            if (rect_size.width < rect_size.height) {
                swap(rect_size.width, rect_size.height);
                angle += 90.0;
            }
            
            // Get rotation matrix
            Mat rotationMatrix = getRotationMatrix2D(rotatedRect.center, angle, 1.0);
            
            // Rotate the image
            Mat rotated;
            warpAffine(input, rotated, rotationMatrix, input.size(), INTER_CUBIC);
            
            // Crop the plate region
            Rect bbox = rotatedRect.boundingRect();
            bbox.x = max(bbox.x, 0);
            bbox.y = max(bbox.y, 0);
            bbox.width = min(bbox.width, input.cols - bbox.x);
            bbox.height = min(bbox.height, input.rows - bbox.y);
            
            Mat cropped = rotated(bbox);
            
            // Resize to standard size for better OCR
            Mat resized;
            resize(cropped, resized, Size(400, 100));
            
            // Save debug output
            imwrite("debug_rotated.jpg", rotated);
            imwrite("debug_cropped.jpg", cropped);
            imwrite("debug_resized.jpg", resized);

            detectCharacters(resized);           

            return resized;
        }

        Mat straightenPlateDebugging(const Mat& input, bool saveDebug = true) {
            if (input.empty()) {
                cerr << "Error: Input image is empty!" << endl;
                return input;
            }

            cout << "\n=== PLATE STRAIGHTENING DEBUG ===" << endl;
            cout << "Input size: " << input.cols << "x" << input.rows << endl;

            // ========== STEP 1: GRAYSCALE ==========
            Mat gray;
            if (input.channels() == 3) {
                cvtColor(input, gray, COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }
            
            if (saveDebug) {
                imwrite("debug_01_gray.jpg", gray);
                cout << "[1] Grayscale saved" << endl;
            }

            // ========== STEP 2: NOISE REDUCTION ==========
            Mat denoised;
            // Gunakan bilateral filter untuk keep edges tapi buang noise
            bilateralFilter(gray, denoised, 11, 75, 75);
            
            if (saveDebug) {
                imwrite("debug_02_denoised.jpg", denoised);
                cout << "[2] Denoised saved" << endl;
            }

            // ========== STEP 3: CONTRAST ENHANCEMENT ==========
            Mat enhanced;
            Ptr<CLAHE> clahe = createCLAHE(2.5, Size(8, 8));
            clahe->apply(denoised, enhanced);
            
            if (saveDebug) {
                imwrite("debug_03_enhanced.jpg", enhanced);
                cout << "[3] Enhanced contrast saved" << endl;
            }

            // ========== STEP 4: THRESHOLDING (Multiple Methods) ==========
            
            // Method A: Adaptive Threshold
            Mat thresh_adaptive;
            adaptiveThreshold(enhanced, thresh_adaptive, 255, 
                            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 3);
            
            // Method B: Otsu Threshold
            Mat thresh_otsu;
            threshold(enhanced, thresh_otsu, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
            
            // Method C: Kombinasi (lebih robust)
            Mat thresh_combined;
            bitwise_and(thresh_adaptive, thresh_otsu, thresh_combined);
            
            if (saveDebug) {
                imwrite("debug_04a_thresh_adaptive.jpg", thresh_adaptive);
                imwrite("debug_04b_thresh_otsu.jpg", thresh_otsu);
                imwrite("debug_04c_thresh_combined.jpg", thresh_combined);
                cout << "[4] All thresholding methods saved" << endl;
            }

            // Pilih threshold terbaik (bisa ganti sesuai hasil)
            Mat thresh = thresh_otsu.clone(); // DEFAULT: Otsu usually best
            
            // ========== STEP 5: MORPHOLOGICAL OPERATIONS ==========
            Mat morph;
            Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
            Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3));
            
            // Close untuk connect nearby components
            morphologyEx(thresh, morph, MORPH_CLOSE, kernel_close);
            // Open untuk remove small noise
            morphologyEx(morph, morph, MORPH_OPEN, kernel_open);
            
            if (saveDebug) {
                imwrite("debug_05_morphology.jpg", morph);
                cout << "[5] Morphology saved" << endl;
            }

            // ========== STEP 6: FIND CONTOURS ==========
            vector<vector<Point>> contours;
            findContours(morph.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            cout << "[6] Total contours found: " << contours.size() << endl;
            
            // Draw all contours untuk debug
            if (saveDebug) {
                Mat allContours = input.clone();
                drawContours(allContours, contours, -1, Scalar(0, 255, 0), 2);
                imwrite("debug_06_all_contours.jpg", allContours);
                cout << "[6] All contours saved" << endl;
            }

            // ========== STEP 7: FILTER & FIND BEST PLATE CONTOUR ==========
            vector<Point> plateContour;
            double maxScore = 0;
            int bestContourIdx = -1;
            
            Mat candidatesDebug = input.clone();
            
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                
                // Filter by minimum area
                if (area < 500) continue;
                
                RotatedRect rect = minAreaRect(contours[i]);
                float width = rect.size.width;
                float height = rect.size.height;
                
                // Ensure width > height
                if (width < height) swap(width, height);
                
                float aspectRatio = width / height;
                
                // Indonesian plate typical ratio: 2.5 - 5.5
                if (aspectRatio < 2.0 || aspectRatio > 6.0) continue;
                
                // Scoring: area + aspect ratio quality
                float idealRatio = 4.3; // 520mm x 120mm
                float ratioScore = 1.0 - min(1.0f, abs(aspectRatio - idealRatio) / idealRatio);
                float score = area * ratioScore;
                
                // Draw candidate dengan score
                Point2f vertices[4];
                rect.points(vertices);
                for (int j = 0; j < 4; j++) {
                    line(candidatesDebug, vertices[j], vertices[(j+1)%4], 
                        Scalar(0, 255, 255), 2);
                }
                
                string label = "A:" + to_string((int)area) + 
                            " R:" + to_string(aspectRatio).substr(0, 4) +
                            " S:" + to_string((int)score);
                putText(candidatesDebug, label, rect.center, 
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
                
                if (score > maxScore) {
                    maxScore = score;
                    plateContour = contours[i];
                    bestContourIdx = i;
                }
            }
            
            if (saveDebug) {
                imwrite("debug_07_candidates.jpg", candidatesDebug);
                cout << "[7] Candidate contours with scores saved" << endl;
            }
            
            if (plateContour.empty()) {
                cerr << "Error: No suitable plate contour found!" << endl;
                if (saveDebug) {
                    imwrite("debug_ERROR_no_contour.jpg", input);
                }
                return input;
            }
            
            cout << "[7] Best contour index: " << bestContourIdx 
                << " (score: " << maxScore << ")" << endl;

            // ========== STEP 8: GET CORNERS & PERSPECTIVE TRANSFORM ==========
            RotatedRect rotatedRect = minAreaRect(plateContour);
            Point2f vertices[4];
            rotatedRect.points(vertices);
            
            // Order points: TL, TR, BR, BL
            vector<Point2f> srcPts(vertices, vertices + 4);
            
            // Sort by Y (top vs bottom)
            sort(srcPts.begin(), srcPts.end(), [](const Point2f& a, const Point2f& b) {
                return a.y < b.y;
            });
            
            // Top 2 points
            Point2f tl = srcPts[0].x < srcPts[1].x ? srcPts[0] : srcPts[1];
            Point2f tr = srcPts[0].x < srcPts[1].x ? srcPts[1] : srcPts[0];
            
            // Bottom 2 points
            Point2f bl = srcPts[2].x < srcPts[3].x ? srcPts[2] : srcPts[3];
            Point2f br = srcPts[2].x < srcPts[3].x ? srcPts[3] : srcPts[2];
            
            srcPts = {tl, tr, br, bl};
            
            // Calculate output dimensions
            float width = max(norm(br - bl), norm(tr - tl));
            float height = max(norm(tr - br), norm(tl - bl));
            
            vector<Point2f> dstPts = {
                Point2f(0, 0),
                Point2f(width - 1, 0),
                Point2f(width - 1, height - 1),
                Point2f(0, height - 1)
            };
            
            // Draw corners dengan nomor
            if (saveDebug) {
                Mat cornersDebug = input.clone();
                vector<Scalar> colors = {
                    Scalar(255, 0, 0),   // TL - Blue
                    Scalar(0, 255, 0),   // TR - Green
                    Scalar(0, 0, 255),   // BR - Red
                    Scalar(255, 255, 0)  // BL - Cyan
                };
                vector<string> labels = {"TL", "TR", "BR", "BL"};
                
                for (int i = 0; i < 4; i++) {
                    circle(cornersDebug, srcPts[i], 8, colors[i], -1);
                    circle(cornersDebug, srcPts[i], 10, Scalar(255, 255, 255), 2);
                    putText(cornersDebug, labels[i], srcPts[i] + Point2f(15, 15),
                        FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
                    line(cornersDebug, srcPts[i], srcPts[(i+1)%4], Scalar(0, 255, 0), 2);
                }
                imwrite("debug_08_corners.jpg", cornersDebug);
                cout << "[8] Corners visualization saved" << endl;
            }
            
            // ========== STEP 9: PERSPECTIVE WARP ==========
            Mat perspectiveMatrix = getPerspectiveTransform(srcPts, dstPts);
            Mat warped;
            warpPerspective(input, warped, perspectiveMatrix, Size(width, height));
            
            if (saveDebug) {
                imwrite("debug_09_warped_raw.jpg", warped);
                cout << "[9] Raw perspective warp saved" << endl;
            }
            
            // ========== STEP 10: DESKEW (Fix Parallelogram) ==========
            Mat warpedGray, warpedThresh;
            if (warped.channels() == 3) {
                cvtColor(warped, warpedGray, COLOR_BGR2GRAY);
            } else {
                warpedGray = warped.clone();
            }
            
            threshold(warpedGray, warpedThresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
            
            vector<vector<Point>> contoursWarped;
            findContours(warpedThresh.clone(), contoursWarped, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            Mat deskewed = warped.clone();
            
            if (!contoursWarped.empty()) {
                // Find largest contour
                size_t largestIdx = 0;
                double maxArea = 0;
                for (size_t i = 0; i < contoursWarped.size(); i++) {
                    double area = contourArea(contoursWarped[i]);
                    if (area > maxArea) {
                        maxArea = area;
                        largestIdx = i;
                    }
                }
                
                RotatedRect rectWarped = minAreaRect(contoursWarped[largestIdx]);
                float angle = rectWarped.angle;
                
                // Adjust angle
                if (rectWarped.size.width < rectWarped.size.height) {
                    angle += 90.0;
                }
                
                // Only deskew if angle is significant
                if (abs(angle) > 0.5) {
                    Point2f center(warped.cols / 2.0, warped.rows / 2.0);
                    Mat M = getRotationMatrix2D(center, angle, 1.0);
                    warpAffine(warped, deskewed, M, warped.size(), 
                            INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));
                    
                    if (saveDebug) {
                        imwrite("debug_10_deskewed.jpg", deskewed);
                        cout << "[10] Deskewed (angle: " << angle << ")" << endl;
                    }
                } else {
                    if (saveDebug) {
                        cout << "[10] No deskew needed (angle: " << angle << ")" << endl;
                    }
                }
            }
            
            // ========== STEP 11: CHECK ORIENTATION ==========
            Mat orientCheck;
            if (deskewed.channels() == 3) {
                cvtColor(deskewed, orientCheck, COLOR_BGR2GRAY);
            } else {
                orientCheck = deskewed.clone();
            }
            
            int h3 = orientCheck.rows / 3;
            Scalar topMean = mean(orientCheck(Rect(0, 0, orientCheck.cols, h3)));
            Scalar botMean = mean(orientCheck(Rect(0, orientCheck.rows - h3, orientCheck.cols, h3)));
            
            Mat oriented = deskewed.clone();
            bool isFlipped = false;
            
            // If bottom is darker (has text), it's upside down
            if (botMean[0] < topMean[0] * 0.85) {
                rotate(oriented, oriented, ROTATE_180);
                isFlipped = true;
            }
            
            if (saveDebug) {
                imwrite("debug_11_oriented.jpg", oriented);
                cout << "[11] Orientation check (flipped: " << (isFlipped ? "YES" : "NO") << ")" << endl;
                cout << "     Top mean: " << topMean[0] << ", Bottom mean: " << botMean[0] << endl;
            }
            
            // ========== STEP 12: CROP TO CONTENT ==========
            Mat cropGray;
            if (oriented.channels() == 3) {
                cvtColor(oriented, cropGray, COLOR_BGR2GRAY);
            } else {
                cropGray = oriented.clone();
            }
            
            Mat cropThresh;
            threshold(cropGray, cropThresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
            
            vector<vector<Point>> cropContours;
            findContours(cropThresh, cropContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            Mat cropped = oriented.clone();
            
            if (!cropContours.empty()) {
                Rect boundBox = boundingRect(cropContours[0]);
                for (size_t i = 1; i < cropContours.size(); i++) {
                    boundBox |= boundingRect(cropContours[i]);
                }
                
                // Add small padding
                int pad = 10;
                boundBox.x = max(0, boundBox.x - pad);
                boundBox.y = max(0, boundBox.y - pad);
                boundBox.width = min(oriented.cols - boundBox.x, boundBox.width + 2*pad);
                boundBox.height = min(oriented.rows - boundBox.y, boundBox.height + 2*pad);
                
                cropped = oriented(boundBox);
                
                if (saveDebug) {
                    imwrite("debug_12_cropped.jpg", cropped);
                    cout << "[12] Cropped to content" << endl;
                }
            }
            
            // ========== STEP 13: RESIZE TO OPTIMAL SIZE ==========
            Mat resized;
            int targetHeight = 100;
            float scale = targetHeight / (float)cropped.rows;
            int targetWidth = (int)(cropped.cols * scale);
            
            resize(cropped, resized, Size(targetWidth, targetHeight), 0, 0, INTER_CUBIC);
            
            if (saveDebug) {
                imwrite("debug_13_resized.jpg", resized);
                cout << "[13] Resized to " << targetWidth << "x" << targetHeight << endl;
            }
            
            // ========== STEP 14: FINAL OCR PREPROCESSING OPTIONS ==========
            
            // Option A: Grayscale + CLAHE + Otsu
            Mat finalA;
            if (resized.channels() == 3) {
                cvtColor(resized, finalA, COLOR_BGR2GRAY);
            } else {
                finalA = resized.clone();
            }
            Ptr<CLAHE> clahe2 = createCLAHE(3.0, Size(8, 8));
            clahe2->apply(finalA, finalA);
            threshold(finalA, finalA, 0, 255, THRESH_BINARY | THRESH_OTSU);
            copyMakeBorder(finalA, finalA, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255));
            
            // Option B: Direct Otsu
            Mat finalB;
            if (resized.channels() == 3) {
                cvtColor(resized, finalB, COLOR_BGR2GRAY);
            } else {
                finalB = resized.clone();
            }
            threshold(finalB, finalB, 0, 255, THRESH_BINARY | THRESH_OTSU);
            copyMakeBorder(finalB, finalB, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255));
            
            // Option C: Adaptive + Bilateral
            Mat finalC;
            if (resized.channels() == 3) {
                cvtColor(resized, finalC, COLOR_BGR2GRAY);
            } else {
                finalC = resized.clone();
            }
            GaussianBlur(finalC, finalC, Size(5, 5), 0);
            adaptiveThreshold(finalC, finalC, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                            THRESH_BINARY, 15, 5);
            copyMakeBorder(finalC, finalC, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255));
            
            if (saveDebug) {
                imwrite("debug_14a_final_clahe_otsu.jpg", finalA);
                imwrite("debug_14b_final_otsu.jpg", finalB);
                imwrite("debug_14c_final_adaptive.jpg", finalC);
                cout << "[14] Three OCR-ready versions saved" << endl;
            }
            
            cout << "\n=== DEBUG COMPLETE ===" << endl;
            cout << "Check debug_XX_*.jpg files to choose best result!" << endl;
            cout << "\nRecommended for OCR:" << endl;
            cout << "  - debug_14a_final_clahe_otsu.jpg (BEST for most cases)" << endl;
            cout << "  - debug_14b_final_otsu.jpg (Good for clean plates)" << endl;
            cout << "  - debug_14c_final_adaptive.jpg (Good for uneven lighting)" << endl;
            
            // Return default best option
            return finalA;
        }

        Mat preprocessPlateForOCR(const Mat& plate) {
            Mat processed;
            
            // Apply bilateral filter to reduce noise while preserving edges
            Mat bilateral;
            bilateralFilter(plate, bilateral, 11, 17, 17);
            
            // Convert to grayscale if not already
            Mat gray;
            if (plate.channels() == 3) {
                cvtColor(bilateral, gray, COLOR_BGR2GRAY);
            } else {
                bilateral.copyTo(gray);
            }
            
            // Enhance contrast using CLAHE
            Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
            Mat enhanced;
            clahe->apply(gray, enhanced);
            // cout << "Enhanced data:  " << enhanced.data << endl;
            cout << "Enhanced cols:   " << enhanced.cols << endl;
            cout << "Enhanced rows:   " << enhanced.rows << endl;
            cout << "Enhanced channels: " << enhanced.channels() << endl;
            cout << "Enhanced step:   " << enhanced.step1() << endl;
            
            // Apply adaptive thresholding
            adaptiveThreshold(enhanced, processed, 255, 
                            ADAPTIVE_THRESH_GAUSSIAN_C, 
                            THRESH_BINARY, 15, 5);
            
            return processed;
        }
};