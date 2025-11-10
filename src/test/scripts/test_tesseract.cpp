#include <test_tesseract.hpp>

void HandlerTesseract::analyzeImage() {
    if (imageTextures.size() > 0){
        imageTextures.clear();
    }

    if (currentFile.empty()) {
        cout << "No file selected" << endl;
        return;
    }

    // Read image
    Mat image = imread(currentFile, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Failed to load image: " << currentFile << endl;
        return;
    } else {
        cout << "Image properties:" << endl;
        cout << "Size: " << image.size() << endl;
        cout << "Channels: " << image.channels() << endl;
        cout << "Type: " << image.type() << endl;
    }

    // Resize if too large
    Mat resized = image.clone();
    if (image.cols > 1280) {
        double scale = 1280.0 / image.cols;
        resize(image, resized, Size(), scale, scale);
    }

    // Straighten the plate
    cout << "Straightening plate..." << endl;
    Mat straightened = straightenPlate(resized);
    if (straightened.empty()) {
        cout << "Plate straightening failed, using original image" << endl;
        straightened = resized.clone();
    }
    
    // Save debug image
    imwrite("debug_straightened.jpg", straightened);

    // string text = ocrPaddle(
    //     straightened,
    //     "models/detections/det.onnx",
    //     "models/language/rec.onnx",
    //     "models/language/dict.txt"
    // );

    // cout << "OCR Result: " << text << endl;

    // Preprocess for OCR

    string plateNumber = recognizePlateWithTesseract(image);
    string plateNumberFromStraight = recognizePlateWithTesseract(straightened);
    cout << "Detected plate: " << plateNumber << endl;
    cout << "Straight plate: " << plateNumberFromStraight << endl;

    cout << "Preprocessing for OCR..." << endl;
    Mat processed = preprocessPlateForOCR(straightened);
    // detectCharacters(processed);
    // Save debug image
    imwrite("debug_processed.jpg", processed);

    // Perform OCR

    // cout << "Processed data:  " << processed.data << endl;
    cout << "Processed cols:   " << processed.cols << endl;
    cout << "Processed rows:   " << processed.rows << endl;
    cout << "Processed channels: " << processed.channels() << endl;
    cout << "Processed step:   " << processed.step1() << endl;
    ocr->SetImage(processed.data, processed.cols, processed.rows, 1, processed.step);
    
    // Get OCR result
    char* outText = ocr->GetUTF8Text();
    float confidence = ocr->MeanTextConf();

    // Store results
    cout << "Out text: " << outText << endl;
    recognizedText = plateNumber;
    textConfidence = confidence;

    // Clean up
    delete[] outText;

    cout << "OCR Results:" << endl;
    cout << "Text: " << recognizedText << endl;
    cout << "Confidence: " << textConfidence << "%" << endl;
}

void HandlerTesseract::window() {
    static bool isOpened = false;
    Begin("Image to text", &isOpened);
    
    Text("Image to text converter");
    if (Button("Open File")) {
        openFile();
    }
    Text(currentFile.c_str(), currentFile.size());

    if (!currentFile.empty()) {
        if (Button("Analyze Image")) {
            analyzeImage();
        }
        
        if (!recognizedText.empty()) {
            Separator();
            // show thresh image
            if (imageTextures.size() >= 1) {
                Text("Processed Image");
                Text("Threshed Image:");
                ImGui::Image((ImTextureID)(uintptr_t)imageTextures[0], ImVec2(400, 200));
                Text("Detected Characters:");
                ImGui::Image((ImTextureID)(uintptr_t)imageTextures[1], ImVec2(400, 200));
            }

            Separator();
            Text("OCR Results:");
            TextWrapped(recognizedText.c_str(), recognizedText.size());
            Text("Confidence: %.2f%%", textConfidence);
        }
    }
    
    End();
}

void HandlerTesseract::openFile(){
    NFD::Guard nfdGuard;

    // auto-freeing memory
    NFD::UniquePath outPath;

    // prepare filters for the dialog
    nfdfilteritem_t filterItem[1] = {{"Source code", "png,jpg,jpeg,webp"}};

    // show the dialog
    nfdresult_t result = NFD::OpenDialog(outPath, filterItem, 1);
    if (result == NFD_OKAY) {
        std::cout << "Success!" << std::endl << outPath.get() << std::endl;
        currentFile = outPath.get();
    } else if (result == NFD_CANCEL) {
        std::cout << "User pressed cancel." << std::endl;
    } else {
        std::cout << "Error: " << NFD::GetError() << std::endl;
    }

}

GLuint HandlerTesseract::cvMatToGLTexture(const Mat& image) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Determine the format
    GLenum format = (image.channels() == 3) ? GL_BGR : GL_LUMINANCE;

    // Upload the image data to the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, format, GL_UNSIGNED_BYTE, image.data);

    glBindTexture(GL_TEXTURE_2D, 0); // Unbind the texture

    return textureID;
}

string HandlerTesseract::tesseractWithThreshImage(const Mat& input){
        // Assuming 'thresh' is your cv::Mat image after adaptiveThreshold
    // Initialize Tesseract
    tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
    ocr->Init(NULL, "eng"); // Initialize with English language

    // Convert OpenCV Mat to Tesseract's Pix format
    // Tesseract expects a Pix image, so convert cv::Mat to Pix
    // This is a common conversion for Tesseract integration
    Pix* image = pixCreate(input.cols, input.rows, 8); // Create an 8-bit Pix image
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            // Assuming thresh is a binary image (0 or 255)
            // Set pixel value in Pix image
            // cout << input.at<uchar>(y, x) << endl;
            pixSetPixel(image, x, y, (input.at<uchar>(y, x) == 0) ? 1 : 0); // Tesseract often expects 0 for text, 1 for background
        }
    }

    ocr->SetImage(image);

    // Perform OCR
    char* text = ocr->GetUTF8Text();
    string output = string(text);

    // Print the recognized text
    std::cout << "OCR Result: " << text << std::endl;

    // Clean up
    delete[] text;
    ocr->End();
    delete ocr;
    pixDestroy(&image);
    return output;
}