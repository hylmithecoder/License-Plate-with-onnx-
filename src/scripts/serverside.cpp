#include <serverside.hpp>

#pragma region static Helper only
string getFileNameFromParams(const unordered_map<string, string>& map) {
    auto it = map.find("filename");
    if (it != map.end()) {
        return it->second;
    }
    return "";
}

#pragma endregion

void ServerSide::uploadsRoute(SimpleApp& app){
    CROW_ROUTE(app, "/upload")
    .methods(crow::HTTPMethod::Post)
    ([this](const crow::request& req) -> crow::response {
        string content_type = req.get_header_value("Content-Type");
        if (content_type.find("multipart/form-data") == string::npos) {
            return crow::response(crow::status::BAD_REQUEST, "Invalid Content-Type");
        }

        try {
            crow::multipart::message file_message(req);
            for (const auto& part : file_message.parts) {
                string file_name = getFileNameFromParams(
                    part.headers.find("Content-Disposition")->second.params
                );

                if (!file_name.empty()) {
                    fs::create_directories("images");
                    string saveFilePath = "images/" + file_name;

                    ofstream out(saveFilePath, ios::binary);
                    if (!out) {
                        return crow::response(500, "Failed to save file");
                    }

                    out.write(part.body.c_str(), part.body.size());
                    out.close();

                    // ⚡ Jalankan inference async
                    auto future = async(launch::async, [this, saveFilePath]() {
                        return inference->detectPlateWithOCR(saveFilePath);
                    });

                    // Tunggu hasil tanpa nge-block main thread event loop
                    PlateResult result = future.get();

                    nlohmann::json response_data = showInferenceResult(result);
                    response_data["filename"] = file_name;
                    response_data["size"] = part.body.size();

                    string response_json = response_data.dump();
                    crow::response res(200);
                    res.set_header("Content-Type", "application/json");
                    res.body = response_json;
                    return res;
                }
            }
            return crow::response(400, "No file found in request");

        } catch (const exception& e) {
            CROW_LOG_ERROR << "Upload error: " << e.what();
            return crow::response(500, "Internal Server Error");
        }
    });
}

nlohmann::json ServerSide::showInferenceResult(const PlateResult& result){
    nlohmann::json response_data;
    response_data["pelat_nomor"] = result.plateText;

    return response_data;
}

void ServerSide::response(SimpleApp& app){
    CROW_ROUTE(app, "/json_complex")
    ([]{
        json::wvalue user;
        user["id"] = 123;
        user["name"] = "John Doe";
        user["email"] = "john.doe@example.com";

        json::wvalue address = json::wvalue::object();
        address["street"] = "123 Main St";
        address["city"] = "Anytown";
        address["zip"] = "12345";

        user["address"] = address.dump(); // Nesting the address object

        json::wvalue response_data;
        response_data["user_info"] = user.dump();
        response_data["timestamp"] = time(nullptr); // Example of including dynamic data

        return response_data;
    });
}

void ServerSide::handleServer(){
    inference->initModel("models/license-plate-finetune-v1m.onnx");
    SimpleApp app;
    uploadsRoute(app);
    response(app);
    cout << "Route / berjalan di port 18080" << endl;
    app.port(18080).multithreaded().run();
}