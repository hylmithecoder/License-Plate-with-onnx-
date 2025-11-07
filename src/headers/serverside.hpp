#ifndef SERVERSIDE_HPP
#define SERVERSIDE_HPP

#pragma once
#include <crow.h>
#include <crow/json.h>
#include <iostream>
#include <filesystem>
#include <multipart.h>
#include <string>
#include <regex>
#include <fstream>
#include <nlohmann/json.hpp>
#include <handlerimage.hpp>

namespace fs = std::filesystem;
using namespace crow;
using namespace std;

class ServerSide{
    public:
        ReadCar* inference = new ReadCar();
        PlateResult result;
        nlohmann::json showInferenceResult(const PlateResult& result);
        void handleServer();
        void response(SimpleApp& app);
        void uploadsRoute(SimpleApp& app);
};
#endif // SERVERSIDE_HPP