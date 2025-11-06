cmake_minimum_required(VERSION 4.0)

project(ReadCar)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE "")
file(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/scripts/*.cpp")
foreach(SOURCE_FILE ${SOURCE_FILES})
    list(APPEND SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
endforeach()

set(HEADER "")
file(GLOB_RECURSE HEADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/headers/*.hpp")
foreach(HEADER_FILE ${HEADER_FILES})
    list(APPEND HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}")
endforeach()

find_package(OpenCV REQUIRED)
find_package(Crow REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(nlohmann_json REQUIRED)

pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPTONICA REQUIRED lept)

include_directories(
    /usr/include/opencv4
    /usr/include/crow
    src/headers
    /usr/include/tesseract
    /usr/include/onnxruntime
)

file(COPY images DESTINATION ${CMAKE_BINARY_DIR})
file(COPY models DESTINATION ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME} src/main.cpp ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS} 
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    nlohmann_json::nlohmann_json
    onnxruntime
)
