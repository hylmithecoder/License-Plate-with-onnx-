cmake_minimum_required(VERSION 4.0)

project(ReadCar)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NFD_PATH "/home/hylmi/Hylmi/Pemrograman_Berorientasi_Objek/C++/nativefiledialog-extended")

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

set(TEST_SOURCE
    src/test/main_test.cpp
    src/test/scripts/test_tesseract.cpp
    src/test/scripts/render_image.cpp
)

set(NFD_SOURCE
    ${NFD_PATH}/src/nfd_gtk.cpp
)

set(IMGUI_SOURCES
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_demo.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/imgui/backends/imgui_impl_glfw.cpp
    vendor/imgui/backends/imgui_impl_opengl3.cpp
)

find_package(OpenCV REQUIRED)
find_package(Crow REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(glfw3 REQUIRED)

pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPTONICA REQUIRED lept)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

include_directories(
    /usr/include/opencv4
    /usr/include/crow
    src/headers
    /usr/include/tesseract
    /usr/include/onnxruntime
    vendor/imgui
    ${NFD_PATH}/src/include
    ${GTK3_INCLUDE_DIRS}
    /usr/include/stb
)

set(GLAD_SOURCE
    vendor/glad4.4/src/glad.c
)

file(COPY images DESTINATION ${CMAKE_BINARY_DIR})
file(COPY models DESTINATION ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME} src/main.cpp ${SOURCE_FILES} ${HEADER_FILES})
add_executable(TestTesseract ${TEST_SOURCE} ${IMGUI_SOURCES} ${NFD_SOURCE} ${GLAD_SOURCE})

target_include_directories(TestTesseract PRIVATE 
    src/test/headers
    ${CMAKE_SOURCE_DIR}/vendor/glad4.4/include
)

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS} 
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    nlohmann_json::nlohmann_json
    onnxruntime
)

target_link_libraries(TestTesseract 
    ${OpenCV_LIBS} 
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    glfw
    nlohmann_json::nlohmann_json
    onnxruntime
    ${NFD_PATH}/build/src/libnfd.so.1.2.1
    ${GTK3_LINK_LIBRARIES}
    OpenGL::GL
)
