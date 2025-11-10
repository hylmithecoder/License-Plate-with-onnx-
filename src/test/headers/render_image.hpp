#pragma once
#include <iostream>
#include <GLFW/glfw3.h>
using namespace std;

class HandlerImage {
    public:
        GLuint loadTextureFromFile(const char* filename);
};
