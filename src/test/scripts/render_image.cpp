#include <render_image.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

GLuint HandlerImage::loadTextureFromFile(const char* filename) {
    // Load image from file
    int channels, width, height;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 4);
    if (data == NULL) {
        cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }

    // Create OpenGL texture
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    // Setup texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Upload image data to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    // Free image memory
    stbi_image_free(data);
    
    return texture_id;
}