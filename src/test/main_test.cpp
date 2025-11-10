#include <test_tesseract.hpp>

int main(){
    HandlerTesseract tesseract;
    // cv::VideoCapture cap(0); // Open the default camera
    // if (!cap.isOpened()) {
    //     // Handle error
    //     cout << "Error: Could not open camera." << endl;
    // }
    // cv::Mat frame;
    // GLuint cameraTexture = 0;
    // Init
    if (!glfwInit())
        return 1;
    
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    GLFWwindow* window = glfwCreateWindow(1280, 800, "Test Tesseract", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    io.FontGlobalScale = 1.5f;
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    //Update
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        // 2. Capture a new frame
        // cap >> frame; 
        // if (!frame.empty()) {
        //     // Update the OpenGL texture
        //     if (cameraTexture == 0) {
        //         cameraTexture = tesseract.cvMatToGLTexture(frame);
        //     } else {
        //         glBindTexture(GL_TEXTURE_2D, cameraTexture);
        //         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
        //     }
        // }

        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

        // Main Dockspace
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        
        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(2);
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        // #ifdef ImGuiConfigFlags_DockingEnable
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

        tesseract.window();

        // if (frame.empty()) {
        //     Text("Loading camera feed...");
        // } else {
        //     Begin("Physical Camera Feed");
        //     // ImGui::Image expects a texture ID (void* or GLuint cast to void*)
        //     ImGui::Image((ImTextureID)(uintptr_t)cameraTexture, ImVec2(frame.cols, frame.rows));
        //     End();
        // }
        // End Dockspace
        End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            UpdatePlatformWindows();
            RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    DestroyContext();

    // glDeleteTextures(1, &cameraTexture);
    // cap.release();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}