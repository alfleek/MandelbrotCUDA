#pragma once

#include <SFML/Graphics.hpp>
#include <functional>
#include <vector>
#include <chrono>
#include "Viewport.h"

class RendererSFML {
private:
    sf::RenderWindow window;
    sf::Texture texture;
    sf::Sprite sprite;
    
    // Note: windowWidth/Height now refer to the texture size
    int textureWidth;
    int textureHeight;
    bool dataUpdated;

    // Overscan factor to maintain constant texture/window ratio
    float overscanFactor;
    
    // Mandelbrot parameters
    int maxIterations;
    
    // Color palette
    std::vector<sf::Color> colorPalette;
    
    // Viewport for coordinate mapping
    Viewport viewport;
    
    // Mouse interaction
    bool isDragging;
    bool isInteracting;
    sf::Vector2i lastMousePos;
    sf::Vector2i startMousePos; // For stable panning
    sf::Vector2f startSpritePos; // For stable panning
    
    // Visual transformation for immediate feedback
    Viewport visualViewport; // For immediate feedback during interaction
    bool isTransforming;
    
    // Timer for zoom commit
    std::chrono::steady_clock::time_point lastZoomTime;
    static constexpr int ZOOM_COMMIT_TIMEOUT_MS = 200;
    
    // Callback for parameter changes
    std::function<void(double, double, double, int)> onParamsChanged;
    
    // Callback for window resize events
    std::function<void(int, int)> onWindowResize;
    
    void setupColorPalette();
    sf::Color getColor(int iteration) const;
    void applyVisualTransformation();

public:
    RendererSFML();
    ~RendererSFML();
    
    bool init(int textureWidth, int textureHeight, int windowWidth, int windowHeight, const std::string& title = "Mandelbrot Set");
    void close();
    
    void updateMandelbrotData(const std::vector<int>& data);
    void updateMandelbrotData(const int* data, int size);
    
    void setMandelbrotParams(double realCenter, double imagCenter, double distance, int maxIterations);
    void setParamsChangedCallback(std::function<void(double, double, double, int)> callback);
    void setWindowResizeCallback(std::function<void(int, int)> callback);
    
    void update();
    void updateTexture();
    void render();
    void drawUI();
    void handleEvents();
    
    bool isOpen() const;
    
    // Window resize handling
    void handleWindowResize(int newWidth, int newHeight);
    
    // Mouse event handlers
    void handleMousePress(int x, int y, bool leftButton);
    void handleMouseRelease(int x, int y);
    void handleMouseMove(int x, int y);
    void handleMouseWheel(int delta);
    
    // Zoom commit method
    void commitZoom();
    
    // Get current viewport for external use
    const Viewport& getViewport() const { return viewport; }

    float getOverscanFactor() const { return overscanFactor; }
}; 