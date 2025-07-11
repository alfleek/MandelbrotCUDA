#include "renderer_sfml.h"
#include <iostream>
#include <cmath>

// Define M_PI for Windows if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

RendererSFML::RendererSFML() 
    : textureWidth(1600), textureHeight(900), dataUpdated(false),
      maxIterations(1000), overscanFactor(1.4f),
      // viewport represents the entire texture, visualViewport represents the window
      viewport(0.0, 0.0, 4.0, 1600, 900),
      visualViewport(0.0, 0.0, 4.0, 1600, 900), // Will be updated in init
      isDragging(false), isInteracting(false), lastMousePos(0, 0),
      isTransforming(false), lastZoomTime(std::chrono::steady_clock::now()),
      onParamsChanged(nullptr), onWindowResize(nullptr) {
    setupColorPalette();
}

RendererSFML::~RendererSFML() {
    close();
}

bool RendererSFML::init(int texWidth, int texHeight, int winWidth, int winHeight, const std::string& title) {
    textureWidth = texWidth;
    textureHeight = texHeight;
    
    // Derive overscan factor from initial sizes to keep it constant thereafter
    overscanFactor = static_cast<float>(textureWidth) / static_cast<float>(winWidth);
    
    // Create window with specified dimensions
    window.create(sf::VideoMode(winWidth, winHeight), title);
    window.setFramerateLimit(60);
    
    // Create texture with oversized dimensions
    if (!texture.create(textureWidth, textureHeight)) {
        std::cerr << "Failed to create texture" << std::endl;
        return false;
    }
    
    sprite.setTexture(texture);
    
    // viewport represents the entire texture
    viewport.updateWindowSize(textureWidth, textureHeight);
    // visualViewport represents the visible window area
    visualViewport.updateWindowSize(winWidth, winHeight);
    
    std::cout << "SFML renderer initialized successfully" << std::endl;
    return true;
}

void RendererSFML::close() {
    if (window.isOpen()) {
        window.close();
    }
}

void RendererSFML::setupColorPalette() {
    colorPalette.clear();
    
    // Create a smooth color palette
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        
        // Create smooth color transitions
        int r = static_cast<int>(255 * (0.5f + 0.5f * sin(t * 2.0f * M_PI)));
        int g = static_cast<int>(255 * (0.5f + 0.5f * sin(t * 2.0f * M_PI + 2.0f * M_PI / 3.0f)));
        int b = static_cast<int>(255 * (0.5f + 0.5f * sin(t * 2.0f * M_PI + 4.0f * M_PI / 3.0f)));
        
        colorPalette.push_back(sf::Color(r, g, b));
    }
}

sf::Color RendererSFML::getColor(int iteration) const {
    if (iteration >= maxIterations) {
        return sf::Color::Black; // Inside the set
    }
    
    // Logarithmic mapping for better visual distribution
    float normalized = std::log(static_cast<float>(iteration) + 1.0f) / std::log(static_cast<float>(maxIterations));
    int index = static_cast<int>(normalized * (colorPalette.size() - 1));
    
    if (index >= 0 && index < static_cast<int>(colorPalette.size())) {
        return colorPalette[index];
    }
    
    return sf::Color::White;
}

void RendererSFML::updateMandelbrotData(const std::vector<int>& data) {
    if (data.size() != static_cast<size_t>(textureWidth * textureHeight)) {
        std::cerr << "Data size mismatch: expected " << (textureWidth * textureHeight) << ", got " << data.size() << std::endl;
        return;
    }
    
    // Prepare a pixel buffer (RGBA)
    std::vector<sf::Uint8> pixels(textureWidth * textureHeight * 4);
    for (size_t i = 0; i < textureWidth * textureHeight; ++i) {
        sf::Color color = getColor(data[i]);
        pixels[i * 4 + 0] = color.r;
        pixels[i * 4 + 1] = color.g;
        pixels[i * 4 + 2] = color.b;
        pixels[i * 4 + 3] = 255;
    }
    
    // Update texture
    texture.update(pixels.data());
    dataUpdated = true;
}

void RendererSFML::updateMandelbrotData(const int* data, int size) {
    if (size != textureWidth * textureHeight) {
        std::cerr << "Data size mismatch: expected " << (textureWidth * textureHeight) << ", got " << size << std::endl;
        return;
    }
    
    // Prepare a pixel buffer (RGBA)
    std::vector<sf::Uint8> pixels(textureWidth * textureHeight * 4);
    for (int i = 0; i < textureWidth * textureHeight; ++i) {
        sf::Color color = getColor(data[i]);
        pixels[i * 4 + 0] = color.r;
        pixels[i * 4 + 1] = color.g;
        pixels[i * 4 + 2] = color.b;
        pixels[i * 4 + 3] = 255;
    }
    
    // Update texture
    texture.update(pixels.data());
    dataUpdated = true;
}

void RendererSFML::setMandelbrotParams(double realCenter, double imagCenter, double distance, int maxIterations) {
    this->maxIterations = maxIterations;
    
    // Update viewport with new parameters
    viewport.setCenter(realCenter, imagCenter);
    viewport.setDistance(distance);
    
    // Reset visual transformation
    isTransforming = false;
    sprite.setScale(1.0f, 1.0f);
    sprite.setPosition(0, 0);
    
    // Update visual viewport to match
    visualViewport.setCenter(realCenter, imagCenter);
    visualViewport.setDistance(distance);
    
    // Recreate color palette for new max iterations
    setupColorPalette();
}

void RendererSFML::setParamsChangedCallback(std::function<void(double, double, double, int)> callback) {
    onParamsChanged = callback;
}

void RendererSFML::setWindowResizeCallback(std::function<void(int, int)> callback) {
    onWindowResize = callback;
}

void RendererSFML::update() {
    // Check if we should commit zoom after timeout
    if (isInteracting && !isDragging) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastZoomTime);
        if (elapsed.count() >= ZOOM_COMMIT_TIMEOUT_MS) {
            commitZoom();
        }
    }
}

void RendererSFML::updateTexture() {
    if (dataUpdated) {
        dataUpdated = false;
    }
}

void RendererSFML::render() {
    window.clear(sf::Color::Black);
    
    // Update texture if data changed
    updateTexture();
    
    // Apply visual transformation only during interaction
    if (isInteracting)
        applyVisualTransformation();
    else {
        // When not interacting, the sprite is not scaled (1:1 pixel mapping).
        // It is positioned to center the oversized texture within the window.
        float offsetX = (viewport.getWidth() - visualViewport.getWidth()) / 2.0f;
        float offsetY = (viewport.getHeight() - visualViewport.getHeight()) / 2.0f;

        sprite.setPosition(-offsetX, -offsetY);
        sprite.setScale(1.0f, 1.0f);
    }
    
    // Draw the Mandelbrot image
    window.draw(sprite);
    
    // Draw UI
    drawUI();
    
    window.display();
}

void RendererSFML::drawUI() {
    // Simple text overlay with parameters
    std::string title = "Mandelbrot Set - Center: (" + 
                       std::to_string(viewport.getCenterReal()).substr(0, 8) + ", " + 
                       std::to_string(viewport.getCenterImag()).substr(0, 8) + ") Zoom: " + 
                       std::to_string(1.0 / viewport.getDistance()).substr(0, 8);
    window.setTitle(title);
}

void RendererSFML::handleEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        switch (event.type) {
            case sf::Event::Closed:
                window.close();
                break;
                
            case sf::Event::Resized:
                handleWindowResize(event.size.width, event.size.height);
                break;
                
            case sf::Event::MouseButtonPressed:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    handleMousePress(event.mouseButton.x, event.mouseButton.y, true);
                }
                break;
                
            case sf::Event::MouseButtonReleased:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    handleMouseRelease(event.mouseButton.x, event.mouseButton.y);
                }
                break;
                
            case sf::Event::MouseMoved:
                handleMouseMove(event.mouseMove.x, event.mouseMove.y);
                break;
                
            case sf::Event::MouseWheelScrolled:
                handleMouseWheel(static_cast<int>(event.mouseWheelScroll.delta));
                break;
                
            case sf::Event::KeyPressed:
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                }
                break;
        }
    }
}

bool RendererSFML::isOpen() const {
    return window.isOpen();
}

void RendererSFML::handleWindowResize(int newWidth, int newHeight) {
    // The window itself has been resized. We need to update our state.
    // newWidth and newHeight are the new WINDOW dimensions.

    // Update the view to match the new window size
    sf::View view = window.getView();
    view.setSize(static_cast<float>(newWidth), static_cast<float>(newHeight));
    view.setCenter(static_cast<float>(newWidth) / 2.f, static_cast<float>(newHeight) / 2.f);
    window.setView(view);

    // The texture size needs to be recalculated based on the new window size
    int newTextureWidth = static_cast<int>(newWidth * overscanFactor);
    int newTextureHeight = static_cast<int>(newHeight * overscanFactor);

    textureWidth = newTextureWidth;
    textureHeight = newTextureHeight;
    
    // Update viewports with new size and sync their centers/distances
    viewport.updateWindowSize(textureWidth, textureHeight);
    visualViewport.updateWindowSize(newWidth, newHeight);
    visualViewport.setCenter(viewport.getCenterReal(), viewport.getCenterImag());
    visualViewport.setDistance(viewport.getDistance());

    // Cancel any ongoing interaction
    isDragging = false;
    isInteracting = false;
    
    // Recreate texture for new size
    if (!texture.create(textureWidth, textureHeight)) {
        std::cerr << "Failed to recreate texture for new window size" << std::endl;
        return;
    }
    sprite.setTexture(texture, true);
    
    // Reset visual transformation to prevent artifacts
    sprite.setPosition(0, 0);
    sprite.setScale(1.0f, 1.0f);
    
    // Mark that data needs to be updated for new size
    dataUpdated = true;
    
    // Notify main application of window resize
    if (onWindowResize) {
        onWindowResize(newWidth, newHeight);
    }
    
    std::cout << "[DEBUG] Renderer: Window resized to " << newWidth << "x" << newHeight << std::endl;
    std::cout << "[DEBUG] Renderer: Texture size is now " << texture.getSize().x << "x" << texture.getSize().y << std::endl;
    std::cout << "[DEBUG] Renderer: viewport size is " << viewport.getWidth() << "x" << viewport.getHeight() << std::endl;
    std::cout << "[DEBUG] Renderer: visualViewport size is " << visualViewport.getWidth() << "x" << visualViewport.getHeight() << std::endl;
}

void RendererSFML::handleMousePress(int x, int y, bool leftButton) {
    if (leftButton) {
        isDragging = true;
        isInteracting = true;
        // Store the starting positions for the drag operation
        startMousePos = sf::Vector2i(x, y);
        startSpritePos = sprite.getPosition();
        lastMousePos = startMousePos; // Initialize lastMousePos for delta calcs
        
        // Sync visual viewport to the committed viewport's state
        // without corrupting its dimensions.
        visualViewport.setCenter(viewport.getCenterReal(), viewport.getCenterImag());
        visualViewport.setDistance(viewport.getDistance());

        isTransforming = true;
    }
}

void RendererSFML::handleMouseRelease(int x, int y) {
    if (isDragging) {
        isDragging = false;
        isInteracting = false;

        // The visual viewport has been updated by the sequence of pans.
        // Now we commit this state to the main viewport.
        viewport.setCenter(visualViewport.getCenterReal(), visualViewport.getCenterImag());
        viewport.setDistance(visualViewport.getDistance());
        
        if (onParamsChanged) {
            onParamsChanged(viewport.getCenterReal(), viewport.getCenterImag(), viewport.getDistance(), maxIterations);
        }
    }
}

void RendererSFML::handleMouseMove(int x, int y) {
    if (isDragging) {
        // For panning, we calculate the new sprite position directly
        // relative to the start of the drag for a stable 1:1 mapping.
        float deltaX = static_cast<float>(x - startMousePos.x);
        float deltaY = static_cast<float>(y - startMousePos.y);
        
        sprite.setPosition(startSpritePos.x + deltaX, startSpritePos.y + deltaY);

        // We still need to update the visual viewport for the final commit.
        // This tracks the total pan distance in the complex plane.
        int frameDeltaX = x - lastMousePos.x;
        int frameDeltaY = y - lastMousePos.y;

        // Because the texture is larger than the window by overscanFactor,
        // each window-pixel drag corresponds to 1/overscanFactor texture pixels
        // in the complex plane.
        double scaledDeltaX = static_cast<double>(frameDeltaX) / overscanFactor;
        double scaledDeltaY = static_cast<double>(frameDeltaY) / overscanFactor;

        // pan() expects integer deltas in pixel space. Round for stability.
        visualViewport.pan(static_cast<int>(std::round(scaledDeltaX)),
                           static_cast<int>(std::round(scaledDeltaY)));
        lastMousePos = sf::Vector2i(x, y);
    }
}

void RendererSFML::handleMouseWheel(int delta) {
    sf::Vector2i mousePos = sf::Mouse::getPosition(window);
    isInteracting = true;
    lastZoomTime = std::chrono::steady_clock::now();
    
    // Calculate zoom factor
    double zoomFactor = delta > 0 ? 0.9 : 1.1;
    
    // Use viewport's zoom method
    visualViewport.zoomAt(mousePos.x, mousePos.y, zoomFactor);
    applyVisualTransformation();
}

void RendererSFML::applyVisualTransformation() {
    // The goal is to transform the sprite (which represents the `viewport`)
    // so that it visually matches the state of `visualViewport`.

    // 1. Calculate the scale factor. This is purely the ratio of the "zoom levels" (distance).
    double scale = viewport.getDistance() / visualViewport.getDistance();

    // 2. The anchor for the transformation is the center of the target view.
    //    Find where the center of the `visualViewport` is located in the texture's pixel coords.
    std::pair<double, double> visualCenter = {visualViewport.getCenterReal(), visualViewport.getCenterImag()};
    std::pair<int, int> pixelOfVisualCenter = viewport.complexToScreen(visualCenter.first, visualCenter.second);

    // 3. The target on-screen position for this anchor point is the center of the window.
    float targetX = visualViewport.getWidth() / 2.0f;
    float targetY = visualViewport.getHeight() / 2.0f;

    // 4. We want to position the sprite such that the point `pixelOfVisualCenter` (after scaling) 
    //    ends up at `(targetX, targetY)`.
    //    The screen position of a texture point `p` is `spritePos + p * scale`.
    //    So, we solve for the sprite's position: `spritePos = target - pixelOfVisualCenter * scale`.
    float offsetX = targetX - (static_cast<float>(pixelOfVisualCenter.first) * scale);
    float offsetY = targetY - (static_cast<float>(pixelOfVisualCenter.second) * scale);

    sprite.setPosition(offsetX, offsetY);
    sprite.setScale(scale, scale);
}

void RendererSFML::commitZoom() {
    if (isInteracting && !isDragging) {
        isInteracting = false;
        
        // Commit visual parameters to actual parameters
        viewport.setCenter(visualViewport.getCenterReal(), visualViewport.getCenterImag());
        viewport.setDistance(visualViewport.getDistance());
        
        if (onParamsChanged) {
            onParamsChanged(viewport.getCenterReal(), viewport.getCenterImag(), viewport.getDistance(), maxIterations);
        }
    }
}

 