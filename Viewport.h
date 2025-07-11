#pragma once
#include <utility>

class Viewport {
public:
    double centerReal;
    double centerImag;
    double distance; // Height of the view in complex plane
    int windowWidth;
    int windowHeight;

    Viewport(double real = 0.0, double imag = 0.0, double dist = 4.0, int width = 800, int height = 600)
        : centerReal(real), centerImag(imag), distance(dist), windowWidth(width), windowHeight(height) {}

    // Getters
    double getCenterReal() const { return centerReal; }
    double getCenterImag() const { return centerImag; }
    double getDistance() const { return distance; }
    int getWidth() const { return windowWidth; }
    int getHeight() const { return windowHeight; }

    // Setters
    void setCenter(double real, double imag) { centerReal = real; centerImag = imag; }
    void setCenterReal(double real) { centerReal = real; }
    void setCenterImag(double imag) { centerImag = imag; }
    void setDistance(double dist) { distance = dist; }

    void updateWindowSize(int width, int height) {
        windowWidth = width;
        windowHeight = height;
    }

    // Map screen (pixel) coordinates to complex plane
    std::pair<double, double> screenToComplex(int x, int y) const {
        double aspect = static_cast<double>(windowWidth) / windowHeight;
        double realRange = distance * aspect;
        double imagRange = distance;
        double pixelWidth = realRange / windowWidth;
        double pixelHeight = imagRange / windowHeight;
        double real = centerReal + (x - windowWidth / 2.0) * pixelWidth;
        double imag = centerImag + (windowHeight / 2.0 - y) * pixelHeight;
        return std::make_pair(real, imag);
    }

    // Map complex plane coordinates to screen (pixel)
    std::pair<int, int> complexToScreen(double real, double imag) const {
        double aspect = static_cast<double>(windowWidth) / windowHeight;
        double realRange = distance * aspect;
        double imagRange = distance;
        double pixelWidth = realRange / windowWidth;
        double pixelHeight = imagRange / windowHeight;
        int x = static_cast<int>((real - centerReal) / pixelWidth + windowWidth / 2.0);
        int y = static_cast<int>((centerImag - imag) / pixelHeight + windowHeight / 2.0);
        return std::make_pair(x, y);
    }

    // Zoom around a screen point (x, y)
    void zoomAt(int x, int y, double zoomFactor) {
        std::pair<double, double> coords = screenToComplex(x, y);
        double mouseReal = coords.first;
        double mouseImag = coords.second;
        distance *= zoomFactor;
        // Keep the point under the cursor fixed
        centerReal = mouseReal - (x - windowWidth / 2.0) * (distance * (static_cast<double>(windowWidth) / windowHeight) / windowWidth);
        centerImag = mouseImag + (y - windowHeight / 2.0) * (distance / windowHeight);
    }

    // Pan by screen deltas (in pixels)
    void pan(int deltaX, int deltaY) {
        double aspect = static_cast<double>(windowWidth) / windowHeight;
        double realRange = distance * aspect;
        double imagRange = distance;
        double pixelWidth = realRange / windowWidth;
        double pixelHeight = imagRange / windowHeight;
        centerReal -= deltaX * pixelWidth;
        centerImag += deltaY * pixelHeight;
    }
}; 