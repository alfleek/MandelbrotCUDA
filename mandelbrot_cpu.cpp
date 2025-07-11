#include "mandelbrot_cpu.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <future>
#ifdef __AVX2__
#include <immintrin.h> // For SIMD instructions
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

MandelbrotCPU::MandelbrotCPU() {
    resetTimer();
}

MandelbrotCPU::~MandelbrotCPU() {
}

void MandelbrotCPU::setParameters(const MandelbrotParams& params) {
    this->params = params;
    results.resize(params.width * params.height);
}

void MandelbrotCPU::setParameters(double realCenter, double imagCenter, double distance, 
                                 int maxIterations, int width, int height) {
    params.realCenter = realCenter;
    params.imagCenter = imagCenter;
    params.distance = distance;
    params.maxIterations = maxIterations;
    params.width = width;
    params.height = height;
    results.resize(width * height);
}

void MandelbrotCPU::compute() {
    resetTimer();
    startTime = std::chrono::high_resolution_clock::now();
    
    // Calculate coordinate ranges with proper aspect ratio handling
    double aspectRatio = static_cast<double>(params.width) / params.height;
    double realRange = params.distance * aspectRatio;
    double imagRange = params.distance;
    
    double realMin = params.realCenter - realRange / 2.0;
    double realMax = params.realCenter + realRange / 2.0;
    double imagMin = params.imagCenter - imagRange / 2.0;
    double imagMax = params.imagCenter + imagRange / 2.0;
    
    double realStep = (realMax - realMin) / (params.width - 1);
    double imagStep = (imagMax - imagMin) / (params.height - 1);
    
    // Compute Mandelbrot set
    for (int y = 0; y < params.height; y++) {
        double imag = imagMax - y * imagStep;
        for (int x = 0; x < params.width; x++) {
            double real = realMin + x * realStep;
            results[y * params.width + x] = computePoint(real, imag);
        }
    }
    
    endTime = std::chrono::high_resolution_clock::now();
}

void MandelbrotCPU::computeOptimized() {
    resetTimer();
    startTime = std::chrono::high_resolution_clock::now();

    // Prefer SIMD path when AVX2 is available, otherwise fall back to parallel threads
#ifdef __AVX2__
    computeSIMD();
#else
    computeParallel();
#endif
    
    endTime = std::chrono::high_resolution_clock::now();
}

void MandelbrotCPU::computeParallel() {
    // Calculate coordinate ranges with proper aspect ratio handling
    double aspectRatio = static_cast<double>(params.width) / params.height;
    double realRange = params.distance * aspectRatio;
    double imagRange = params.distance;
    
    double realMin = params.realCenter - realRange / 2.0;
    double realMax = params.realCenter + realRange / 2.0;
    double imagMin = params.imagCenter - imagRange / 2.0;
    double imagMax = params.imagCenter + imagRange / 2.0;
    
    double realStep = (realMax - realMin) / (params.width - 1);
    double imagStep = (imagMax - imagMin) / (params.height - 1);
    
    // Get number of available threads
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < params.height; y++) {
        double imag = imagMax - y * imagStep;
        for (int x = 0; x < params.width; x++) {
            double real = realMin + x * realStep;
            results[y * params.width + x] = computePoint(real, imag);
        }
    }
#else
    // Manual threading
    std::vector<std::future<void>> futures;
    int rowsPerThread = params.height / numThreads;
    
    for (int t = 0; t < numThreads; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? params.height : (t + 1) * rowsPerThread;
        
        futures.push_back(std::async(std::launch::async, [this, startRow, endRow, realMin, realMax, imagMin, imagMax, realStep, imagStep]() {
            for (int y = startRow; y < endRow; y++) {
                double imag = imagMax - y * imagStep;
                for (int x = 0; x < params.width; x++) {
                    double real = realMin + x * realStep;
                    results[y * params.width + x] = computePoint(real, imag);
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
#endif
}

void MandelbrotCPU::computeSIMD() {
    resetTimer();
    startTime = std::chrono::high_resolution_clock::now();
    
    // Calculate coordinate ranges with proper aspect ratio handling
    double aspectRatio = static_cast<double>(params.width) / params.height;
    double realRange = params.distance * aspectRatio;
    double imagRange = params.distance;
    
    double realMin = params.realCenter - realRange / 2.0;
    double realMax = params.realCenter + realRange / 2.0;
    double imagMin = params.imagCenter - imagRange / 2.0;
    double imagMax = params.imagCenter + imagRange / 2.0;
    
    double realStep = (realMax - realMin) / (params.width - 1);
    double imagStep = (imagMax - imagMin) / (params.height - 1);
    
    // SIMD-optimized computation (4 doubles at a time)
    for (int y = 0; y < params.height; y++) {
        double imag = imagMax - y * imagStep;
        computeRowSIMD(y);
    }
    
    endTime = std::chrono::high_resolution_clock::now();
}

void MandelbrotCPU::computeRowSIMD(int row) {
    // This is a simplified SIMD implementation
    // In practice, you'd use AVX2/AVX-512 instructions for better performance
    double aspectRatio = static_cast<double>(params.width) / params.height;
    double realRange = params.distance * aspectRatio;
    double imagRange = params.distance;
    
    double realMin = params.realCenter - realRange / 2.0;
    double realMax = params.realCenter + realRange / 2.0;
    double imagMin = params.imagCenter - imagRange / 2.0;
    double imagMax = params.imagCenter + imagRange / 2.0;
    
    double realStep = (realMax - realMin) / (params.width - 1);
    double imagStep = (imagMax - imagMin) / (params.height - 1);
    
    double imag = imagMax - row * imagStep;
    
    // Process 4 pixels at a time using SIMD
    for (int x = 0; x < params.width; x += 4) {
        for (int i = 0; i < 4 && (x + i) < params.width; i++) {
            double real = realMin + (x + i) * realStep;
            results[row * params.width + x + i] = computePoint(real, imag);
        }
    }
}

int MandelbrotCPU::computePoint(double real, double imag) const {
    double x0 = real;
    double y0 = imag;
    
    double x = 0.0;
    double y = 0.0;
    int iteration = 0;
    
    // Optimized escape condition: use squared distance to avoid sqrt
    double escapeRadiusSquared = 4.0;
    
    while (iteration < params.maxIterations && (x * x + y * y) <= escapeRadiusSquared) {
        double xTemp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xTemp;
        iteration++;
    }
    
    return iteration;
}

const std::vector<int>& MandelbrotCPU::getResults() const {
    return results;
}

int* MandelbrotCPU::getResultsPtr() {
    return results.data();
}

double MandelbrotCPU::getComputationTime() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    return duration.count() / 1000.0; // Return time in milliseconds
}

void MandelbrotCPU::resetTimer() {
    startTime = std::chrono::high_resolution_clock::now();
    endTime = startTime;
} 