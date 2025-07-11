#ifndef MANDELBROT_CPU_H
#define MANDELBROT_CPU_H

#include <vector>
#include <chrono>

struct MandelbrotParams {
    double realCenter;
    double imagCenter;
    double distance;
    int maxIterations;
    int width;
    int height;
};

class MandelbrotCPU {
private:
    std::vector<int> results;
    MandelbrotParams params;
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;

public:
    MandelbrotCPU();
    ~MandelbrotCPU();
    
    // Set parameters
    void setParameters(const MandelbrotParams& params);
    void setParameters(double realCenter, double imagCenter, double distance, 
                      int maxIterations, int width, int height);
    
    // Compute Mandelbrot set
    void compute();
    
    // Get results
    const std::vector<int>& getResults() const;
    int* getResultsPtr();
    
    // Performance measurement
    double getComputationTime() const;
    void resetTimer();
    
    // Optimized computation methods
    void computeOptimized();
    void computeSIMD();
    void computeParallel();
    
private:
    // Core Mandelbrot computation
    int computePoint(double real, double imag) const;
    
    // Optimization helpers
    void computeRow(int row);
    void computeBlock(int startRow, int endRow);
    
    // SIMD helpers (if available)
    void computeRowSIMD(int row);
};

#endif // MANDELBROT_CPU_H 