#ifndef GPU_DETECTOR_H
#define GPU_DETECTOR_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <vector>

struct GPUInfo {
    int deviceId;
    std::string name;
    size_t totalMemory;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int multiProcessorCount;
    int warpSize;
    int maxGridSize[3];
    int maxBlockSize[3];
    bool supportsDoublePrecision;
    bool supportsWarpShuffle;
    bool supportsSharedMemory;
};

class GPUDetector {
private:
    std::vector<GPUInfo> gpuList;
    bool cudaAvailable;
    int selectedDevice;

public:
    GPUDetector();
    ~GPUDetector();
    
    // Basic CUDA capability detection
    bool isCUDAAvailable() const;
    
    // Detailed GPU scanning
    bool scanGPUs();
    
    // Get GPU information
    const std::vector<GPUInfo>& getGPUList() const;
    const GPUInfo& getSelectedGPU() const;
    int getSelectedDeviceId() const;
    
    // GPU optimization helpers
    void selectOptimalGPU();
    void setOptimalGridBlockSize(int totalPixels, int& gridSize, int& blockSize);
    
    // Error handling
    std::string getLastError() const;
    
private:
    void printGPUInfo(const GPUInfo& gpu);
    bool checkDoublePrecisionSupport(int deviceId);
    bool checkWarpShuffleSupport(int deviceId);
    bool checkSharedMemorySupport(int deviceId);
};

#endif // GPU_DETECTOR_H 