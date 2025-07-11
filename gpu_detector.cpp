#include "gpu_detector.h"
#include <iostream>
#include <cstring>
#include <algorithm>

GPUDetector::GPUDetector() : cudaAvailable(false), selectedDevice(-1) {
    // Initialize CUDA runtime
    cudaError_t error = cudaFree(0);
    if (error == cudaSuccess) {
        cudaAvailable = true;
    }
}

GPUDetector::~GPUDetector() {
    if (cudaAvailable) {
        cudaDeviceReset();
    }
}

bool GPUDetector::isCUDAAvailable() const {
    return cudaAvailable;
}

bool GPUDetector::scanGPUs() {
    if (!cudaAvailable) {
        return false;
    }
    
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    gpuList.clear();
    
    for (int i = 0; i < deviceCount; i++) {
        GPUInfo gpu;
        gpu.deviceId = i;
        
        // Get device properties
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error != cudaSuccess) {
            continue;
        }
        
        // Basic info
        gpu.name = prop.name;
        gpu.totalMemory = prop.totalGlobalMem;
        gpu.computeCapabilityMajor = prop.major;
        gpu.computeCapabilityMinor = prop.minor;
        gpu.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        gpu.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        gpu.multiProcessorCount = prop.multiProcessorCount;
        gpu.warpSize = prop.warpSize;
        
        // Grid and block sizes
        for (int j = 0; j < 3; j++) {
            gpu.maxGridSize[j] = prop.maxGridSize[j];
            gpu.maxBlockSize[j] = prop.maxThreadsDim[j];
        }
        
        // Feature support
        gpu.supportsDoublePrecision = checkDoublePrecisionSupport(i);
        gpu.supportsWarpShuffle = checkWarpShuffleSupport(i);
        gpu.supportsSharedMemory = checkSharedMemorySupport(i);
        
        gpuList.push_back(gpu);
        printGPUInfo(gpu);
    }
    
    if (!gpuList.empty()) {
        selectOptimalGPU();
        return true;
    }
    
    return false;
}

const std::vector<GPUInfo>& GPUDetector::getGPUList() const {
    return gpuList;
}

const GPUInfo& GPUDetector::getSelectedGPU() const {
    static GPUInfo empty;
    if (selectedDevice >= 0 && selectedDevice < gpuList.size()) {
        return gpuList[selectedDevice];
    }
    return empty;
}

int GPUDetector::getSelectedDeviceId() const {
    return selectedDevice;
}

void GPUDetector::selectOptimalGPU() {
    if (gpuList.empty()) {
        selectedDevice = -1;
        return;
    }
    
    // Select GPU with highest compute capability and most memory
    int bestDevice = 0;
    int bestScore = 0;
    
    for (size_t i = 0; i < gpuList.size(); i++) {
        const GPUInfo& gpu = gpuList[i];
        
        // Calculate score based on compute capability and memory
        int score = gpu.computeCapabilityMajor * 100 + gpu.computeCapabilityMinor;
        score += (gpu.totalMemory / (1024 * 1024 * 1024)) * 10; // Add memory score
        
        if (score > bestScore) {
            bestScore = score;
            bestDevice = i;
        }
    }
    
    selectedDevice = bestDevice;
    
    // Set the selected device
    cudaSetDevice(gpuList[selectedDevice].deviceId);
    
    std::cout << "Selected GPU: " << gpuList[selectedDevice].name << std::endl;
}

void GPUDetector::setOptimalGridBlockSize(int totalPixels, int& gridSize, int& blockSize) {
    if (selectedDevice < 0 || selectedDevice >= gpuList.size()) {
        // Fallback values
        gridSize = (totalPixels + 255) / 256;
        blockSize = 256;
        return;
    }
    
    const GPUInfo& gpu = gpuList[selectedDevice];
    
    // Optimal block size based on GPU capabilities
    blockSize = std::min(256, gpu.maxThreadsPerBlock);
    
    // Ensure block size is a multiple of warp size
    if (blockSize % gpu.warpSize != 0) {
        blockSize = (blockSize / gpu.warpSize) * gpu.warpSize;
    }
    
    // Calculate grid size
    gridSize = (totalPixels + blockSize - 1) / blockSize;
    
    // Ensure grid size doesn't exceed limits
    if (gridSize > gpu.maxGridSize[0]) {
        gridSize = gpu.maxGridSize[0];
        blockSize = (totalPixels + gridSize - 1) / gridSize;
    }
}

std::string GPUDetector::getLastError() const {
    cudaError_t error = cudaGetLastError();
    return cudaGetErrorString(error);
}

void GPUDetector::printGPUInfo(const GPUInfo& gpu) {
    std::cout << "=== GPU " << gpu.deviceId << " ===" << std::endl;
    std::cout << "Name: " << gpu.name << std::endl;
    std::cout << "Memory: " << (gpu.totalMemory / (1024 * 1024 * 1024)) << " GB" << std::endl;
    std::cout << "Compute Capability: " << gpu.computeCapabilityMajor << "." << gpu.computeCapabilityMinor << std::endl;
    std::cout << "Max Threads per Block: " << gpu.maxThreadsPerBlock << std::endl;
    std::cout << "Multi-Processors: " << gpu.multiProcessorCount << std::endl;
    std::cout << "Warp Size: " << gpu.warpSize << std::endl;
    std::cout << "Double Precision: " << (gpu.supportsDoublePrecision ? "Yes" : "No") << std::endl;
    std::cout << "Warp Shuffle: " << (gpu.supportsWarpShuffle ? "Yes" : "No") << std::endl;
    std::cout << "Shared Memory: " << (gpu.supportsSharedMemory ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

bool GPUDetector::checkDoublePrecisionSupport(int deviceId) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    if (error != cudaSuccess) {
        return false;
    }
    
    return (prop.major >= 2) || (prop.major == 1 && prop.minor >= 3);
}

bool GPUDetector::checkWarpShuffleSupport(int deviceId) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    if (error != cudaSuccess) {
        return false;
    }
    
    return prop.major >= 3;
}

bool GPUDetector::checkSharedMemorySupport(int deviceId) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    if (error != cudaSuccess) {
        return false;
    }
    
    return prop.sharedMemPerBlock > 0;
} 