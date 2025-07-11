#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

// CUDA kernel for Mandelbrot computation
__global__ void mandelbrot_kernel(
    int* results,
    double realMin, double realMax,
    double imagMin, double imagMax,
    int width, int height, int maxIterations
) {
    // Calculate global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate coordinates for this pixel
    double real = realMin + (realMax - realMin) * x / (width - 1);
    double imag = imagMax - (imagMax - imagMin) * y / (height - 1);
    
    // Mandelbrot computation
    double x0 = real;
    double y0 = imag;
    
    double zx = 0.0;
    double zy = 0.0;
    int iteration = 0;
    
    // Optimized escape condition: use squared distance to avoid sqrt
    double escapeRadiusSquared = 4.0;
    
    while (iteration < maxIterations && (zx * zx + zy * zy) <= escapeRadiusSquared) {
        double zxTemp = zx * zx - zy * zy + x0;
        zy = 2.0 * zx * zy + y0;
        zx = zxTemp;
        iteration++;
    }
    
    // Store result
    results[y * width + x] = iteration;
}

// Optimized kernel with early exit optimizations
__global__ void mandelbrot_kernel_optimized(
    int* results,
    double realMin, double realMax,
    double imagMin, double imagMax,
    int width, int height, int maxIterations
) {
    // Calculate global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate coordinates for this pixel
    double real = realMin + (realMax - realMin) * x / (width - 1);
    double imag = imagMax - (imagMax - imagMin) * y / (height - 1);
    
    // Mandelbrot computation with early exit optimization
    double x0 = real;
    double y0 = imag;
    
    double zx = 0.0;
    double zy = 0.0;
    int iteration = 0;
    
    // Early exit optimization: check if point is in main cardioid
    double q = (x0 - 0.25) * (x0 - 0.25) + y0 * y0;
    if (q * (q + (x0 - 0.25)) <= 0.25 * y0 * y0) {
        iteration = maxIterations; // Point is in main cardioid
    } else {
        // Check if point is in period-2 bulb
        if ((x0 + 1.0) * (x0 + 1.0) + y0 * y0 <= 0.0625) {
            iteration = maxIterations; // Point is in period-2 bulb
        } else {
            // Full computation
            double escapeRadiusSquared = 4.0;
            
            while (iteration < maxIterations && (zx * zx + zy * zy) <= escapeRadiusSquared) {
                double zxTemp = zx * zx - zy * zy + x0;
                zy = 2.0 * zx * zy + y0;
                zx = zxTemp;
                iteration++;
            }
        }
    }
    
    // Store result with coalesced memory access
    results[y * width + x] = iteration;
}

// Single precision version for better performance on older GPUs
__global__ void mandelbrot_kernel_float(
    int* results,
    float realMin, float realMax,
    float imagMin, float imagMax,
    int width, int height, int maxIterations
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    float real = realMin + (realMax - realMin) * x / (width - 1);
    float imag = imagMax - (imagMax - imagMin) * y / (height - 1);
    
    float x0 = real;
    float y0 = imag;
    
    float zx = 0.0f;
    float zy = 0.0f;
    int iteration = 0;
    
    float escapeRadiusSquared = 4.0f;
    
    while (iteration < maxIterations && (zx * zx + zy * zy) <= escapeRadiusSquared) {
        float zxTemp = zx * zx - zy * zy + x0;
        zy = 2.0f * zx * zy + y0;
        zx = zxTemp;
        iteration++;
    }
    
    results[y * width + x] = iteration;
}

// Wrapper function to call the appropriate kernel based on GPU capabilities
extern "C" cudaError_t compute_mandelbrot_cuda(
    int* results,
    double realMin, double realMax,
    double imagMin, double imagMax,
    int width, int height, int maxIterations,
    bool useOptimizedKernel,
    bool useFloatPrecision
) {
    // Calculate grid and block dimensions
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Allocate device memory
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaError_t error = cudaMalloc(&d_results, size);
    if (error != cudaSuccess) {
        return error;
    }

    // Register host memory as pinned so we can copy asynchronously
    error = cudaHostRegister(results, size, cudaHostRegisterPortable);
    if (error != cudaSuccess) {
        cudaFree(d_results);
        return error;
    }

    // Create a stream for concurrent execution
    cudaStream_t stream;
    error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        cudaHostUnregister(results);
        cudaFree(d_results);
        return error;
    }
 
    // Launch appropriate kernel
    if (useFloatPrecision) {
        mandelbrot_kernel_float<<<gridSize, blockSize, 0, stream>>>(
            d_results,
            (float)realMin, (float)realMax,
            (float)imagMin, (float)imagMax,
            width, height, maxIterations
        );
    } else if (useOptimizedKernel) {
        mandelbrot_kernel_optimized<<<gridSize, blockSize, 0, stream>>>(
            d_results,
            realMin, realMax,
            imagMin, imagMax,
            width, height, maxIterations
        );
    } else {
        mandelbrot_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_results,
            realMin, realMax,
            imagMin, imagMax,
            width, height, maxIterations
        );
    }
 
    // Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaStreamDestroy(stream);
        cudaHostUnregister(results);
        cudaFree(d_results);
        return error;
    }
 
    // Copy results back to host
    error = cudaMemcpyAsync(results, d_results, size, cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) {
        cudaStreamDestroy(stream);
        cudaHostUnregister(results);
        cudaFree(d_results);
        return error;
    }

    // Wait for all operations on the stream to finish
    cudaStreamSynchronize(stream);
 
    // Clean up
    cudaStreamDestroy(stream);
    cudaHostUnregister(results);
    cudaFree(d_results);
 
    return error;
}

// Error checking helper
extern "C" void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(error));
    }
} 