# Simple CMake-based build script for CUDA project
Write-Host "=== Building Mandelbrot CUDA Project with CMake ===" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "CMakeLists.txt")) {
    Write-Host "ERROR: CMakeLists.txt not found. Please run this script from the project directory." -ForegroundColor Red
    exit 1
}

# Check if CMake is available
$cmakePath = "C:\Program Files\CMake\bin\cmake.exe"
if (Test-Path $cmakePath) {
    Write-Host "CMake found at: $cmakePath" -ForegroundColor Green
    $cmakeVersion = & $cmakePath --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "CMake version: $cmakeVersion" -ForegroundColor Green
    }
} else {
    Write-Host "ERROR: CMake not found at expected location: $cmakePath" -ForegroundColor Red
    Write-Host "Please install CMake or add it to your PATH" -ForegroundColor Yellow
    Write-Host "Download from: https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

# Check if CUDA is available
try {
    $cudaVersion = nvcc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "CUDA found: $cudaVersion" -ForegroundColor Green
    } else {
        throw "CUDA not found"
    }
} catch {
    Write-Host "ERROR: CUDA not found. Please install CUDA Toolkit." -ForegroundColor Red
    Write-Host "Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    exit 1
}

# Create build directory
$buildDir = "build"
if (-not (Test-Path $buildDir)) {
    Write-Host "Creating build directory..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Configure with CMake
Write-Host "Configuring project with CMake..." -ForegroundColor Cyan
Push-Location $buildDir
try {
    $configResult = & $cmakePath .. 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed:" -ForegroundColor Red
        Write-Host $configResult -ForegroundColor Gray
        exit 1
    }
    Write-Host "CMake configuration successful" -ForegroundColor Green
} finally {
    Pop-Location
}

# Build the project
Write-Host "Building project..." -ForegroundColor Cyan
Push-Location $buildDir
try {
    $buildResult = & $cmakePath --build . --config Release 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed:" -ForegroundColor Red
        Write-Host $buildResult -ForegroundColor Gray
        exit 1
    }
    Write-Host "Build successful!" -ForegroundColor Green
} finally {
    Pop-Location
}

# Check if executable was created
$exePath = Join-Path $buildDir "Release\mandelbrot_cuda.exe"
if (Test-Path $exePath) {
    Write-Host ""
    Write-Host "=== Build completed successfully! ===" -ForegroundColor Green
    Write-Host "Executable: $exePath" -ForegroundColor White
    Write-Host ""
    Write-Host "To run: .\build\Release\mandelbrot_cuda.exe" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "ERROR: Executable not found at expected location" -ForegroundColor Red
    exit 1
}