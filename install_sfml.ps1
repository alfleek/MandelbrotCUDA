# SFML Installation Script for Windows
Write-Host "=== Installing SFML for Mandelbrot CUDA Project ===" -ForegroundColor Green

# SFML version and download URL
$sfmlVersion = "2.6.1"
$sfmlUrl = "https://github.com/SFML/SFML/releases/download/$sfmlVersion/SFML-$sfmlVersion-windows-vc17-64-bit.zip"
$sfmlZip = "SFML-$sfmlVersion-windows-vc17-64-bit.zip"
$sfmlDir = "SFML-$sfmlVersion"

# Check if SFML is already installed
if (Test-Path "SFML") {
    Write-Host "SFML appears to be already installed in the SFML directory." -ForegroundColor Yellow
    Write-Host "If you want to reinstall, please delete the SFML directory first." -ForegroundColor Yellow
    exit 0
}

# Create SFML directory
Write-Host "Creating SFML directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path "SFML" -Force | Out-Null

# Download SFML
Write-Host "Downloading SFML $sfmlVersion..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $sfmlUrl -OutFile $sfmlZip
    Write-Host "Download completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Failed to download SFML: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please download SFML manually from: https://www.sfml-dev.org/download.php" -ForegroundColor Yellow
    exit 1
}

# Extract SFML
Write-Host "Extracting SFML..." -ForegroundColor Cyan
try {
    Expand-Archive -Path $sfmlZip -DestinationPath "SFML" -Force
    Write-Host "Extraction completed!" -ForegroundColor Green
} catch {
    Write-Host "Failed to extract SFML: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Clean up zip file
Remove-Item $sfmlZip -Force

# Create CMake FindSFML.cmake file
Write-Host "Creating CMake configuration..." -ForegroundColor Cyan
$findSfmlContent = @"
# FindSFML.cmake for local SFML installation
set(SFML_DIR "\${CMAKE_CURRENT_SOURCE_DIR}/SFML/lib/cmake/SFML")

# Set SFML paths
set(SFML_INCLUDE_DIR "\${CMAKE_CURRENT_SOURCE_DIR}/SFML/include")
set(SFML_LIBRARY_DIR "\${CMAKE_CURRENT_SOURCE_DIR}/SFML/lib")

# Find SFML components
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

if(SFML_FOUND)
    message(STATUS "SFML found: \${SFML_VERSION}")
    message(STATUS "SFML include dir: \${SFML_INCLUDE_DIR}")
    message(STATUS "SFML library dir: \${SFML_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "SFML not found!")
endif()
"@

$findSfmlContent | Out-File -FilePath "FindSFML.cmake" -Encoding UTF8

Write-Host ""
Write-Host "=== SFML Installation Complete! ===" -ForegroundColor Green
Write-Host "SFML has been installed in the SFML directory." -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Copy SFML DLLs to your executable directory or add SFML/bin to PATH" -ForegroundColor White
Write-Host "2. Run the build script: .\build.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Note: You may need to copy SFML DLLs from SFML/bin to your build directory" -ForegroundColor Yellow 