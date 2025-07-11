# Mandelbrot CUDA with SFML Rendering

This project now includes interactive graphics rendering using SFML (Simple and Fast Multimedia Library).

## Features

- **Interactive Graphics**: Real-time Mandelbrot set visualization
- **Mouse Controls**: 
  - Left-click and drag to pan
  - Mouse wheel to zoom in/out
- **GPU Acceleration**: CUDA-powered computation with CPU fallback
- **Smooth Color Palette**: Beautiful color gradients for the Mandelbrot set
- **Performance Display**: Real-time computation timing

## Setup Instructions

### 1. Install SFML

Run the SFML installation script:
```powershell
.\install_sfml.ps1
```

This will:
- Download SFML 2.6.1 for Windows
- Extract it to the `SFML/` directory
- Set up CMake configuration

### 2. Build the Project

```powershell
.\build.ps1
```

### 3. Run the Program

```powershell
.\build\Release\mandelbrot_cuda.exe
```

## Controls

- **Mouse Drag**: Pan around the Mandelbrot set
- **Mouse Wheel**: Zoom in/out (centered on mouse position)
- **ESC Key**: Exit the program
- **Preset Selection**: Choose from predefined Mandelbrot regions

## Preset Locations

1. **Full Set**: Complete Mandelbrot set overview
2. **Detail 1**: Deep zoom into a spiral region
3. **Detail 2**: Another interesting detail area
4. **Detail 3**: Edge detail exploration
5. **Detail 4**: Fine structure examination
6. **Detail 5**: Another zoomed region

## Technical Details

### Rendering Architecture
- **SFML Graphics**: Hardware-accelerated 2D rendering
- **Texture-based**: Efficient pixel updates using SFML textures
- **Event-driven**: Responsive mouse and keyboard input
- **Real-time**: 60 FPS rendering with smooth interactions

### Color System
- **Smooth Gradients**: Trigonometric color interpolation
- **256-color Palette**: Optimized for performance and beauty
- **Dynamic Scaling**: Colors adapt to iteration count

### Performance
- **GPU Computation**: CUDA kernels for fast Mandelbrot calculation
- **Efficient Rendering**: Minimal CPU overhead for graphics
- **Memory Optimized**: Direct texture updates without unnecessary copies

## Troubleshooting

### SFML Not Found
If you get "SFML not found" errors:
1. Run `.\install_sfml.ps1` to install SFML
2. Make sure the `SFML/` directory exists in your project

### Missing DLLs
If you get DLL errors when running:
1. Copy all `.dll` files from `SFML/bin/` to your executable directory
2. Or add `SFML/bin/` to your system PATH

### Build Errors
If CMake can't find SFML:
1. Delete the `build/` directory
2. Run `.\install_sfml.ps1` again
3. Run `.\build.ps1`

## Development

### Adding New Features
- **New Color Schemes**: Modify `setupColorPalette()` in `renderer_sfml.cpp`
- **Additional Controls**: Add new event handlers in `handleEvents()`
- **UI Elements**: Use SFML's text and shape rendering in `drawUI()`

### Performance Optimization
- **Texture Updates**: Only update when data changes
- **Memory Management**: Efficient pixel data handling
- **GPU Synchronization**: Proper CUDA-Graphics coordination

## File Structure

```
MandelbrotCUDA/
├── renderer_sfml.h          # SFML renderer header
├── renderer_sfml.cpp        # SFML renderer implementation
├── install_sfml.ps1         # SFML installation script
├── CMakeLists.txt           # Build configuration
└── SFML/                    # Local SFML installation
    ├── include/             # SFML headers
    ├── lib/                 # SFML libraries
    └── bin/                 # SFML DLLs
```

## Next Steps

Potential enhancements:
1. **Save Images**: Add screenshot functionality
2. **More Color Schemes**: Implement different color palettes
3. **Animation**: Smooth transitions between zoom levels
4. **Performance Metrics**: Real-time FPS and computation stats
5. **Custom Shaders**: GPU-based color processing 