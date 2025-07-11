# FindSFML.cmake for local SFML installation
set(SFML_DIR "\/SFML/lib/cmake/SFML")

# Set SFML paths
set(SFML_INCLUDE_DIR "\/SFML/include")
set(SFML_LIBRARY_DIR "\/SFML/lib")

# Find SFML components
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

if(SFML_FOUND)
    message(STATUS "SFML found: \")
    message(STATUS "SFML include dir: \")
    message(STATUS "SFML library dir: \")
else()
    message(FATAL_ERROR "SFML not found!")
endif()
