cmake_minimum_required(VERSION 3.15)

project(mandelbrot LANGUAGES CXX)

# Toolchain / language
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "clang++")  # optional; drop if you want default compiler

# Build type default (can be overridden from CLI)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build a shared lib by default; set to OFF for a static lib
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)

# ---- Library ----
# If mandle.cpp currently has a main(), split it:
#   - put library code in src/mandelbrot.cpp (no main)
#   - put CLI in src/main.cpp (with main)
# Adjust the sources below accordingly.
add_library(mandelbrot
    mandle.cpp
)

# Public headers (optional): put headers in include/
# target_include_directories makes the include path visible to users of the lib.
target_include_directories(mandelbrot
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Position independent code is automatic for shared; ensure for static too.
set_target_properties(mandelbrot PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# macOS rpath nicety
if(APPLE)
  set_target_properties(mandelbrot PROPERTIES
    MACOSX_RPATH ON
  )
endif()

# ---- Optional CLI (remove this block if you only want the library) ----
# add_executable(mandelbrot_cli main.cpp)
# target_link_libraries(mandelbrot_cli PRIVATE mandelbrot)

# ---- Installation (optional) ----
# install(TARGETS mandelbrot
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         RUNTIME DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)