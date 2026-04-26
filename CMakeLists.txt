cmake_minimum_required(VERSION 3.16)
project(wann_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Dependency: nlohmann/json (header-only)
#   Prefers the system package; falls back to FetchContent if not found.
#   On Arch Linux: sudo pacman -S nlohmann-json
#   On Ubuntu/Debian: sudo apt install nlohmann-json3-dev
# ---------------------------------------------------------------------------
find_package(nlohmann_json 3.2.0 QUIET)

if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found in system, fetching via FetchContent...")
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
  )
  set(JSON_BuildTests      OFF CACHE BOOL "" FORCE)
  set(JSON_Install         OFF CACHE BOOL "" FORCE)
  set(JSON_MultipleHeaders OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
else()
  message(STATUS "Found system nlohmann_json: ${nlohmann_json_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Library: core WANN algorithm
# ---------------------------------------------------------------------------
add_library(wann_lib STATIC
  src/Ind.cpp
  src/NsgaSort.cpp
  src/Wann.cpp
  src/DataGatherer.cpp
)

target_include_directories(wann_lib PUBLIC include)
target_link_libraries(wann_lib PUBLIC nlohmann_json::nlohmann_json)

target_compile_options(wann_lib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# ---------------------------------------------------------------------------
# Executable: wann_train
# ---------------------------------------------------------------------------
add_executable(wann_train src/main.cpp)
target_link_libraries(wann_train PRIVATE wann_lib)
