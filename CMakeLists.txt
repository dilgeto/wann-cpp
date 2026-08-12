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
# OpenMP – parallelise population evaluation
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
target_link_libraries(wann_lib PUBLIC OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_train  (standalone, no task)
# ---------------------------------------------------------------------------
add_executable(wann_train src/main.cpp)
target_link_libraries(wann_train PRIVATE wann_lib)

# ---------------------------------------------------------------------------
# SNN Simulator – link against pre-built static libraries.
# Build the simulator first:  cd ../snn-simulator && ./build.sh
# ---------------------------------------------------------------------------
set(SNN_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/../snn-simulator)
set(SNN_BUILD ${SNN_DIR}/build_cmake)

add_library(snn_core_imp STATIC IMPORTED GLOBAL)
set_target_properties(snn_core_imp PROPERTIES
    IMPORTED_LOCATION             ${SNN_BUILD}/libsnn_core.a
    INTERFACE_INCLUDE_DIRECTORIES ${SNN_DIR}/include)

add_library(snn_encoding_imp STATIC IMPORTED GLOBAL)
set_target_properties(snn_encoding_imp PROPERTIES
    IMPORTED_LOCATION             ${SNN_BUILD}/libsnn_encoding.a
    INTERFACE_INCLUDE_DIRECTORIES ${SNN_DIR}/include)

add_library(snn_engine_imp STATIC IMPORTED GLOBAL)
set_target_properties(snn_engine_imp PROPERTIES
    IMPORTED_LOCATION             ${SNN_BUILD}/libsnn_engine.a
    INTERFACE_INCLUDE_DIRECTORIES ${SNN_DIR}/include)

# rl-tools is header-only (bundled inside snn-simulator/external/)
add_library(rl_tools_iface INTERFACE)
target_include_directories(rl_tools_iface INTERFACE
    ${SNN_DIR}/external/rl-tools/include)

# ---------------------------------------------------------------------------
# Executable: wann_snn – WANN + SNN simulator + rl-tools Pendulum
# ---------------------------------------------------------------------------
add_executable(wann_snn
    src/main_snn.cpp
    src/SnnPendulumTask.cpp)

target_include_directories(wann_snn PRIVATE include)

# Link order: engine depends on core + encoding, list them after engine so the
# linker can resolve symbols in a single left-to-right pass.
target_link_libraries(wann_snn PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_acrobot – WANN + SNN simulator + rl-tools Acrobot
# ---------------------------------------------------------------------------
add_executable(wann_acrobot
    src/main_acrobot.cpp
    src/SnnAcrobotTask.cpp)

target_include_directories(wann_acrobot PRIVATE include)

target_link_libraries(wann_acrobot PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_car – WANN + SNN simulator + rl-tools CarTrack
# ---------------------------------------------------------------------------
add_executable(wann_car
    src/main_car.cpp
    src/SnnCarTask.cpp)

target_include_directories(wann_car PRIVATE include)

target_link_libraries(wann_car PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_mountain_car – WANN + SNN simulator + rl-tools MountainCar
# ---------------------------------------------------------------------------
add_executable(wann_mountain_car
    src/main_mountain_car.cpp
    src/SnnMountainCarTask.cpp)

target_include_directories(wann_mountain_car PRIVATE include)

target_link_libraries(wann_mountain_car PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_l2f – WANN + SNN simulator + rl-tools L2F (Crazyflie quadrotor)
# ---------------------------------------------------------------------------
add_executable(wann_l2f
    src/main_l2f.cpp
    src/SnnL2FTask.cpp)

target_include_directories(wann_l2f PRIVATE include)

target_link_libraries(wann_l2f PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_disc_mc – WANN + SNN simulator + Gymnasium MountainCar-v0 (discrete)
# ---------------------------------------------------------------------------
add_executable(wann_disc_mc
    src/main_disc_mc.cpp
    src/SnnDiscMCTask.cpp)

target_include_directories(wann_disc_mc PRIVATE include)

target_link_libraries(wann_disc_mc PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executable: wann_car_replay – corre la mejor red y exporta trayectoria CSV
# ---------------------------------------------------------------------------
add_executable(wann_car_replay
    src/main_replay.cpp
    src/SnnCarTask.cpp)

target_include_directories(wann_car_replay PRIVATE include)

target_link_libraries(wann_car_replay PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface)

# ---------------------------------------------------------------------------
# Executable: wann_acrobot_eval – evalúa N episodios con la mejor red Acrobot
# ---------------------------------------------------------------------------
add_executable(wann_acrobot_eval
    src/main_eval_acrobot.cpp
    src/SnnAcrobotTask.cpp)

target_include_directories(wann_acrobot_eval PRIVATE include)

target_link_libraries(wann_acrobot_eval PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface)

# ---------------------------------------------------------------------------
# Executable: wann_mountain_car_eval – evalúa N episodios con la mejor red MountainCar
# ---------------------------------------------------------------------------
add_executable(wann_mountain_car_eval
    src/main_eval_mountain_car.cpp
    src/SnnMountainCarTask.cpp)

target_include_directories(wann_mountain_car_eval PRIVATE include)

target_link_libraries(wann_mountain_car_eval PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface)

# ---------------------------------------------------------------------------
# Executable: wann_l2f_eval – evalúa N episodios con la mejor red L2F
# ---------------------------------------------------------------------------
add_executable(wann_l2f_eval
    src/main_eval_l2f.cpp
    src/SnnL2FTask.cpp)

target_include_directories(wann_l2f_eval PRIVATE include)

target_link_libraries(wann_l2f_eval PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface)

# ---------------------------------------------------------------------------
# Executable: wann_car_eval – evalúa N episodios con la mejor red Car
# ---------------------------------------------------------------------------
add_executable(wann_car_eval
    src/main_eval_car.cpp
    src/SnnCarTask.cpp)

target_include_directories(wann_car_eval PRIVATE include)

target_link_libraries(wann_car_eval PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface)

# ---------------------------------------------------------------------------
# Executable: wann_disc_mc_eval – evalúa N episodios con la mejor red MountainCar discreto
# ---------------------------------------------------------------------------
add_executable(wann_disc_mc_eval
    src/main_eval_disc_mc.cpp
    src/SnnDiscMCTask.cpp)

target_include_directories(wann_disc_mc_eval PRIVATE include)

target_link_libraries(wann_disc_mc_eval PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Executables: wann_eval_weights_{acrobot,car,disc_mc} – evalúan
# getDistFitness() (los N pesos compartidos) de una red guardada para una
# lista de seeds. Un binario por tarea: cada TaskXxx.cpp instancia símbolos
# no-inline de rl-tools que producen "multiple definition" si se linkean dos
# tareas en el mismo binario, así que la tarea se fija en tiempo de
# compilación vía EVAL_TASK_*.
# ---------------------------------------------------------------------------
add_executable(wann_eval_weights_acrobot
    src/main_eval_weights.cpp
    src/SnnAcrobotTask.cpp)
target_compile_definitions(wann_eval_weights_acrobot PRIVATE EVAL_TASK_ACROBOT)
target_include_directories(wann_eval_weights_acrobot PRIVATE include)
target_link_libraries(wann_eval_weights_acrobot PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

add_executable(wann_eval_weights_car
    src/main_eval_weights.cpp
    src/SnnCarTask.cpp)
target_compile_definitions(wann_eval_weights_car PRIVATE EVAL_TASK_CAR)
target_include_directories(wann_eval_weights_car PRIVATE include)
target_link_libraries(wann_eval_weights_car PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)

add_executable(wann_eval_weights_disc_mc
    src/main_eval_weights.cpp
    src/SnnDiscMCTask.cpp)
target_compile_definitions(wann_eval_weights_disc_mc PRIVATE EVAL_TASK_DISC_MC)
target_include_directories(wann_eval_weights_disc_mc PRIVATE include)
target_link_libraries(wann_eval_weights_disc_mc PRIVATE
    wann_lib
    snn_engine_imp
    snn_encoding_imp
    snn_core_imp
    rl_tools_iface
    OpenMP::OpenMP_CXX)
