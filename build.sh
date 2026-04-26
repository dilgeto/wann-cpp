#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
BUILD_TYPE="${1:-Release}"   # Release | Debug

echo "==> Configurando (${BUILD_TYPE})..."
cmake -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -G "Unix Makefiles"

echo "==> Compilando con $(nproc) hilos..."
cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo "==> Listo: ${BUILD_DIR}/wann_train"
