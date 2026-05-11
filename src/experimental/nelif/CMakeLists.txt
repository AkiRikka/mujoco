# Copyright 2026

cmake_minimum_required(VERSION 3.16)

set(MUJOCO_NELIF_TARGET_NAME mujoco_nelif)
get_filename_component(MUJOCO_NELIF_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

find_package(ZLIB REQUIRED)

option(MUJOCO_NELIF_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend for NeLiF" OFF)
set(MUJOCO_NELIF_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime C/C++ SDK root")

add_library(${MUJOCO_NELIF_TARGET_NAME} STATIC)

target_sources(${MUJOCO_NELIF_TARGET_NAME}
    PUBLIC
        exr_writer.h
        gbuffer_pass.h
        input_contract.h
        onnx_backend.h
        runtime_pass.h
        screen_space_light_pass.h
    PRIVATE
        exr_writer.cc
        gbuffer_pass.cc
        onnx_backend.cc
        runtime_pass.cc
        screen_space_light_pass.cc
)

target_include_directories(${MUJOCO_NELIF_TARGET_NAME}
    PUBLIC
        ${MUJOCO_NELIF_SOURCE_DIR}/include
        ${MUJOCO_NELIF_SOURCE_DIR}/src
)

target_link_libraries(${MUJOCO_NELIF_TARGET_NAME}
    PUBLIC
        mujoco::mujoco
        glfw
        ZLIB::ZLIB
)

if(MUJOCO_NELIF_ENABLE_ONNXRUNTIME)
  find_path(
      MUJOCO_NELIF_ONNXRUNTIME_INCLUDE_DIR
      onnxruntime_cxx_api.h
      HINTS
          ${MUJOCO_NELIF_ONNXRUNTIME_ROOT}/include
          ${MUJOCO_NELIF_ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
  )
  find_library(
      MUJOCO_NELIF_ONNXRUNTIME_LIBRARY
      NAMES onnxruntime libonnxruntime
      HINTS
          ${MUJOCO_NELIF_ONNXRUNTIME_ROOT}/lib
          ${MUJOCO_NELIF_ONNXRUNTIME_ROOT}/lib64
          ${MUJOCO_NELIF_ONNXRUNTIME_ROOT}/capi
  )
  if(NOT MUJOCO_NELIF_ONNXRUNTIME_INCLUDE_DIR OR NOT MUJOCO_NELIF_ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime C/C++ SDK was not found. Set "
        "-DMUJOCO_NELIF_ONNXRUNTIME_ROOT=/path/to/onnxruntime, or disable "
        "MUJOCO_NELIF_ENABLE_ONNXRUNTIME.")
  endif()
  target_compile_definitions(${MUJOCO_NELIF_TARGET_NAME}
      PUBLIC MUJOCO_NELIF_HAS_ONNXRUNTIME=1)
  target_include_directories(${MUJOCO_NELIF_TARGET_NAME}
      PUBLIC ${MUJOCO_NELIF_ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(${MUJOCO_NELIF_TARGET_NAME}
      PUBLIC ${MUJOCO_NELIF_ONNXRUNTIME_LIBRARY})
else()
  target_compile_definitions(${MUJOCO_NELIF_TARGET_NAME}
      PUBLIC MUJOCO_NELIF_HAS_ONNXRUNTIME=0)
endif()

if(APPLE)
  target_link_libraries(${MUJOCO_NELIF_TARGET_NAME} PUBLIC "-framework OpenGL")
elseif(UNIX)
  target_link_libraries(${MUJOCO_NELIF_TARGET_NAME} PUBLIC GL)
endif()

add_library(mujoco::nelif ALIAS ${MUJOCO_NELIF_TARGET_NAME})
