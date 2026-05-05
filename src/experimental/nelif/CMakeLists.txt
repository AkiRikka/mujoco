# Copyright 2026

cmake_minimum_required(VERSION 3.16)

set(MUJOCO_NELIF_TARGET_NAME mujoco_nelif)
get_filename_component(MUJOCO_NELIF_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

find_package(ZLIB REQUIRED)

add_library(${MUJOCO_NELIF_TARGET_NAME} STATIC)

target_sources(${MUJOCO_NELIF_TARGET_NAME}
    PUBLIC
        exr_writer.h
        gbuffer_pass.h
        input_contract.h
        screen_space_light_pass.h
    PRIVATE
        exr_writer.cc
        gbuffer_pass.cc
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

if(APPLE)
  target_link_libraries(${MUJOCO_NELIF_TARGET_NAME} PUBLIC "-framework OpenGL")
elseif(UNIX)
  target_link_libraries(${MUJOCO_NELIF_TARGET_NAME} PUBLIC GL)
endif()

add_library(mujoco::nelif ALIAS ${MUJOCO_NELIF_TARGET_NAME})
