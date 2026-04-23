cmake_minimum_required(VERSION 3.20)
project(tracking_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TRACKING_CPP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
  if(TRACKING_CPP_WARNINGS_AS_ERRORS)
    add_compile_options(/WX)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(TRACKING_CPP_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)
find_package(yaml-cpp REQUIRED)
find_package(ONNXRuntime REQUIRED)

add_library(tracking_core
  src/core/config.cpp
  src/core/ort_runtime.cpp
  src/utils/logging.cpp
)

target_include_directories(tracking_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tracking_core
  PUBLIC
    yaml-cpp::yaml-cpp
    ONNXRuntime::ONNXRuntime
)

add_library(tracking_detector
  src/detector/yolo_onnx_detector.cpp
)

target_include_directories(tracking_detector
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracking_detector
  PUBLIC
    tracking_core
    ONNXRuntime::ONNXRuntime
    ${OpenCV_LIBS}
)

add_library(tracking_tracker
  src/tracker/passthrough_tracker.cpp
  src/tracker/botsort_tracker.cpp
)

target_include_directories(tracking_tracker
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracking_tracker
  PUBLIC
    tracking_core
    ONNXRuntime::ONNXRuntime
    ${OpenCV_LIBS}
)

add_executable(tracking_app
  app/main.cpp
)

target_include_directories(tracking_app
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracking_app
  PRIVATE
    tracking_core
    tracking_detector
    tracking_tracker
    ${OpenCV_LIBS}
)

if(WIN32)
  get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set(ONNXRUNTIME_DLL_PATH "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
  file(GLOB ONNXRUNTIME_PROVIDER_DLLS "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers*.dll")
  if(EXISTS "${ONNXRUNTIME_DLL_PATH}")
    add_custom_command(TARGET tracking_app POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${ONNXRUNTIME_DLL_PATH}"
              $<TARGET_FILE_DIR:tracking_app>
      COMMENT "Copying onnxruntime.dll next to tracking_app"
    )
    foreach(_provider_dll IN LISTS ONNXRUNTIME_PROVIDER_DLLS)
      add_custom_command(TARGET tracking_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_provider_dll}"
                $<TARGET_FILE_DIR:tracking_app>
        COMMENT "Copying ONNX Runtime provider DLL: ${_provider_dll}"
      )
    endforeach()
  else()
    message(WARNING "onnxruntime.dll was not found at ${ONNXRUNTIME_DLL_PATH}; runtime may fail if PATH resolves a different ONNX Runtime version")
  endif()
endif()

set_target_properties(tracking_app PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
