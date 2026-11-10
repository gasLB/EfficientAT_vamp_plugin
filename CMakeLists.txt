cmake_minimum_required(VERSION 3.19)

# Project Options
option(EFFICIENTAT_VAMP_PLUGIN_ENABLE_DEBUG "Enables the debug scheme using the Partiels application" ON)
option(EFFICIENTAT_VAMP_PLUGIN_REFRESH_MODEL "Force re-download and re-export of the EfficientAT model during CMake configure" OFF)
option(EFFICIENTAT_VAMP_PLUGIN_BUILD_UNIT_TESTS "Build the standalone unit_test_runner executable" OFF)

set(EFFICIENTAT_VAMP_PLUGIN_TESTER_ENABLE ON CACHE STRING "Enables the tests with the vamp plugin tester")
set(EFFICIENTAT_VAMP_PLUGIN_TESTER_TIMEOUT 600 CACHE STRING "The timeout in seconds of the test with the vamp plugin tester (zero for no timeout)")

### Sccache ###
set(EFFICIENTAT_VAMP_PLUGIN_USE_SCCACHE OFF CACHE BOOL "Uses sccache as the compiler launcher")
if(EFFICIENTAT_VAMP_PLUGIN_USE_SCCACHE)
  if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
		find_program(EFFICIENTAT_VAMP_PLUGIN_SCCACHE_PROGRAM sccache)
		if(EFFICIENTAT_VAMP_PLUGIN_SCCACHE_PROGRAM)
			set(CMAKE_C_COMPILER_LAUNCHER "${EFFICIENTAT_VAMP_PLUGIN_SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
			set(CMAKE_CXX_COMPILER_LAUNCHER "${EFFICIENTAT_VAMP_PLUGIN_SCCACHE_PROGRAM}" CACHE STRING "C++ compiler launcher" FORCE)
			set(EFFICIENTAT_VAMP_PLUGIN_SCCACHE_LAUNCHER_MANAGED ON CACHE INTERNAL "Whether the Classifier Vamp Plugin enabled the compiler launcher with sccache")
      message(VERBOSE "sccache found, compiler launcher enabled")
    else()
      message(VERBOSE "sccache not found, continuing without compiler cache")
    endif()
  endif()
elseif(EFFICIENTAT_VAMP_PLUGIN_SCCACHE_LAUNCHER_MANAGED)
  unset(CMAKE_C_COMPILER_LAUNCHER CACHE)
  unset(CMAKE_CXX_COMPILER_LAUNCHER CACHE)
	unset(EFFICIENTAT_VAMP_PLUGIN_SCCACHE_LAUNCHER_MANAGED CACHE)
  message(VERBOSE "sccache compiler launcher disabled")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_XCODE_GENERATE_SCHEME ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum OS X deployment version")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "Default value for MSVC_RUNTIME_LIBRARY of targets" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE CACHE BOOL "Default value for POSITION_INDEPENDENT_CODE of targets" FORCE)

### Project ###
project(ClassifierVampPlugin VERSION 1.0.0 LANGUAGES C CXX)

### Include Vamp (IVE) ###
set(IVE_BUILD_HOST_LIBRARY OFF)
set(IVE_BUILD_PLUGIN_LIBRARY ON)
add_subdirectory(ircam-vamp-extension EXCLUDE_FROM_ALL)

### Include LibTorch ###
include(Resources/libtorch.cmake)

### Models ###
execute_process(
  COMMAND ${CMAKE_COMMAND}
    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
	-DREFRESH_MODEL=${EFFICIENTAT_VAMP_PLUGIN_REFRESH_MODEL}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/Resources/download_model.cmake
)

### Version ###
execute_process(COMMAND git log -1 --format=%h WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} OUTPUT_VARIABLE GIT_COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git status --porcelain WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} OUTPUT_VARIABLE GIT_HAS_DIFF OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT GIT_HAS_DIFF STREQUAL "")
  set(GIT_COMMIT_ID "${GIT_COMMIT_ID}-head")
endif()

if(NOT EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG OR EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG STREQUAL "")
	set(EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG ${CMAKE_PROJECT_VERSION})
endif()
message(STATUS "Classifier Vamp Plugin: " ${EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG} " (" ${GIT_COMMIT_ID} ")")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.txt ${EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG})

### Package ###
set(VPP_NAME "Classifier" CACHE PATH "The name of the project" FORCE)
set(VPP_COMPANY "Ircam" CACHE PATH "The name of the company" FORCE)
set(VPP_URL "www.ircam.fr" CACHE PATH "The URL of the project")
# Generate a new UUID for your plugin (do not reuse IPT's UUID to avoid installer collisions)
set(VPP_APPID "8f4b2c10-9d3a-4e7b-a12c-3d4e5f6a7b8c" CACHE PATH "The unique ID of the project")
set(VPP_BUILD_TAG "${EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG}" CACHE STRING "The current tag of the project" FORCE)
set(VPP_ABOUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Resources/About.txt" CACHE PATH "The about file to add to the distribution" FORCE)
set(VPP_CHANGELOG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Resources/ChangeLog.txt" CACHE PATH "The changelog file to add to the distribution" FORCE)
set(VPP_ICON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Resources/Ircam-logo-noir-RS.png" CACHE PATH "The icon file to add to the distribution" FORCE)
set(VPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package CACHE PATH "The directory of the package" FORCE)
include(vamp-plugin-packager/vamp-plugin-packager.cmake)

### Source ###
set(PLUGIN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(EFFICIENTAT_VAMP_PLUGIN_SOURCES
  "${PLUGIN_SOURCE_DIR}/eat_vamp_plugin.cpp"
  "${PLUGIN_SOURCE_DIR}/eat_vamp_plugin.h"
)
source_group("sources" FILES ${EFFICIENTAT_VAMP_PLUGIN_SOURCES})

### Target ###
add_library(eat_vamp_plugin MODULE ${EFFICIENTAT_VAMP_PLUGIN_SOURCES})
ive_prepare_plugin_target(eat_vamp_plugin)
target_compile_definitions(eat_vamp_plugin PRIVATE USE_IVE_EXTENSION)
target_include_directories(eat_vamp_plugin PRIVATE "${PLUGIN_SOURCE_DIR}")
if(UNIX AND NOT APPLE)
	target_link_libraries(eat_vamp_plugin PRIVATE stdc++fs)
endif()

if(EFFICIENTAT_VAMP_PLUGIN_TESTER_ENABLE)
  enable_testing()
endif()

vpp_add_plugin(eat_vamp_plugin
  ${CMAKE_CURRENT_SOURCE_DIR}/source/ircamEfficientAT.cat
  OUTPUT_NAME ircamEfficientAT
  COPYRIGHT "Copyright 2026 Ircam. All rights reserved"
  VERSION ${PROJECT_VERSION_MAJOR}
  INSTALL ON
  DEBUG ${EFFICIENTAT_VAMP_PLUGIN_ENABLE_DEBUG}
  DEBUG_DOCUMENT ${CMAKE_CURRENT_SOURCE_DIR}/Resources/test.ptldoc
  TESTER ${EFFICIENTAT_VAMP_PLUGIN_TESTER_ENABLE}
  TESTER_ARGUMENTS -a -t C1
  TESTER_TIMEOUT ${EFFICIENTAT_VAMP_PLUGIN_TESTER_TIMEOUT}
)

vpp_create_clang_format_targets(eat_vamp_plugin "${EFFICIENTAT_VAMP_PLUGIN_SOURCES}")

### LibTorch libraries and models ###
setup_libtorch(eat_vamp_plugin)

add_custom_command(TARGET eat_vamp_plugin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_BINARY_DIR}/models/"
	"$<TARGET_FILE_DIR:eat_vamp_plugin>/../Resources/models"
)

### Optional: Custom Unit Test Runner ###
if(EFFICIENTAT_VAMP_PLUGIN_BUILD_UNIT_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/source/unit_test_runner.cpp")
  add_executable(unit_test_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/source/unit_test_runner.cpp
	${EFFICIENTAT_VAMP_PLUGIN_SOURCES}
  )
  ive_prepare_plugin_target(unit_test_runner)
  target_compile_definitions(unit_test_runner PRIVATE USE_IVE_EXTENSION)
  target_include_directories(unit_test_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source)
  setup_libtorch(unit_test_runner)
endif()

### Publishing ###
set(PROJECT_URL "https://github.com/Ircam-Partiels/EfficientAT-vamp-plugin")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/EfficientAT-release.md "## Downloads\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/EfficientAT-release.md "- [Classifier-Linux.zip](${PROJECT_URL}/releases/download/${EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG}/Classifier-Linux.zip)\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/EfficientAT-release.md "- [Classifier-MacOS.pkg](${PROJECT_URL}/releases/download/${EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG}/Classifier-MacOS.pkg)\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/EfficientAT-release.md "- [Classifier-Windows.exe](${PROJECT_URL}/releases/download/${EFFICIENTAT_VAMP_PLUGIN_BUILD_TAG}/Classifier-Windows.exe)\n")
