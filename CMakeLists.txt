cmake_minimum_required(VERSION 3.19)
project(FA2026)

# Set the directories
set(AUDIO_PROCESSORS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/engine)

# Find required packages
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(nlohmann_json REQUIRED)

# Export compile config
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Automatically include all .cpp files from engine directory
file(GLOB ENGINE_FILES "${AUDIO_PROCESSORS_DIR}/*.cpp")

# Automatically include all .cpp files from utility directory
file(GLOB UTILITY_FILES "${UTILITY_DIR}/*.cpp")

# Automatically include all .cpp files from dsp directory
file(GLOB DSP_FILES "${DSP_DIR}/*.cpp")

# Function to create executable from a cpp file
function(add_executable_from_source SOURCE_FILE)
  # Get the filename without extension
  get_filename_component(EXEC_NAME ${SOURCE_FILE} NAME_WE)
  
  # Create executable
  add_executable(${EXEC_NAME} ${SOURCE_FILE} ${ENGINE_FILES} ${UTILITY_FILES} ${DSP_FILES})
  
  # Set include directories
  target_include_directories(${EXEC_NAME} PRIVATE 
    ${AUDIO_PROCESSORS_DIR}
    ${UTILITY_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${DSP_DIR}
  )
  
  # Link libraries
  target_link_libraries(${EXEC_NAME} PUBLIC 
    Eigen3::Eigen 
    ${HDF5_LIBRARIES} 
    nlohmann_json::nlohmann_json
  )
  
  # Set C++ standard
  target_compile_features(${EXEC_NAME} PRIVATE cxx_std_17)
endfunction()


# Automatically create executables for all .cpp files in src/mains/
file(GLOB MAIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/mains/*.cpp")
foreach(MAIN_SOURCE ${MAIN_SOURCES})
  add_executable_from_source(${MAIN_SOURCE})
endforeach()

# Automatically create executables for all .cpp files in src/test/
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
  add_executable_from_source(${TEST_SOURCE})
endforeach()
