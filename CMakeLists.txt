cmake_minimum_required(VERSION 3.10)

project(genai_model)
set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_PYTHON_VERSION 3.12)
set(CMAKE_CXX_FLAGS "-O2 -Wall -shared -std=c++2a -fPIC")

find_package(pybind11 REQUIRED)

pybind11_add_module(${PROJECT_NAME} app/genai_model.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})

#g++ -O3 -Wall -shared -std=c++2a -fPIC $(python3.12 -m pybind11 --includes) genai_model.cpp -o genai_model$(python3.12-config --extension-suffix)
#c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) genai_model.cpp -o genai_model$(python3.12-config --extension-suffix)
# add the executable
# add_executable(${PROJECT_NAME} main.cpp)
# target_include_directories( ${PROJECT_NAME} PRIVATE)
# target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
