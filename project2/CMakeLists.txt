project(project_1_fys4411)

# Add directories containing header-files
include_directories(
    .
    Math
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Math/*.cpp"
)

set (CMAKE_CXX_FLAGS "-O3 -fopenmp")

# Create executable called "vmc"
add_executable(vmc ${SOURCES})
