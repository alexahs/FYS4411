project(project_1_fys4411)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    InitialStates
    Math
    WaveFunctions
    InitialStates
    Misc
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "Math/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
    GLOB SOURCES "Misc/*.cpp"
)

set (CMAKE_CXX_FLAGS "-O3 -fopenmp")

# Create executable called "vmc"
add_executable(vmc ${SOURCES})
