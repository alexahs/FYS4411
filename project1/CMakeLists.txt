project(project_1_fys4411)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    simple_vmc
    WaveFunctions
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "simple_vmc/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
)


# Create executable called "vmc.x"
add_executable(vmc.x ${SOURCES})
