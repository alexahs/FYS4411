project(project_1_fys4411)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
)


# Create executable called "vmc.x"
add_executable(vmc.x ${SOURCES})
