CMAKE_MINIMUM_REQUIRED ( VERSION 3.8 )
PROJECT("ViolonOptimalCasADi" CXX C)

# Set folder to look path to
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Set a default build type to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
	SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
ENDIF()

find_package(CASADI REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(RBDL REQUIRED)
find_package(Dlib REQUIRED)
find_package(Biorbd REQUIRED)
link_directories(${filename}
  ${CMAKE_INSTALL_PREFIX}/lib
)


add_library(GJCS_casadi SHARED src/GJCS_casadi.cpp)

target_link_libraries(GJCS_casadi
    ${Biorbd_LIBRARY}
    ${RBDL_LIBRARY}
  )
    
set_target_properties(GJCS_casadi PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

target_include_directories(GJCS_casadi PUBLIC
    include
    ${CMAKE_INSTALL_PREFIX}/include
    ${Biorbd_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${RBDL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

# Write down all files to include here
set(source_files
    eocar.cpp
)

# Setup each project
foreach(src_file ${source_files})
    get_filename_component(filename ${src_file} NAME_WE)

    # link_directories(${ACADO_LIBRARY_DIRS})
    ADD_EXECUTABLE (${filename}
        ${filename}.cpp
    )
    target_include_directories(${filename} PUBLIC
        ${CASADI_INCLUDE_DIRS}
        ${Biorbd_INCLUDE_DIR}
    )

    target_link_libraries(${filename}
        GJCS_casadi
        ${CASADI_LIBRARIES}
        ${Biorbd_LIBRARIES}
      )
endforeach()

install(TARGETS GJCS_casadi DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
