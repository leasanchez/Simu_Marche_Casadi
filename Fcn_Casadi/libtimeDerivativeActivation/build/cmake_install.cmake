# Install script for directory: /home/leasanchez/programmation/Marche_Florent/Marche_Florent_Casadi/Fcn_Casadi/libtimeDerivativeActivation

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/leasanchez/programmation/miniconda3/envs/marche")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so"
         RPATH "/home/leasanchez/programmation/miniconda3/envs/marche/lib:/home/leasanchez/programmation/miniconda3/envs/marche/lib/biorbd")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/leasanchez/programmation/miniconda3/envs/marche/lib" TYPE SHARED_LIBRARY FILES "/home/leasanchez/programmation/Marche_Florent/Marche_Florent_Casadi/Fcn_Casadi/libtimeDerivativeActivation/build/libtime_Derivative_Activation.so")
  if(EXISTS "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so"
         OLD_RPATH "/home/leasanchez/programmation/miniconda3/envs/marche/lib:/home/leasanchez/programmation/miniconda3/envs/marche/lib/biorbd:"
         NEW_RPATH "/home/leasanchez/programmation/miniconda3/envs/marche/lib:/home/leasanchez/programmation/miniconda3/envs/marche/lib/biorbd")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/leasanchez/programmation/miniconda3/envs/marche/lib/libtime_Derivative_Activation.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/leasanchez/programmation/Marche_Florent/Marche_Florent_Casadi/Fcn_Casadi/libtimeDerivativeActivation/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
