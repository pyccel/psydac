#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)
IF(OPTIONS_FILE)
INCLUDE(${OPTIONS_FILE})
ENDIF()

#Name your project here
project(SPL)
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
SET(CMAKE_SHARED_LIBRARY_LINK_FORTRAN_FLAGS "")

enable_language (Fortran)

IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} 
       CACHE STRING "Choose the type of build, options are: Debug Release.")
ELSE()
   SET(CMAKE_BUILD_TYPE Debug 
       CACHE STRING "Choose the type of build, options are: Debug Release.")
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES Debug)
   ADD_DEFINITIONS(-DDEBUG)
ENDIF()

set(SPLINES_BUILD_DIR $ENV{PWD} )

IF(APPLE)
  cmake_policy(SET CMP0042 NEW)
ENDIF(APPLE)

################################
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

# the RPATH to be used when installing
SET(CMAKE_INSTALL_RPATH "")

# don't add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
################################

# Add new directory to find the FIND<PACKAGE>.cmake files. CMAKE_SOURCE_DIR is
# the directory where the project sources are located. CMAKE_MODULE_PATH is 
# the list of directories in which to search for CMake modules.
SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH} )

SET(BUILD_LIBRARY         ON  CACHE BOOL "Build SPL library")
SET(BUILD_TESTING         OFF CACHE BOOL "Build SPL library tests")
SET(BUILD_EXAMPLES        ON  CACHE BOOL "Build SPL library examples")
SET(BUILD_PACKAGE         OFF CACHE BOOL "Build SPL python package")
SET(MPI_ENABLED           OFF CACHE BOOL "Enable use of MPI library")

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  GET_FILENAME_COMPONENT(INSTALL_DIR "${CMAKE_SOURCE_DIR}/../usr" ABSOLUTE)
  SET(CMAKE_INSTALL_PREFIX ${INSTALL_DIR} CACHE PATH
    "Default installation directory" FORCE)
ENDIF()

INCLUDE(FortranConfig)
INCLUDE(HDF5Config)
INCLUDE(CTest)
INCLUDE(PreprocessorTarget)

set( CLAPP_DIR $ENV{CLAPP_DIR} )
if( NOT CLAPP_DIR )
  set( CLAPP_DIR ${CMAKE_INSTALL_PREFIX} )
endif()


IF(MPI_ENABLED)
   INCLUDE(MPIConfig)
   INCLUDE(ProcessorCount)
   INCLUDE(POEConfig)
   ADD_DEFINITIONS(-DMPI_ENABLED)
   SET(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
ENDIF()

FIND_PACKAGE(PythonInterp 2.7.3)
SET(PYTHON_FOUND FALSE)
MESSAGE(STATUS "PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")
MESSAGE(STATUS "PYTHON_VERSION_STRING:${PYTHON_VERSION_STRING}")
IF(PYTHON_EXECUTABLE AND PYTHON_VERSION_STRING MATCHES "2.*")
   STRING(COMPARE GREATER ${PYTHON_VERSION_STRING} "2.6.0" PYTHON_FOUND)
ENDIF()
MESSAGE(STATUS "PYTHON_FOUND:${PYTHON_FOUND}")

file(MAKE_DIRECTORY ${SPLINES_BUILD_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SPLINES_BUILD_DIR}/bin)

file(MAKE_DIRECTORY ${SPLINES_BUILD_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${SPLINES_BUILD_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SPLINES_BUILD_DIR}/lib)


# Define folder to be deleted by make clean         
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/modules/)
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/)
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/include/)

ADD_CUSTOM_TARGET( clear
   COMMAND find . -name parameter* -delete
   COMMAND find . -name *.mod -delete
   COMMAND find . -name *.a -delete
   COMMAND find . -name *.o -delete
   COMMAND find . -name *.tp -delete
   COMMAND find . -name *.cmake -delete
   COMMAND find . -name *.xmf -delete
   COMMAND find . -name *.h5 -delete
   COMMAND find . -name *.gnu -delete
   COMMAND find . -name *.dat -delete
   COMMAND find . -name *.mat -delete
   COMMAND find . -name *.mtx -delete
   COMMAND find . -name *.txt -delete
   COMMAND find . -name *.log -delete
   COMMAND rm -rf bin
   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
   COMMENT "Delete all .a, .mod, .o, .tp, .cmake, .xmf, .h5, .gnu, .dat, .mat, .mtx, .txt, .log and folders /modules, /bin" VERBATIM)

##########################################################################
# 
##########################################################################
ENABLE_TESTING()

# Overwrite 'add_library' and 'add_executable' commands to collect info
INCLUDE(PreprocessorTarget)
# library and models

ADD_SUBDIRECTORY(fortran)

# python 
#IF(BUILD_PACKAGE)
#   add_subdirectory(python)
#ENDIF(BUILD_PACKAGE) 
##########################################################################

##########################################################################
IF(CMAKE_Fortran_COMPILER MATCHES "gfortran*")
  SET(COMPILER_ID 1 CACHE INTEGER "Identifier for the GCC compiler")

ELSEIF(CMAKE_Fortran_COMPILER MATCHES "ifort")
  SET(COMPILER_ID 2 CACHE INTEGER "Identifier for the IFORT compiler")

ELSEIF(CMAKE_Fortran_COMPILER MATCHES "pgi")
  SET(COMPILER_ID 3 CACHE INTEGER "Identifier for the PGI compiler")
ENDIF()
##########################################################################

##########################################################################
MESSAGE(STATUS "############################################################")
MESSAGE(STATUS "####")
MESSAGE(STATUS "#### CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "#### Fortran_COMPILER_NAME:${Fortran_COMPILER_NAME}")
MESSAGE(STATUS "#### CMAKE_Fortran_COMPILER:${CMAKE_Fortran_COMPILER}")
MESSAGE(STATUS "#### Fortran FLAGS ")
IF(CMAKE_BUILD_TYPE MATCHES "Debug") 
  MESSAGE(STATUS "#### CMAKE_Fortran_FLAGS_DEBUG: ${CMAKE_Fortran_FLAGS_DEBUG}")
ELSEIF(CMAKE_BUILD_TYPE MATCHES "Release") 
  MESSAGE(STATUS "#### CMAKE_Fortran_FLAGS_RELEASE: ${CMAKE_Fortran_FLAGS_RELEASE}")
ELSE()
  MESSAGE(STATUS "#### CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}")
ENDIF()
MESSAGE(STATUS "####")
MESSAGE(STATUS "############################################################")
##########################################################################

# Create single library 'spllib.a' from smaller libraries in src
ADD_SUBDIRECTORY( package )
