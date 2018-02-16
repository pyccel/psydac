# Create local variables
set( SPL_LIBS "" )  # Internal libraries (part of the SPL project)
set( SPL_EXT  "" )  # External libraries (built with SPL)
set( SPL_DEPS "" )  # Dependencies (pre-installed libraries)

# Extract names of all libraries built in SPL
get_property( library_targets GLOBAL PROPERTY LIBRARY_TARGETS )

# Distinguish between "internal" and "external" libraries
foreach( lib ${library_targets} )
  if( ${lib} MATCHES "spl_[a-zA-Z0-9]" )
    list( APPEND SPL_LIBS ${lib} )
  else()
    get_target_property( libpath ${lib} SOURCE_DIR )
    if( libpath MATCHES "${CMAKE_SOURCE_DIR}/external" )
      list( APPEND SPL_EXT ${lib} )
    endif()
  endif()
endforeach()

# Extract system dependencies
foreach( lib ${library_targets} )
  get_target_property( deps ${lib} INTERFACE_LINK_LIBRARIES )
  if( deps )
    foreach( dep ${deps} )
      list( FIND library_targets ${dep} idx )
      if( ${idx} EQUAL -1 )
        #get_filename_component(libdep "${dep}" NAME_WE)
        #list( APPEND SPL_DEPS ${libdep} )
        list( APPEND SPL_DEPS ${dep} )
      endif()
    endforeach()
  endif()
endforeach()

# Remove duplicates without breaking dependency order
list( REVERSE SPL_DEPS )
list( REMOVE_DUPLICATES SPL_DEPS )
list( REVERSE SPL_DEPS )

#===============================================================================

# Print names of all sub-libraries in SPL
message( STATUS "SPL LIBRARIES:" )
foreach( lib ${SPL_LIBS} )
  message( STATUS "  ${lib}" )
endforeach()

# Print names of built-in external 
message( STATUS "BUILT-IN EXTERNAL LIBRARIES:" )
foreach( lib ${SPL_EXT} )
  message( STATUS "  ${lib}" )
endforeach()

# Print all external dependencies
message( STATUS "EXTERNAL DEPENDENCIES:" )
foreach( lib ${SPL_DEPS} )
  message( STATUS "  ${lib}" )
endforeach()

#===============================================================================

# Extract file paths of all libraries
set( libpaths )
foreach( lib ${SPL_LIBS} )
  list( APPEND libpaths $<TARGET_FILE:${lib}> )
endforeach()

# CMake custom command: repackage SPL into single archive "libSPL.a"
# NOTE: Command does not run if file is newer than its dependencies
set( SPL_LIB "${CMAKE_CURRENT_BINARY_DIR}/libspl.a" )
add_custom_command( OUTPUT ${SPL_LIB}
  DEPENDS ${SPL_LIBS}
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_package.py ${SPL_LIB} ${libpaths}
  COMMENT "Repackage all static libraries into a single archive: 'libspl.a'"
  VERBATIM )

# CMake target: update "libSPL.a" if obsolete, otherwise do nothing
# USAGE: make SPL-archive
add_custom_target( SPL-archive ALL DEPENDS ${SPL_LIB} )

# Wrap "libSPL.a" with CMake imported library "SPL", which depends on
# above "SPL-archive" target and therefore creates archive only if needed.
# USAGE: target_link_libraries( <MY_LIB> SPL ${SPL_EXT} ${SPL_DEPS} )
add_library( SPL STATIC IMPORTED GLOBAL )
set_target_properties( SPL PROPERTIES IMPORTED_LOCATION ${SPL_LIB} )
add_dependencies( SPL SPL-archive )

#===============================================================================

# Store cache variables (available to simulations)
set( SPL_LIB  ${SPL_LIB}  CACHE INTERNAL "Path to full SPL library" )
set( SPL_EXT  ${SPL_EXT}  CACHE INTERNAL "List of built-in external libraries" )
set( SPL_DEPS ${SPL_DEPS} CACHE INTERNAL "List of external dependencies" )

# Installation (library, headers, mod files)
install( FILES ${SPL_LIB}    DESTINATION lib )
install( DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}/"
  DESTINATION "include/spl" FILES_MATCHING PATTERN "spl_*.mod" )

#===============================================================================
# Print PALFConfig.cmake for external use with cmake

set( filepath "${CMAKE_CURRENT_BINARY_DIR}/SPLConfig.cmake" )
file( WRITE  ${filepath} "SET( CLAPP_DIR ${CMAKE_INSTALL_PREFIX})\n" )
file( APPEND ${filepath} "\nSET( SPL_LIBRARIES \${CLAPP_DIR}/lib/libspl.a" )

foreach( lib ${SPL_EXT} )
  file( APPEND ${filepath} "\n" )
  file( APPEND ${filepath} "	\${CLAPP_DIR}/lib/lib${lib}.a" )
endforeach()
foreach( lib ${SPL_DEPS} )
  file( APPEND ${filepath} "\n" )
  #file( APPEND ${filepath} "	\${CLAPP_DIR}/lib/${lib}.a" )
  file( APPEND ${filepath} "	${lib}" )
endforeach()
file( APPEND ${filepath} ")\n" )

# Set include path
file( APPEND ${filepath} "\nSET( SPL_INCLUDE_PATH  \${CLAPP_DIR}/include/spl)\n" )

# Add include path
file( APPEND ${filepath} "\nINCLUDE_DIRECTORIES( \${SPL_INCLUDE_PATH} )\n" )



# Installation
install( CODE "file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/cmake)" )
install( FILES ${filepath} DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake )

