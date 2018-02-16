##########################################################################
set(files 
	global.F90)

add_library(spl_global ${files})
target_link_libraries(spl_global ${PLAF_LIBRARIES})

##########################################################################

