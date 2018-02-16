##########################################################################
SET(files 
	abstract.F90)

add_library(spl_abstract ${files})
target_link_libraries(spl_abstract spl_global)

##########################################################################
