##########################################################################
set(files 
	  sort.F90
	  calculus.F90
          utilities.F90)

add_library(spl_utilities ${files})
target_link_libraries(spl_utilities spl_global)


##########################################################################
