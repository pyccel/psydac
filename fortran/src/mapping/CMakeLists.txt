##########################################################################
set(files 
	mapping_abstract.F90
	mapping_1d.F90
	mapping_2d.F90
	mapping_3d.F90
	mapping_cad.F90
	mapping_gallery.F90)

add_library(spl_mapping ${files})

target_link_libraries(spl_mapping spl_bspline spl_abstract spl_utilities spl_global)

##########################################################################
