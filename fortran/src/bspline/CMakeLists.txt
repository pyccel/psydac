##########################################################################
set(files 
	    bsplvb.f90 
      interv.f90
	    bsplpp.f90 
	    ppvalu.f90 
	    pppack.f90 
      pp_form.F90 
	    bspline.F90)

add_library(spl_bspline ${files})

##########################################################################
