##########################################################################
# splines tests directory
##########################################################################

# ... ex1
ADD_EXECUTABLE(bspline_ex1 ex1.F90)
TARGET_LINK_LIBRARIES(bspline_ex1 spl_bspline)
add_test(NAME bsplines_ex1 COMMAND bspline_ex1)
# ...
# ... ex2
ADD_EXECUTABLE(bspline_ex2 ex2.F90)
TARGET_LINK_LIBRARIES(bspline_ex2 spl_bspline)
add_test(NAME bsplines_ex2 COMMAND bspline_ex2)
# ...
# ... ex3
ADD_EXECUTABLE(bspline_ex3 ex3.F90)
TARGET_LINK_LIBRARIES(bspline_ex3 spl_bspline)
add_test(NAME bsplines_ex3 COMMAND bspline_ex3)
# ...
# ... ex4
ADD_EXECUTABLE(bspline_ex4 ex4.F90)
TARGET_LINK_LIBRARIES(bspline_ex4 spl_bspline spl_utilities)
add_test(NAME bsplines_ex4 COMMAND bspline_ex4)
# ...
##########################################################################
