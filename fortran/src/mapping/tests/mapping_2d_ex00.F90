!> @brief 
!> mapping example in 2d 
!> @details
!> example of export/read_from_file for a 2d mapping in the nml format.
!> usage:
!>   $> ./mapping_2d_ex00 filename_mapping 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_bilinear
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  character(len=256)             :: filename_mapping
  logical                        :: file_exists, equal, empty

  ! ............................................
  ! Check that input argument was given
  ! ............................................
  if (command_argument_count() /= 1 ) then
    write(*,*) "ERROR: exactly 1 input argument is required"
    stop
  end if
  ! ............................................

  ! ............................................
  ! Read name of reference file from input argument
  ! ............................................
  call get_command_argument( 1, filename_mapping )
  ! ............................................

  ! ............................................
  ! Check that file exists    
  ! ............................................
  inquire( file=trim( filename_mapping ), exist=file_exists )
  if (.not. file_exists) then
    write(*,*) &
      "ERROR: reference file '"//trim( filename_mapping )//"' does not exist"
    stop
  end if
  ! ............................................

  ! ............................................
  call mapping % read_from_file(filename_mapping)
  call mapping % export('mapping_2d_ex00.nml')
  call mapping % free() 
  ! ............................................

end program main
! ............................................
