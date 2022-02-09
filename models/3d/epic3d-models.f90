! =============================================================================
!               This program writes fields to HDF5 in EPIC format.
! =============================================================================
program epic3d_models
    use options, only : filename, verbose
    use taylor_green_3d
    use robert_3d
    use moist_3d
    use constants, only : pi
    use parameters, only : nx, ny, nz, dx, lower, extent
    use h5_utils
    use h5_writer
    implicit none

    character(len=512) :: model = ''
    character(len=512) :: h5fname = ''
    integer(hid_t)     :: h5handle

    type box_type
        integer          :: ncells(3)   ! number of cells
        double precision :: extent(3)   ! size of domain
        double precision :: origin(3)   ! origin of domain (lower left corner)
    end type box_type

    type(box_type) :: box


    ! Read command line (verbose, filename, etc.)
    call parse_command_line

    call read_config_file

    call initialise_hdf5

    call generate_fields

    call finalise_hdf5

    contains

        subroutine generate_fields

            call create_h5_file(h5fname, .false., h5handle)

            dx = box%extent / dble(box%ncells)
            nx = box%ncells(1)
            ny = box%ncells(2)
            nz = box%ncells(3)

            select case (trim(model))
                case ('TaylorGreen')
                    ! make origin and extent always a multiple of pi
                    box%origin = pi * box%origin
                    box%extent = pi * box%extent
                    dx = dx * pi

                    call taylor_green_init(h5handle, nx, ny, nz, box%origin, dx)
                case ('Robert')
                    call robert_init(h5handle, nx, ny, nz, box%origin, dx)
                case ('MoistPlume')
                    call moist_init(h5handle, nx, ny, nz, box%origin, dx)
                case default
                    print *, "Unknown model: '", trim(model), "'."
                    stop
            end select

            ! write box
            lower = box%origin
            extent = box%extent
            call write_h5_box(h5handle, lower, extent, (/nx, ny, nz/))
            call close_h5_file(h5handle)
        end subroutine generate_fields


        ! parse configuration file
        ! (see https://cyber.dabamos.de/programming/modernfortran/namelists.html [8 March 2021])
        subroutine read_config_file
            integer :: ios
            integer :: fn = 1
            logical :: exists = .false.

            ! namelist definitions
            namelist /MODELS/ model, h5fname, box, tg_flow, robert_flow, moist

            ! check whether file exists
            inquire(file=filename, exist=exists)

            if (exists .eqv. .false.) then
                print *, 'Error: input file "', trim(filename), '" does not exist.'
                stop
            endif

            ! open and read Namelist file.
            open(action='read', file=filename, iostat=ios, newunit=fn)

            read(nml=MODELS, iostat=ios, unit=fn)

            if (ios /= 0) then
                print *, 'Error: invalid Namelist format.'
                stop
            end if

            close(fn)

            ! check whether h5 file already exists
            inquire(file=h5fname, exist=exists)

            if (exists) then
                print *, 'Error: output file "', trim(h5fname), '" already exists.'
                stop
            endif
        end subroutine read_config_file

        ! Get the file name provided via the command line
        subroutine parse_command_line
            integer                          :: i
            character(len=512)               :: arg

            i = 0
            do
                call get_command_argument(i, arg)
                if (len_trim(arg) == 0) then
                    exit
                endif

                if (arg == '--config') then
                    i = i + 1
                    call get_command_argument(i, arg)
                    filename = trim(arg)
                else if (arg == '--verbose') then
                    verbose = .true.
                else if (arg == '--help') then
                    print *, 'Run code with "epic3d-models --config [config file]"'
                    stop
                endif
                i = i+1
            end do

            if (filename == '') then
                print *, 'No configuration file provided. Run code with "epic3d-models --config [config file]"'
                stop
            endif
        end subroutine parse_command_line
end program epic3d_models
