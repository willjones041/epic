! =============================================================================
!                       Test ellipse multi merge
!
!         This unit test checks the merging of five ellipses. The biggest
!         ellipse is located ad the origin. The smaller ellipses are located
!         on all four sides. The final ellipse is a circle located at the
!         origin.
! =============================================================================
program test_ellipse_multi_merge
    use constants, only : pi
    use parcel_container
    use parcel_merge, only : merge_ellipses
    use options, only : parcel_info, grid
    use parameters, only : update_parameters
    use ellipse
    implicit none

    double precision :: ab, a1b1, a2b2, B11, B12, B22, error, vol

    grid = (/2, 2/)

    call update_parameters()

    call parcel_alloc(5)

    n_parcels = 5
    a1b1 = 2.0d0
    a2b2 = 1.0d0


    !
    ! muti-geometric merging
    !

    call parcel_setup

    ! geometric merge
    parcel_info%lambda = 5.0
    parcel_info%merge_type = 'multi-geometric'
    parcel_info%vfraction = 2

    call merge_ellipses(parcels)

    ! reference solution
    ab = a1b1 + 4.0d0 * a2b2  ! a == b since it is a circle
    B11 = ab
    B12 = 0.0d0
    B22 = ab
    vol = ab * pi

    !
    ! check result
    !
    error = 0.0d0

    error = max(error, abs(dble(n_parcels - 1)))
    error = max(error, abs(parcels%B(1, 1) - B11))
    error = max(error, abs(parcels%B(1, 2) - B12))
    error = max(error, abs(get_B22(parcels%B(1, 1), &
                                   parcels%B(1, 2), &
                                   parcels%volume(1, 1)) - B22))
    error = max(error, sum(abs(parcels%position(1, :))))
    error = max(error, abs(parcels%volume(1, 1) - vol))

    if (error > 1.0e-15) then
        print '(a32, a7)', 'Test ellipse multi-merge (geo):', 'FAILED'
    else
        print '(a32, a7)', 'Test ellipse multi-merge (geo):', 'PASSED'
    endif

    !
    ! muti-optimal merging
    !

    call parcel_setup

    ! geometric merge
    parcel_info%lambda = 5.0
    parcel_info%merge_type = 'multi-geometric'
    parcel_info%vfraction = 2
    n_parcels = 5

    call parcel_setup

    ! geometric merge
    parcel_info%lambda = 5.0
    parcel_info%merge_type = 'multi-optimal'
    parcel_info%vfraction = 2

    call merge_ellipses(parcels)

    !
    ! check result
    !
    error = 0.0d0

    error = max(error, abs(dble(n_parcels - 1)))
    error = max(error, abs(parcels%B(1, 1) - B11))
    error = max(error, abs(parcels%B(1, 2) - B12))
    error = max(error, abs(get_B22(parcels%B(1, 1), &
                                   parcels%B(1, 2), &
                                   parcels%volume(1, 1)) - B22))
    error = max(error, sum(abs(parcels%position(1, :))))
    error = max(error, abs(parcels%volume(1, 1) - vol))


    if (error > 1.0e-15) then
        print '(a32, a7)', 'Test ellipse multi-merge (opt):', 'FAILED'
    else
        print '(a32, a7)', 'Test ellipse multi-merge (opt):', 'PASSED'
    endif

    call parcel_dealloc

    contains

        subroutine parcel_setup
            parcels%position(1, :) = 0.0d0
            parcels%volume(1, 1) = a1b1 * pi
            parcels%B(1, 1) = a1b1
            parcels%B(1, 2) = 0.0d0

            ! small parcel left
            parcels%position(2, 1) = -1.5d0
            parcels%position(2, 2) = 0.0d0
            parcels%volume(2, 1) = a2b2 * pi
            parcels%B(2, 1) = a2b2
            parcels%B(2, 2) = 0.0d0

            ! small parcel right
            parcels%position(3, 1) = 1.5d0
            parcels%position(3, 2) = 0.0d0
            parcels%volume(3, 1) = a2b2 * pi
            parcels%B(3, 1) = a2b2
            parcels%B(3, 2) = 0.0d0

            ! small parcel below
            parcels%position(4, 1) = 0.0d0
            parcels%position(4, 2) = -1.5d0
            parcels%volume(4, 1) = a2b2 * pi
            parcels%B(4, 1) = a2b2
            parcels%B(4, 2) = 0.0d0

            ! small parcel above
            parcels%position(5, 1) = 0.0d0
            parcels%position(5, 2) = 1.5d0
            parcels%volume(5, 1) = a2b2 * pi
            parcels%B(5, 1) = a2b2
            parcels%B(5, 2) = 0.0d0
        end subroutine parcel_setup

end program test_ellipse_multi_merge
