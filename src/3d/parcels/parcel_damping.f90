! =============================================================================
! This module contains the subroutines to do damping of parcel properties to
! gridded fields in a conservative manner. It does this by nudging all the  
! parcels associated with a grid point to the grid point value, with the strength
! of damping proportional to the grid point contribution to the gridded value and
! the strain rate at the grid point.
! =============================================================================
module parcel_damping
    use constants, only :  f14, zero, one
    use mpi_timer, only : start_timer, stop_timer
    use parameters, only : nx, nz, vmin 
    use parcel_container, only : parcels, n_parcels
    use parcel_ellipsoid
    use parcel_interpl
    use fields
    use omp_lib
    use mpi_layout, only : box 
    use options, only : damping
    use inversion_mod, only : vor2vel
    use rk_utils, only : get_strain_magnitude_field
    implicit none


    private

    ! interpolation indices
    ! (first dimension x, y, z; second dimension l-th index)
    integer :: is, js, ks
    integer :: damping_timer
 
    ! interpolation weights
    double precision :: weights(0:1,0:1,0:1)
    double precision :: weight(0:1,0:1,0:1)
    double precision :: time_fact(0:1,0:1,0:1)
  
    public :: parcel_damp, damping_timer

    contains

        subroutine parcel_damp(dt)
            double precision, intent(in)  :: dt

            if (damping%l_vorticity .or. damping%l_scalars .or. &
                damping%l_surface_vorticity .or. damping%l_surface_scalars) then 
                ! ensure gridded fields are up to date
                call par2grid(.false.)
                call vor2vel

                ! Reflect beyond boundaries to ensure damping is conservative
                ! This is because the points below the surface contribute to the level above
                !$omp parallel workshare
                vortg(-1,   :, :, :) = vortg(1, :, :, :)
                vortg(nz+1, :, :, :) = vortg(nz-1, :, :, :)

#ifndef ENABLE_DRY_MODE
                qvg(-1,   :, :) = qvg(1, :, :)
                qvg(nz+1, :, :) = qvg(nz-1, :, :)
                qlg(-1,   :, :) = qlg(1, :, :)
                qlg(nz+1, :, :) = qlg(nz-1, :, :)
#endif
                thetag(-1,   :, :) = thetag(1, :, :)
                thetag(nz+1, :, :) = thetag(nz-1, :, :)
                !$omp end parallel workshare

                call get_strain_magnitude_field
                call perturbation_damping(dt, .true.)
            end if

        end subroutine parcel_damp

        ! 
        ! @pre 
        subroutine perturbation_damping(dt, l_reuse)
            double precision, intent(in)  :: dt
            logical, intent(in)           :: l_reuse
#if defined (ENABLE_P2G_1POINT) && !defined (NDEBUG)
            logical                       :: l_reuse_dummy
#endif
            integer                       :: n, p, l 
            double precision              :: points(3, n_points_p2g)
            double precision              :: pvol
            ! tendencies need to be summed up between associated 4 points
            ! before modifying the parcel attribute
            double precision              :: vortend(3)
            double precision              :: thetatend
#ifndef ENABLE_DRY_MODE
            double precision              :: qltend
            double precision              :: qvtend
#endif

            call start_timer(damping_timer)

            ! This is only here to allow debug compilation
            ! with a warning for unused variables
#if defined (ENABLE_P2G_1POINT) && !defined (NDEBUG)
            l_reuse_dummy=l_reuse
#endif

            !$omp parallel default(shared)
            !$omp do private(n, p, l, points, pvol, weight) &
#ifndef ENABLE_DRY_MODE
            !$omp& private(is, js, ks, weights, vortend, thetatend, qvtend, qltend, time_fact) 
#else
            !$omp& private(is, js, ks, weights, vortend, thetatend, time_fact) 
#endif
            do n = 1, n_parcels
                pvol = parcels%volume(n)
#ifndef ENABLE_P2G_1POINT
                points = get_ellipsoid_points(parcels%position(:, n), &
                                              parcels%B(:, n), n, l_reuse)
#else
                points(:, 1) = parcels%position(:, n)
#endif
                vortend = zero
                thetatend = zero
#ifndef ENABLE_DRY_MODE
                qvtend = zero
                qltend = zero
#endif

                ! we have 4 points per ellipsoid
                do p = 1, n_points_p2g
                    call trilinear(points(:, p), is, js, ks, weights)
                    weight = point_weight_p2g * weights
                    if (damping%l_vorticity) then
                        ! Note this exponential factor can be different for vorticity/scalars
                        time_fact = one - exp(-damping%vorticity_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
                        do l = 1,3
                            vortend(l) = vortend(l)+sum(weight * time_fact * (vortg(ks:ks+1, js:js+1, is:is+1, l) &
                                       - parcels%vorticity(l,n)))
                        enddo
                    else if (damping%l_surface_vorticity) then
                        ! outside the bounds
                        if (ks < box%lo(3))  then
                            ! Note this exponential factor can be different for vorticity/scalars
                            time_fact = one - exp(-damping%vorticity_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
                            do l = 1,3
                                vortend(l) = vortend(l)+sum(weight(1, :, :) * time_fact(1, :, :) * &
                                         (vortg(ks+1, js:js+1, is:is+1, l)  - parcels%vorticity(l,n)))
                            enddo
                        elseif ((ks < box%lo(3)+1) .or. (ks+1 > box%hi(3))) then
                            ! Note this exponential factor can be different for vorticity/scalars
                            time_fact = one - exp(-damping%vorticity_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
                            do l = 1,3
                                vortend(l) = vortend(l)+sum(weight(0, :, :) * time_fact(0, :, :) * &
                                         (vortg(ks, js:js+1, is:is+1, l)  - parcels%vorticity(l,n)))
                            enddo
                        elseif (ks+1 > box%hi(3)-1) then
                            ! Note this exponential factor can be different for vorticity/scalars
                            time_fact = one - exp(-damping%vorticity_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
                            do l = 1,3
                                vortend(l) = vortend(l)+sum(weight(1, :, :) * time_fact(1, :, :) * &
                                         (vortg(ks+1, js:js+1, is:is+1, l)  - parcels%vorticity(l,n)))
                            enddo
                         endif
                    endif
                    if (damping%l_scalars) then
                        time_fact = one - exp(-damping%scalars_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
#ifndef ENABLE_DRY_MODE
                        qvtend = qvtend + sum(weight * time_fact * (qvg(ks:ks+1, js:js+1, is:is+1) - parcels%qv(n)))
                        qltend = qltend + sum(weight * time_fact * (qlg(ks:ks+1, js:js+1, is:is+1) - parcels%ql(n)))
#endif
                        thetatend = thetatend + sum(weight * time_fact * (thetag(ks:ks+1, js:js+1, is:is+1) - parcels%theta(n)))
                    else if (damping%l_surface_scalars) then
                        ! outside the bounds
                        if (ks < box%lo(3))  then
                            ! Note this exponential factor can be different for vorticity/scalars
                            time_fact = one - exp(-damping%scalars_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
#ifndef ENABLE_DRY_MODE
                            qvtend = qvtend + sum(weight(1, :, :) * time_fact(1, :, :) * &
                                                  (qvg(ks+1, js:js+1, is:is+1) - parcels%qv(n)))
                            qltend = qltend + sum(weight(1, :, :) * time_fact(1, :, :) * &
                                                  (qlg(ks+1, js:js+1, is:is+1) - parcels%ql(n)))
#endif
                            thetatend = thetatend + sum(weight(1, :, :) * time_fact(1, :, :) * &
                                                        (thetag(ks+1, js:js+1, is:is+1) - parcels%theta(n)))
                        elseif ((ks < box%lo(3)+1) .or. (ks+1 > box%hi(3))) then
                            ! Note this exponential factor can be different for vorticity/scalars
                            time_fact = one - exp(-damping%scalars_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
#ifndef ENABLE_DRY_MODE
                            qvtend = qvtend + sum(weight(0, :, :) * time_fact(0, :, :) * &
                                                  (qvg(ks, js:js+1, is:is+1) - parcels%qv(n)))
                            qltend = qltend + sum(weight(0, :, :) * time_fact(0, :, :) * &
                                                  (qlg(ks, js:js+1, is:is+1) - parcels%ql(n)))
#endif
                            thetatend = thetatend + sum(weight(0, :, :) * time_fact(0, :, :) * &
                                                        (thetag(ks, js:js+1, is:is+1) - parcels%theta(n)))
                        elseif (ks+1 > box%hi(3)-1) then
                            ! Note this exponential factor can be different for vorticity/scalars
                            time_fact = one - exp(-damping%scalars_prefactor * strain_mag(ks:ks+1, js:js+1, is:is+1) * dt)
#ifndef ENABLE_DRY_MODE
                            qvtend = qvtend + sum(weight(1, :, :) * time_fact(1, :, :) * &
                                                  (qvg(ks+1, js:js+1, is:is+1) - parcels%qv(n)))
                            qltend = qltend + sum(weight(1, :, :) * time_fact(1, :, :) * &
                                                  (qlg(ks+1, js:js+1, is:is+1) - parcels%ql(n)))
#endif
                            thetatend = thetatend + sum(weight(1, :, :) * time_fact(1, :, :) * &
                                                        (thetag(ks+1, js:js+1, is:is+1) - parcels%theta(n)))
                         endif
                    endif
                enddo
                ! Add all the tendencies only at the end
                if (damping%l_vorticity .or. damping%l_surface_vorticity) then
                    do l=1,3
                        parcels%vorticity(l,n) = parcels%vorticity(l,n) + vortend(l)
                    enddo
                endif
                if (damping%l_scalars .or. damping%l_surface_scalars) then
#ifndef ENABLE_DRY_MODE
                    parcels%qv(n) = parcels%qv(n) + qvtend
                    parcels%ql(n) = parcels%ql(n) + qltend
#endif
                    parcels%theta(n) = parcels%theta(n) + thetatend
                endif
            enddo
            !$omp end do
            !$omp end parallel

            call stop_timer(damping_timer)

        end subroutine perturbation_damping


end module parcel_damping
