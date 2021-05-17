! =============================================================================
!               Low-storage 4th order Runge-Kutta method
!            (see https://doi.org/10.5194/gmd-10-3145-2017)
! =============================================================================
module ls_rk4
    use options, only : parcel_info
    use parcel_container
    use parcel_bc
    use rk4_utils, only: get_B
    use parcel_interpl, only : par2grid, grid2par, grid2par_add
    use fields, only : velgradg, velog, vortg, velgradg
    use tri_inversion, only : vor2vel
    use parameters, only : nx, nz
    implicit none

    integer, parameter :: dp=kind(0.d0)           ! double precision

    double precision, allocatable, dimension(:, :) :: &
        velocity_p, &   ! position integration
        strain, &   ! strain at parcel location
        dbdt, &        ! B matrix integration
        dwdt           ! vorticity integration

    double precision, parameter :: &
        ca1 = - 567301805773.0_dp/1357537059087.0_dp,  &
        ca2 = -2404267990393.0_dp/2016746695238.0_dp,  &
        ca3 = -3550918686646.0_dp/2091501179385.0_dp,  &
        ca4 = -1275806237668.0_dp/842570457699.0_dp,   &
        ca5 = 0.0,   &  !dummy value, not actually used
        cb1 =  1432997174477.0_dp/9575080441755.0_dp,  &
        cb2 =  5161836677717.0_dp/13612068292357.0_dp, &
        cb3 =  1720146321549.0_dp/2090206949498.0_dp,  &
        cb4 =  3134564353537.0_dp/4481467310338.0_dp,  &
        cb5 =  2277821191437.0_dp/14882151754819.0_dp

    contains

        ! allocate memory of temporaries
        subroutine ls_rk4_alloc(num)
            integer, intent(in) :: num

            allocate(velocity_p(num, 2))
            allocate(dwdt(num, 1))

            if (parcel_info%is_elliptic) then
                allocate(strain(num, 4))
                allocate(dbdt(num, 2))
            endif

        end subroutine ls_rk4_alloc

        ! deallocate memory of temporaries
        subroutine ls_rk4_dealloc

            ! TODO
            deallocate(velocity_p)
            deallocate(dwdt)

            if (parcel_info%is_elliptic) then
               deallocate(strain)
               deallocate(dbdt)
            endif

        end subroutine ls_rk4_dealloc


        subroutine ls_rk4_step(dt)
            double precision, intent(in) :: dt

            if (parcel_info%is_elliptic) then
                call ls_rk4_elliptic(dt)
            else
                call ls_rk4_non_elliptic(dt)
            endif

        end subroutine ls_rk4_step


        subroutine ls_rk4_elliptic_substep(ds, ca, cb, dt, step)
            double precision, intent(in) :: ds(-1:nz+1, 0:nx-1, 1)  ! vorticity tendency
            double precision, intent(in) :: ca
            double precision, intent(in) :: cb
            double precision, intent(in) :: dt
            integer, intent(in) :: step

            call par2grid
            call vor2vel(vortg, velog, velgradg)

            if(step==1) then
               call grid2par(parcels%position, parcels%volume, velocity_p, velog, parcels%B)
               call grid2par(parcels%position, parcels%volume, dwdt, ds)
            else
               call grid2par_add(parcels%position, parcels%volume, velocity_p, velog, parcels%B)
               call grid2par_add(parcels%position, parcels%volume, dwdt, ds)
            endif
            call grid2par(parcels%position, parcels%volume, strain, velgradg, parcels%B)
            if(step==1) then
               dbdt(1:n_parcels,:) = get_B(parcels%B(1:n_parcels,:), strain(1:n_parcels,:), &
                                           parcels%volume(1:n_parcels, 1))
            else
               dbdt(1:n_parcels,:) = dbdt(1:n_parcels,:) &
                                   + get_B(parcels%B(1:n_parcels,:), strain(1:n_parcels,:), &
                                           parcels%volume(1:n_parcels, 1))
            endif
            parcels%position(1:n_parcels,:) = parcels%position(1:n_parcels,:) + cb*dt*velocity_p(1:n_parcels,:)
            parcels%vorticity(1:n_parcels, :) = parcels%vorticity(1:n_parcels, :) + cb*dt*dwdt(1:n_parcels, :)
            parcels%B(1:n_parcels,:) = parcels%B(1:n_parcels,:) + cb*dt*dbdt(1:n_parcels,:)
            call apply_parcel_bc(parcels%position, velocity_p)
            if(step==5) then
               return
            endif
            velocity_p(1:n_parcels,:) = ca*velocity_p(1:n_parcels,:)
            dwdt(1:n_parcels, :) = ca * dwdt(1:n_parcels, :)
            dbdt(1:n_parcels,:) = ca*dbdt(1:n_parcels,:)
            return

        end subroutine ls_rk4_elliptic_substep


        subroutine ls_rk4_elliptic(dt)
            double precision, intent(in) :: ds(-1:nz+1, 0:nx-1, 1)  ! vorticity tendency
            double precision, intent(in) :: dt

            ! at the moment we have no tendency!
            ds = zero

            call ls_rk4_elliptic_substep(ds, ca1, cb1, dt, 1)
            call ls_rk4_elliptic_substep(ds, ca2, cb2, dt, 2)
            call ls_rk4_elliptic_substep(ds, ca3, cb3, dt, 3)
            call ls_rk4_elliptic_substep(ds, ca4, cb4, dt, 4)
            call ls_rk4_elliptic_substep(ds, ca5, cb5, dt, 5)

        end subroutine ls_rk4_elliptic


        subroutine ls_rk4_non_elliptic_substep(ds, ca, cb, dt, step)
            double precision, intent(in) :: ds(-1:nz+1, 0:nx-1, 1)  ! vorticity tendency
            double precision, intent(in) :: ca
            double precision, intent(in) :: cb
            double precision, intent(in) :: dt
            integer, intent(in) :: step

            call par2grid
            call vor2vel(vortg, velog, velgradg)

            if(step==1) then
                call grid2par(parcels%position, parcels%volume, velocity_p, velog)
                call grid2par(parcels%position, parcels%volume, dwdt, ds)
            else
                call grid2par_add(parcels%position, parcels%volume, velocity_p, velog)
                call grid2par_add(parcels%position, parcels%volume, dwdt, ds)
            endif
            parcels%position(1:n_parcels,:) = parcels%position(1:n_parcels,:) + cb*dt*velocity_p(1:n_parcels,:)
            parcels%vorticity(1:n_parcels, :) = parcels%vorticity(1:n_parcels, :) + cb*dt*dwdt(1:n_parcels, :)
            call apply_parcel_bc(parcels%position, velocity_p)
            if(step==5) then
               return
            endif
            velocity_p(1:n_parcels,:) = ca*velocity_p(1:n_parcels,:)
            dwdt(1:n_parcels, :) = ca * dwdt(1:n_parcels, :)
            return

        end subroutine ls_rk4_non_elliptic_substep

        subroutine ls_rk4_non_elliptic(dt)
            double precision, intent(in) :: dt
            double precision             :: ds(-1:nz+1, 0:nx-1, 1)  ! vorticity tendency

            ! at the moment we have no tendency!
            ds = zero

            call ls_rk4_non_elliptic_substep(ds, ca1, cb1, dt, 1)
            call ls_rk4_non_elliptic_substep(ds, ca2, cb2, dt, 2)
            call ls_rk4_non_elliptic_substep(ds, ca3, cb3, dt, 4)
            call ls_rk4_non_elliptic_substep(ds, ca4, cb4, dt, 4)
            call ls_rk4_non_elliptic_substep(ds, ca5, cb5, dt, 5)

        end subroutine ls_rk4_non_elliptic

end module ls_rk4
