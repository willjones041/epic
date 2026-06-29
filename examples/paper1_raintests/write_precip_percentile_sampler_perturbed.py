#!/usr/bin/env python
import numpy as np
import xarray as xr
from scipy.stats import gamma as gamma_dist

# Based on the original percentile sampler, with a position-dependent qr perturbation.


def lambda_from_nr_qr(nr, qr, rho_water, rho_air, mu):
    """Compute slope parameter lambda for CASIM gamma PSD."""
    return ((np.pi * nr * rho_water * (mu + 3) * (mu + 2) * (mu + 1)) /
            (6.0 * qr * rho_air)) ** (1.0 / 3.0)


def sample_mass_weighted(nr, qr, rho_water, rho_air, mu, multiplicity=10):
    """Return `multiplicity` diameters at evenly spaced percentiles of the
    mass-weighted CASIM gamma PSD (avoid 0 and 1).
    """
    lam = lambda_from_nr_qr(nr, qr, rho_water, rho_air, mu)
    shape_mass = mu + 4.0
    scale = 1.0 / lam
    p = np.linspace(0.5 / multiplicity, 1.0 - 0.5 / multiplicity, multiplicity)
    return gamma_dist.ppf(p, a=shape_mass, scale=scale)


def equivalent_number_concentration(D, qr, rho_water, rho_air):
    """Equivalent N_r if all mass q were in droplets of diameter D."""
    mass_per_drop = (np.pi / 6.0) * rho_water * D**3
    return rho_air * qr / mass_per_drop


def qr_perturbation(qr_base, del_x, del_z):
    """Apply the requested position-dependent qr perturbation.

    qr_perturbed = qr * (1 - 0.4 * del_x * del_z)
    """
    qr_perturbed = qr_base * (1.0 - 0.4 * del_x * del_z)
    return max(qr_perturbed, 1e-12)


def main():
    # plume/grid params
    r_plume = 500.0
    centre = [6400, 3000]
    dx_parcel = 10.0
    shape_factor = 1.0

    a = r_plume / shape_factor
    b = r_plume * shape_factor
    n_steps = int(np.ceil(max(a, b) / dx_parcel))
    parcel_shifts = np.arange(-n_steps, n_steps + 1) * dx_parcel

    # microphysics
    qr_parcels = 0.001
    Nr_parcels = 60000
    mu = 2.5
    rho_water = 1000.0
    rho_air = 1.2256
    multiplicity = 160

    x_list = []
    z_list = []
    D_list = []
    qr_list = []
    qr_location_list = []

    for sx in parcel_shifts:
        for sz in parcel_shifts:
            if ((sx / a) ** 2 + (sz / b) ** 2) < 1.0:
                # normalized position variables used in the perturbation
                del_x = sx / a
                del_z = sz / b

                qr_loc = qr_perturbation(qr_parcels, del_x, del_z)
                Ds = sample_mass_weighted(Nr_parcels, qr_loc, rho_water, rho_air, mu, multiplicity)

                print(
                    f"Parcel at shift ({sx:.1f}, {sz:.1f}) has qr={qr_loc:.6e} "
                    f"and diameters: {Ds}"
                )

                for d in Ds:
                    x_list.append(centre[0] + sx)
                    z_list.append(centre[1] + sz)
                    D_list.append(d)
                    qr_list.append(qr_loc / multiplicity)
                    qr_location_list.append(qr_loc)

    # convert to arrays shaped (time, n_parcels)
    n_parcels = len(x_list)
    x_array = np.array(x_list, dtype=float).reshape((1, n_parcels))
    z_array = np.array(z_list, dtype=float).reshape((1, n_parcels))
    D_array = np.array(D_list, dtype=float)
    qr_array = np.array(qr_list, dtype=float).reshape((1, n_parcels))
    qr_location_array = np.array(qr_location_list, dtype=float).reshape((1, n_parcels))

    parcel_volume = dx_parcel * dx_parcel
    volume_array = np.ones((1, n_parcels)) * parcel_volume

    Nr_array = equivalent_number_concentration(D_array, qr_location_array[0, :], rho_water, rho_air) / multiplicity
    Nr_array = Nr_array.reshape((1, n_parcels))

    time = np.array([0.0])
    coords = {
        "time": ("time", time, {"units": "seconds since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"}),
        "n_parcels": ("n_parcels", np.arange(1, n_parcels + 1, dtype=np.int32)),
    }

    ds = xr.Dataset(
        {
            "x_position": xr.DataArray(
                x_array,
                dims=["time", "n_parcels"],
                coords=coords,
                attrs={"units": "m", "long_name": "x position component"},
            ),
            "z_position": xr.DataArray(
                z_array,
                dims=["time", "n_parcels"],
                coords=coords,
                attrs={"units": "m", "long_name": "z position component"},
            ),
            "volume": xr.DataArray(
                volume_array,
                dims=["time", "n_parcels"],
                coords=coords,
                attrs={"units": "m^2", "long_name": "parcel volume"},
            ),
            "qr": xr.DataArray(
                qr_array,
                dims=["time", "n_parcels"],
                coords=coords,
                attrs={"units": "kg/kg", "long_name": "rain mixing ratio"},
            ),
            "Nr": xr.DataArray(
                Nr_array,
                dims=["time", "n_parcels"],
                coords=coords,
                attrs={"units": "1/kg", "long_name": "rain number concentration"},
            ),
        }
    )

    outname = "1g60000_rain_percentiles_sym_160_perturbed.nc"
    ds.to_netcdf(outname, unlimited_dims=["time"])
    print(f"Wrote {outname} with {n_parcels} parcels ({multiplicity} per location).")

    # Visualize the qr distribution spatially and as a histogram.
    try:
        import matplotlib.pyplot as plt

        x_vals = x_array[0, :]
        z_vals = z_array[0, :]
        qr_vals = qr_location_array[0, :]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sc = axes[0].scatter(x_vals, z_vals, c=qr_vals, s=8, cmap="viridis", alpha=0.9)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x position (m)")
        axes[0].set_ylabel("z position (m)")
        axes[0].set_title("Spatial qr perturbation")
        cbar = fig.colorbar(sc, ax=axes[0])
        cbar.set_label("qr (kg/kg)")

        axes[1].hist(qr_vals, bins=40, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("qr (kg/kg)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("qr distribution")

        fig.tight_layout()
        fig.savefig("qr_distribution.png", dpi=160)
        plt.show()
        print("Saved qr_distribution.png")
    except Exception as _err:
        print(f"Plotting skipped: {_err}")


if __name__ == "__main__":
    main()
