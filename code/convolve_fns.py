#!/usr/bin/env python3
# coding: utf-8
"""
convolve_fns.py

Code to convolve ECCOv4r4 adjoint sensitivities with other 2-D and 3D fields.

Required to reproduce data for Boland et al. 2025 (in prep)
See https://github.com/emmomp/CANARI_FWTRANS for details

Updated Feb 2025

@author: emmomp@bas.ac.uk Emma J. D. Boland"""

import os
import sys
import numpy as np
import xarray as xr
import pyresample
from scipy.ndimage import gaussian_filter
import xgcm

sys.path.insert(0, "/users/emmomp/Python/ECCOv4-py")
import ecco_v4_py as ecco


def convolve_ecco(
    ecco_data,
    convolve_data,
    ecco_convolve_map,
    regrid=True,
    ecco_grid=None,
    dir_out=None,
    smooth=False,
    overwrite=False,
    attrs=None,
    **kwargs,
):
    """
    Function to convolve ECCO adjoint sensitivities with given fields to produce convolutions
    ECCO data regridded to convolve data if regrid=True, and, if convolve data is time-dependent, the sensetivities are resampled in time to match

    Parameters
    ----------
    ecco_data : xarray dataset
      ECCO sensitivities to convolve
    convolve_data : xarray dataset
      Fields to convolve with ECCO data
    ecco_convolve_map : dict
      Mapping from each ecco_data variable to convolve_data variable or list of variables. Used to match variables to convolve together
    regrid : logical
      If True (default), regrid ecco_data to convolve_data grid, must have variables 'lon' and 'lat'
    ecco_grid : xarray dataset
      ECCOv4 grid, required if regrid=True or smooth=True, must contain XC and YC
    dir_out : str
      Directory to write data_out to. If None (default), no data written
    smooth : logical (default False)
      If true, smooth adjoint fields according to params smooth_dict before convolving
    overwrite : logical (default False)
      If true, overwrite existing output files
    attrs : dict
      Attributes for output netcdfs
    kwargs : passed to apply_smoothing function

    Returns
    -------
    data_all : xarray dataset
      Convolved data, returned as variables with names '[ecco_var]X[convolve_var]',
      where 'ecco_var' and 'convolve_var' are variables from ecco_data and convolve_data respectively
    """

    if regrid:
        regridder = setup_regrid(
            ecco_grid.XC, ecco_grid.YC, convolve_data.lon, convolve_data.lat
        )

    data_all = [] # Collect data for return
    year = ecco_data.year.data

    for var, conv_var in ecco_convolve_map.items():
        data_all_var = []
        if dir_out is not None:
            if len(conv_var) > 1:
                ftest = []
                for cvar in conv_var:
                    fout = dir_out + f"/ECCOconv_{var}_{cvar}.nc"
                    ftest.append(os.path.isfile(fout))
                if all(ftest):
                    if not overwrite:
                        print(f"Found all {var} files, skipping")
                        continue
            else:
                fout = dir_out + f"/ECCOconv_{var}.nc"
                if os.path.isfile(fout):
                    if not overwrite:
                        print(f"Found {var} file, skipping")
                        continue

        print(var)
        # Select sensitivity for transp and year
        sens = ecco_data[var]
        # Smooth if required
        if smooth:
            ecco_xgcm_grid = ecco.get_llc_grid(ecco_grid)
            sens = apply_smoothing(sens, ecco_xgcm_grid, **kwargs)
        # Regrid in space to match convolve_data if necessary
        if regrid:
            sens = repeat_regrid(sens, regridder, convolve_data.lon, convolve_data.lat)
            if "year" in ecco_data["time"].dims:
                sens["time"] = ecco_data["time"].sel(year=year)
            else:
                sens["time"] = ecco_data["time"]

        data_out = None
        for cvar in ecco_convolve_map[var]:
            if cvar not in convolve_data:
                continue
            print(cvar)
            if dir_out is not None:
                fout = dir_out + f"/ECCOconv_{var}_{cvar}.nc"
                if os.path.isfile(fout):
                    if not overwrite:
                        print(f"Found {var} file, skipping")
                        continue

            # Resample in time to match convolve_data if necessary
            if "time" in convolve_data:
                if convolve_data[cvar]["time"].size > 1:
                    if "time" not in sens.dims:
                        sens = sens.swap_dims({"lag_years": "time"})
                    cdata = convolve_data[cvar].interp(time=sens["time"])
                    data_convolve = sens * cdata
                    data_convolve = data_convolve.swap_dims({"time": "lag_years"})
                    data_convolve = data_convolve.dropna("lag_years", how="all")
                else:
                    data_convolve = sens * convolve_data[cvar]
            elif "month" in convolve_data:
                if "time" not in sens.dims:
                    sens = sens.swap_dims({"lag_years": "time"})
                sens = sens.groupby(sens.time.dt.month)
                cdata = convolve_data[cvar]
                data_convolve = sens * cdata
                data_convolve = data_convolve.swap_dims({"time": "lag_years"})
                data_convolve = data_convolve.dropna("lag_years", how="all")

            else:
                # Perform convolution
                data_convolve = sens * convolve_data[cvar]

            # Take sums (sensitivities already area weighted)
            dims = list(data_convolve.dims)
            dims.remove("lag_years")
            if "model" in dims:
                dims.remove("model")
            data_convolve_sum = data_convolve.sum(dim=dims)
            data_convolve_abssum = np.abs(data_convolve).sum(dim=dims)
            data_convolve_square_sum = (data_convolve**2).sum(dim=dims)

            # Put in a dataset
            out_dict = {
                f"{var}X{cvar}": data_convolve,
                f"{var}X{cvar}_sum": data_convolve_sum,
                f"{var}X{cvar}_abssum": data_convolve_abssum,
                f"{var}X{cvar}_squaresum": data_convolve_square_sum,
            }
            data_out = xr.Dataset(data_vars=out_dict)

            data_out["year"] = (
                "year",
                [
                    year,
                ],
            )

            data_all_var.append(data_out)
            if dir_out:
                if attrs:
                    data_out.attrs.update(attrs)
                data_out.to_netcdf(fout)
                print("Written to", fout)

        data_all_var = xr.merge(data_all_var)
        data_all.append(data_all_var)

    data_all = xr.merge(data_all)
    return data_all


def setup_regrid(
    xc, yc, new_lon=np.linspace(-179, 180, 360), new_lat=np.linspace(-89.5, 89.5, 180)
):
    """
    Function to produce regridding kernel for repeated regrid operations

    Parameters
    ----------
    xc, yc : xarray dataarrays
        original grid lon and lats
    new_lon, new_lat : numpy arrays
        new grid, default to one degree global

    Returns
    ----------
    resample_data : list
        regridding kernel data
    """
    orig_grid = pyresample.geometry.SwathDefinition(
        lons=xc.values.ravel(), lats=yc.values.ravel()
    )
    yi, xi = np.meshgrid(new_lat, new_lon)
    new_grid = pyresample.geometry.GridDefinition(lons=xi, lats=yi)
    resample_data = pyresample.kd_tree.get_neighbour_info(
        orig_grid, new_grid, 100000, neighbours=1
    )
    return resample_data


def repeat_regrid(
    ds,
    resample_data,
    new_lon=np.linspace(-179, 180, 360),
    new_lat=np.linspace(-89.5, 89.5, 180),
    loop_dim="lag_years",
):
    """
    Function to carry out repeated regrid operations over a given dimension,
    based on a provided regridding kernel

    Parameters
    ----------
    ds : xarray dataarray
        data to be regridded
    resample_data : list
        regridding kernel info, generated by setup_regrid
    new_lon, new_lat : numpy arrays
        new grid information for regridded coordinates, default 1 degree global
    loop_dim : str
        dimension along which to repeat regrid operation, default 'lag_years'

    Returns
    ----------
    ds2 : xarray dataarray
        regridded version of input data, ds
    """
    grid_shape = [new_lon.size, new_lat.size]
    stack_dims = ds.dims[1:]
    ds1 = pyresample.kd_tree.get_sample_from_neighbour_info(
        "nn",
        grid_shape,
        ds.stack(z=stack_dims).transpose(..., loop_dim).values,
        resample_data[0],
        resample_data[1],
        resample_data[2],
    )
    ds2 = xr.DataArray(
        ds1,
        dims=["lon", "lat", loop_dim],
        coords={
            "lon": (("lon"), new_lon.data),
            "lat": (("lat"), new_lat.data),
            loop_dim: (loop_dim, ds[loop_dim].data),
        },
    )
    return ds2


def apply_smoothing(da, ecco_xgcm_grid, pad=3, sigma=1):
    """
    Function to apply gaussian smoothing to a data on the ECCO grid

    Parameters
    ----------
    da : xarray dataarray
        data to be smoothed
    ecco_xgcm_grid : dict
        dictionary containing xgcm information about the ECCO grid, produced by ecco.get_llc_grid(ecco_grid)
    pad : int
        how many grid cells to pad before applying smoothing, default 3
    sigma : int
        width of gaussian smoothing kernel in grid cells, default 1

    Returns
    ----------
    da_cut : xarray dataarray
        smoothed version of input data, da
    """
    # pad tiles
    da_pad = xgcm.padding.pad(
        da, ecco_xgcm_grid, {"X": (pad, pad), "Y": (pad, pad)}, "extend"
    )
    # smooth with gaussian
    da_pad_smooth = xr.DataArray(
        data=gaussian_filter(da_pad, sigma=sigma, mode="reflect", radius=[0, 0, 1, 1]),
        dims=da_pad.dims,
    )
    # cut off padding and add back co-ords
    da_cut = (
        da_pad_smooth.isel(i=slice(pad, -pad), j=slice(pad, -pad))
        .assign_coords(da.coords)
        .transpose("time", "tile", "j", "i")
    )
    # zero out land from overlap
    return da_cut
