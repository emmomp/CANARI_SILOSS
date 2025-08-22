#!/usr/bin/env python3
# coding: utf-8
"""
utils.py

Useful functions for analysis.

Required to reproduce data for Boland et al. 2025 (in prep)
See https://github.com/emmomp/CANARI_FWTRANS for details

Updated Feb 2025

@author: emmomp@bas.ac.uk Emma J D Boland
"""
import glob
import sys
import dask
import pandas as pd
from scipy import stats
from scipy import signal
import xarray as xr
import numpy as np
sys.path.insert(0, "/users/emmomp/Python/ECCOv4-py")
import ecco_v4_py as ecco


def plot_ecco(ecco_grid, dplot, cmap="RdBu_r", show_colorbar=True, **kwargs):
    """
    Wrapper for ecco_v4_py plot_proj_to_latlon_grid

    Parameters
    ----------
    ecco_grid : xarray dataset
        Contains coords XC and YC
    dplot : xarray dataarray
        data to plot
    cmap : str, optional
        colormap, default "RdBu_r"
    show_colorbar: logical, optional
        default True
    **kwargs : passed to ecco_v4_py.plot_proj_to_latlon_grid

    Returns
    -------
    [f, ax, p] : figure, axis and plot object handles

    """
    [f, ax, p] = ecco.plot_proj_to_latlon_grid(
        ecco_grid.XC,
        ecco_grid.YC,
        dplot,
        show_colorbar=show_colorbar,
        cmap=cmap,
        **kwargs,
    )[:3]
    return [f, ax, p]


def get_soln(fcname, expdir, year0=1992):
    """
    Load a cost function in mitgcm meta/data format

    Parameters
    ----------
    fcname : str
        Name of mitgcm cost function. Should be string following 'm_' in gencost_barfile, defined in data.ecco
    expdir : str
        directory where data is held
    year0 : int
        Start year of run, default 1992

    Returns
    -------
    fc : xarray datarray
        cost function time series
    """
    metafile=f"m_{fcname}.{129:010g}.meta"
    datafile=f"m_{fcname}.{129:010g}.data"
    with open(f'{expdir}/{metafile}') as f:
        fcmeta = f.read().splitlines() 
    nt=int(fcmeta[-2].split('[')[-1].split(']')[0])
    fcdata = ecco.read_llc_to_tiles(
        f"{expdir}", datafile, use_xmitgcm=True, nl=nt
    )
    ds_fc = xr.DataArray(data=fcdata, dims=["time", "tile", "j", "i"])
    ds_fc.name = fcname
    fc = ds_fc.sum(["tile", "j", "i"])

    dates = pd.date_range(start=f'{year0}-01-01',periods=nt,freq='MS')+np.timedelta64(15,'D')
    fc = fc.assign_coords({'time':('time',dates)})
    fc = fc.assign_coords({'year':("time", fc.time.dt.year.data)})
    return fc
    
def soln_anoms(fc):
    """
    Take climatological anomalies of cost function, return as single and 12 monthly time series

    Parameters
    ----------
    fc : xarray datarray
        cost function time series

    Returns
    -------
    fc_climanom : xarray dataarray
        climatological anomaly of cost function time series
    fc_mth : xarray dataarray
        climatological anomaly of cost function as monthly time series
    """    
    fc = fc.sel(time=slice("1996-01-01", None))
    fc = fc.copy(data=signal.detrend(fc))

    fc_climanom = (
        fc.groupby(fc.time.dt.month)
        - fc.groupby(fc.time.dt.month).mean(dim="time").compute()
    )

    fc_mth = []
    fc_climanom_group = fc_climanom.groupby(fc_climanom.time.dt.month)
    for mth in fc_climanom_group:
        month = mth[1]
        month = month.assign_coords({"month": month.month[0].data})
        fc_mth.append(month.swap_dims({"time": "year"}))
    fc_mth = xr.concat(fc_mth, "month")

    return fc_climanom, fc_mth


##Create 4th order Bworth filter
def butter_lowpass(data, cut, order=4, sample_freq=1):
    """
    A wrapper for scipy butterworth filter
    """
    pass_freq = 1.0 / cut
    sos = signal.butter(order, pass_freq, "low", output="sos", fs=sample_freq)
    filt = signal.sosfiltfilt(sos, data)
    return filt


def butter_ufunc(data, cut, tdim, order=4, sample_freq=1):
    """
    Turn butter_lowpass into an xarray ufunc that can loop over tdim
    """
    filt = xr.apply_ufunc(
        butter_lowpass,
        data,
        cut,
        order,
        sample_freq,
        input_core_dims=[[tdim], [], [], []],
        output_core_dims=[[tdim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )
    return filt


def pcorr(fc, dJ):
    """
    A wrapper for scipy pearson r correlation
    """
    [r, p] = stats.pearsonr(fc, dJ)
    return np.array([r**2, p])


def pcorr_ufunc(fc, dJ, loopdim="year"):
    """
    Turn pcorr into an xarray ufunc that can loop over tdim
    """
    pstats = xr.apply_ufunc(
        pcorr,
        fc,
        dJ,
        input_core_dims=[
            [
                loopdim,
            ],
            [
                loopdim,
            ],
        ],
        output_core_dims=[
            [
                "stats",
            ]
        ],
        dask_gufunc_kwargs={"output_sizes": {"stats": 2}},
        dask="parallelized",
        output_dtypes="float64",
        vectorize=True,
    )
    pstats["stats"] = ["r2", "p"]
    return pstats


def calc_tseries(data, masks=None, var_list=None, weight=None):
    """
    Calculates time series of provided fields

    Parameters
    ----------
    data : xarray with spatial fields to be summed
    masks : dict, optional
        Dictionary of masknames and masks to apply to data before taking mean, defaults to global
    var_list : list, optional
        List of variables to calculate timeseries for

    Returns
    -------
    xarray dataset containing the sum and wsum of the absolute values of the variables in var_list, in the regions specified in the dictionary masks

    """

    if masks:
        ds_out = []
        for mask in masks.keys():
            if weight is None:        
                ds_masked = data.where(masks[mask])
                ds_masked_abs= np.abs(data.where(masks[mask]))
            else:
                ds_masked= data.where(masks[mask]).weighted(weight)
                ds_masked_abs= np.abs(data.where(masks[mask])).weighted(weight)
            ad_mean = ds_masked.sum(dim=["i", "j", "tile"]).assign_coords({"stat": "sum"})
            ad_absmean = (
                ds_masked_abs
                .sum(dim=["i", "j", "tile"])
                .assign_coords({"stat": "abssum"})
            )
            ds_mask = xr.concat([ad_mean, ad_absmean], "stat").assign_coords({"mask": mask})
            ds_out.append(ds_mask)
        ds_out = xr.concat(ds_out, "mask", coords="minimal")
    else:
        if weight:
            data=data.weighted(weight)
            data_abs=np.abs(data).weighted(weight)
        ad_mean = data.sum(dim=["i", "j", "tile"]).assign_coords({"stat": "sum"})
        ad_absmean = (
                    data_abs
                    .sum(dim=["i", "j", "tile"])
                    .assign_coords({"stat": "abssum"})
                )
        ds_out=xr.concat([ad_mean, ad_absmean], "stat")
    

    if var_list:
        return ds_out[var_list]
    else:
        return ds_out

def load_canari_masks():
    """
    Load Arctic and N Atlantic masks, including custom E Greenland mask
    """
    ecco_grid = xr.open_dataset(
        "~/data/orchestra/other_data/ECCO_r3_alt/ECCOv4r3_grid.nc"
    )
    masks = {}
    masks["global"] = ecco_grid.maskC.isel(k=0)
    for basin in ["atl", "arct", "hudson", "med", "north", "baffin", "gin", "barents"]:
        masks[basin] = ecco.get_basin_mask(basin, ecco_grid.maskC.isel(k=0))

    masks["natl"] = masks["atl"] * (ecco_grid.YC > 40)  # north atl
    # masks['arctplus']=masks['arct']+masks['baffin']+masks['hudson']
    masks["satl"] = masks["atl"] - masks["natl"]
   # masks["egland"] = xr.open_dataarray("../data_out/EGland_llcmask.nc")
    masks["gland"] = xr.open_dataarray("../other_data/greenlandsea_mask.nc")
    masks["natl"] = xr.where(masks["gland"] == 1, 0, masks["natl"])
    masks["norw"] = xr.where(masks["gland"] == 1, 0, masks["gin"])
    masks["gland"] = xr.where((masks["norw"] == 1)&(ecco_grid.XC < -17), 1, masks["gland"])
    masks["norw"] = xr.where(ecco_grid.XC < -17, 0, masks["norw"])
    masks["arct"] = xr.where(masks["gland"] == 1, 0, masks["arct"])
    my_masks = masks.copy()
    my_masks.pop('atl')
    my_masks.pop('gin')

    return my_masks


def UVline_from_UXVY(xfld, yfld, coords, section):
    """
    Rotate U and V velocities into along and across transect velocities
    """
    grid = ecco.get_llc_grid(coords)
    [section_pt1, section_pt2] = ecco.get_section_endpoints(section)
    _, maskW, maskS = ecco.get_section_line_masks(section_pt1, section_pt2, coords)
    dx = coords.dxC.where(maskW, drop=True).mean(dim="j").sum()
    dy = coords.dyC.where(maskS, drop=True).mean(dim="i").sum()
    dline = np.sqrt(dx**2 + dy**2)
    cosline = dx / dline
    sinline = dy / dline
    velc = grid.interp_2d_vector({"X": xfld, "Y": yfld}, boundary="fill")
    u_line = velc["X"] * cosline - velc["Y"] * sinline
    v_line = velc["X"] * cosline + velc["Y"] * sinline
    return u_line, v_line


def load_ecco_convs(conv_dir, eyear, var=None, exp=None):
    """
    Load convolutions of adjoint sensitivities produced by reconstruct.py
    """
    mth = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    mthi = dict(zip(mth, list(range(1, 13))))

    if exp:
        cexps=[exp,]
    else:
        cexps_full = glob.glob(f"{conv_dir}/transfw_*_7d_{eyear}")
        cexps = [exp.split("/")[-1] for exp in cexps_full]

    cexps_mdict = {}
    for exp in cexps:
        cexps_mdict[exp] = mthi[exp.split("/")[-1].split("_")[1]]
    cexps_edict = {m: k for k, m in cexps_mdict.items()}

    conv_ecco = []
    for exp in cexps:
        print(f"Loading {exp}")
        ds_exp = []
        for year in range(1996, 2018):
            if var:
                ds=xr.open_mfdataset(f"{conv_dir}/{exp}/{year}/*{var}.nc", coords="minimal")
            else:
                ds = xr.open_mfdataset(f"{conv_dir}/{exp}/{year}/*.nc", coords="minimal")
            ds_exp.append(ds)
        ds_exp = (
            xr.concat(ds_exp, "year")
            .assign_coords({"exp": exp, "month": cexps_mdict[exp]})
        )
        conv_ecco.append(ds_exp)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        conv_ecco = xr.concat(conv_ecco, "exp")
        conv_ecco = conv_ecco.sortby(conv_ecco.lag_years, ascending=False)

    plotdates = []
    for exp in cexps:
        plotdates.append(
            [
                np.datetime64(
                    f"{conv_ecco.year[i].data}-{cexps_mdict[exp]:02.0f}-16", "ns"
                )
                for i in range(0, 22)
            ]
        )
    conv_ecco = conv_ecco.assign_coords(dates=(["exp", "year"], plotdates))

    print("Done loading")

    return conv_ecco,cexps_mdict,cexps_edict

    
def load_ecco_convs_synth(conv_dir, eyear, var=None, exp=None):
    """
    Load convolutions of adjoint sensitivities produced by reconstruct.py
    """
    mth = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    mthi = dict(zip(mth, list(range(1, 13))))

    if exp:
        cexps=[exp,]
    else:
        cexps_full = glob.glob(f"{conv_dir}/transfw_???_noparam_7d_{eyear}")+glob.glob(f"{conv_dir}/transfw_???_noparam_7d_{eyear}_synth")
        cexps = [exp.split("/")[-1] for exp in cexps_full]

    cexps_mdict = {}
    for exp in cexps:
        cexps_mdict[exp] = mthi[exp.split("/")[-1].split("_")[1]]
    cexps_edict = {m: k for k, m in cexps_mdict.items()}

    conv_ecco = []
    for exp in cexps:
        print(f"Loading {exp}")
        ds_exp = []
        for year in range(1996, 2018):
            if var:
                ds=xr.open_mfdataset(f"{conv_dir}/{exp}/{year}/*{var}.nc", coords="minimal")
            else:
                ds = xr.open_mfdataset(f"{conv_dir}/{exp}/{year}/*.nc", coords="minimal")
            ds_exp.append(ds)
        ds_exp = (
            xr.concat(ds_exp, "year")
            .assign_coords({"exp": exp})
        )
        conv_ecco.append(ds_exp)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        conv_ecco = xr.concat(conv_ecco, "exp",coords='minimal',compat='override')
        conv_ecco = conv_ecco.sortby(conv_ecco.lag_years, ascending=False)

    plotdates = []
    for exp in cexps:
        plotdates.append(
            [
                np.datetime64(
                    f"{conv_ecco.year[i].data}-{cexps_mdict[exp]:02.0f}-16", "ns"
                )
                for i in range(0, 22)
            ]
        )
    conv_ecco = conv_ecco.assign_coords(dates=(["exp", "year"], plotdates))
    conv_ecco = conv_ecco.assign_coords(month=(["exp"],[cexps_mdict[exp] for exp in conv_ecco.exp.data]))
    
    print("Done loading")

    return conv_ecco,cexps_mdict,cexps_edict
