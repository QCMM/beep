import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Union, Any, List, Tuple, Callable
import pandas as pd

from .logging_utils import padded_log
from .plotting_utils import zpve_plot


def gauss(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    """
    Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    A : float
        Amplitude of the Gaussian.
    mu : float
        Mean of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray
        Gaussian evaluated at x.
    """
    return A * np.exp(-(x - mu) ** 2. / (2. * sigma ** 2.))


def gauss_fitting(nbins: int, data: np.ndarray, p0: List[float], logger: logging.Logger, nboot: int = 10000) -> List[float]:
    """
    Perform Gaussian fitting using bootstrap resampling.

    Parameters
    ----------
    nbins : int
        Number of bins for the histogram.
    data : np.ndarray
        Data to fit.
    p0 : List[float]
        Initial guesses for the Gaussian parameters [A, mu, sigma].
    logger : logging.Logger
        Logger for info messages.
    nboot : int, optional
        Number of bootstrap iterations (default is 10000).

    Returns
    -------
    List[float]
        Best fit parameters [A, mu, sigma].
    """
    # Histogram of experimental data
    ydata, bin_edges = np.histogram(data, bins=nbins)

    # Poisson error
    err = np.sqrt(ydata)

    # Bin centers
    xbin = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    xbin = xbin.astype("float64")

    # Initialize arrays to store fit variables for each bootstrap iteration
    sigma_boot = np.zeros(nboot)
    mu_boot = np.zeros(nboot)
    a_boot = np.zeros(nboot)

    logger.info("Starting the curve fit within the Poisson error")

    for i in range(nboot):
        # Randomize values with Poisson error
        rdata = np.random.normal(loc=ydata, scale=err, size=len(ydata))
        # Fit randomized values
        coeff, _ = curve_fit(gauss, xbin, rdata, p0=p0)
        # Store coefficients
        a_boot[i], mu_boot[i], sigma_boot[i] = coeff

    # Fit stored coefficients assuming their distribution is Gaussian
    logger.info("Fitting the Gaussian parameters A, mu, and sigma")

    vbest = []
    labels = ["A", "mu", "sigma"]
    plt.figure(figsize=(16, 4))

    p0_param = [[1500, p0[0], 9.0], [1500, p0[1], 300], [1500.0, p0[2], 30]]
    param_list = [a_boot, mu_boot, sigma_boot]

    for i, vdata in enumerate(param_list):
        vydata, vedges = np.histogram(vdata, bins=30)
        vxbin = (vedges[1:] + vedges[:-1]) / 2.0
        vcoef, _ = curve_fit(gauss, vxbin, vydata, p0=p0_param[i])
        vbest.append(vcoef[1])

        # Plot the distribution of the coefficients
        xdata = np.linspace(min(vxbin), max(vxbin), 100)
        zdata = gauss(xdata, *vcoef)
        plt.subplot(1, 4, i + 1)
        plt.hist(vdata, bins=30)
        plt.plot(xdata, zdata)

    # Check if bootstrap fits the original data properly
    xdata = np.linspace(min(xbin), max(xbin), 100)
    plt.subplot(1, 4, 4)
    plt.hist(data, bins=nbins, color='g', alpha=0.6)
    plt.bar(xbin, ydata, width=0, yerr=err, color='b')
    plt.plot(xdata, gauss(xdata, *vbest), color='k')

    logger.info(f"The best fit is: A: {vbest[0]} mu: {vbest[1]} sigma: {vbest[2]}")

    return vbest


def apply_lin_models(df_be, df_be_zpve, meth_fit_dict, be_methods, basis, mol, be_range,
                     generate_plots=False):
    """Apply linear ZPVE correction models to binding energies."""
    logger = logging.getLogger("beep")
    lin_zpve_df = pd.DataFrame()

    gear = "\u2699"
    padded_log(logger, "Applying linear model to correct for ZPVE", padding_char=gear)
    for method, factors in meth_fit_dict.items():
        if method == "Mean":
            continue  # handled below
        m, n, R2 = factors
        column_name = f"{method}/{basis}"
        zpve_column_name = f"{column_name}+ZPVE"

        logger.info(f"Applying linear model to {column_name} BEs")
        if column_name in df_be.columns and zpve_column_name in df_be_zpve.columns:
            scaled_column_name = f"{column_name}_lin_ZPVE"
            lin_zpve_df[scaled_column_name] = df_be[column_name] * m + n

            common_indices = df_be.index.intersection(df_be_zpve.index)
            df_be_filtered = df_be.loc[common_indices]
            df_be_zpve_filtered = df_be_zpve.loc[common_indices]

            x = df_be_filtered[column_name].to_numpy(dtype=float)
            y = df_be_zpve_filtered[zpve_column_name].to_numpy(dtype=float)

            if generate_plots:
                logger.info(
                    f"Creating BE vs BE + \u0394ZPVE plot for {column_name} saving as {mol}/zpve_{mol}_{method}.svg"
                )
                fig = zpve_plot(x, y, [m, n, R2])
                fig.savefig(f"{mol}/zpve_{mol}_{method}.svg")
                plt.close(fig)
        else:
            raise KeyError(
                f"Column {column_name} or {zpve_column_name} not present in the BE or ZPVE dataframe"
            )

    lin_zpve_df["Mean_Eb_all_dft"] = lin_zpve_df.mean(axis=1)
    lin_zpve_df["StdDev_all_dft"] = lin_zpve_df.std(axis=1)

    # Universal model: apply mean linear fit to the mean uncorrected BE
    if "Mean" in meth_fit_dict:
        m, n, R2 = meth_fit_dict["Mean"]
        lin_zpve_df["Mean_lin_ZPVE"] = df_be["Mean_Eb_all_dft"] * m + n
        logger.info(f"Applied universal mean linear model: Mean_BE+\u0394ZPVE = {m:.6f} * Mean_BE + {n:.6f} (R\u00b2={R2:.6f})")

        if generate_plots:
            common_indices = df_be.index.intersection(df_be_zpve.index)
            x = df_be.loc[common_indices, "Mean_Eb_all_dft"].to_numpy(dtype=float)
            y = df_be_zpve.loc[common_indices, "Mean_Eb_all_dft"].to_numpy(dtype=float)
            logger.info(f"Creating universal mean BE vs BE + \u0394ZPVE plot saving as {mol}/zpve_{mol}_Mean.svg")
            fig = zpve_plot(x, y, [m, n, R2])
            fig.savefig(f"{mol}/zpve_{mol}_Mean.svg")
            plt.close(fig)
    lin_zpve_df = lin_zpve_df[
        (lin_zpve_df["Mean_Eb_all_dft"] >= be_range[1])
        & (lin_zpve_df["Mean_Eb_all_dft"] <= be_range[0])
    ]

    return lin_zpve_df


def calculate_mean_std(df_res, mol, logger):
    """Calculate mean and std rows for a binding energy DataFrame."""
    df_with_stats = df_res.copy()
    data_only_df = df_res.loc[~df_res.index.str.startswith(("Mean_", "StdDev_"))]

    mean_row = data_only_df.mean()
    std_row = data_only_df.std()

    stddev_values = data_only_df["StdDev_all_dft"].values
    sem = np.sqrt((1 / len(stddev_values) ** 2) * np.sum(stddev_values**2))

    std_row["StdDev_all_dft"] = sem

    df_with_stats.loc[f"Mean_{mol}"] = mean_row
    df_with_stats.loc[f"StdDev_{mol}"] = std_row

    mean_val = df_with_stats.loc[f"Mean_{mol}", "Mean_Eb_all_dft"]
    std_val = df_with_stats.loc[f"StdDev_{mol}", "StdDev_all_dft"]

    return df_with_stats, mean_val, std_val
