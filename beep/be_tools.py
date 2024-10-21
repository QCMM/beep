import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Union, Any, List, Tuple, Callable
import pandas as pd
import os,sys
from functools import wraps
import logging


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



