import numpy as np

def scf_xtpl_helgaker_2(functionname: str, zLO: int, valueLO, zHI, valueHI, verbose: int = 1, alpha = None):
    r"""Extrapolation scheme using exponential form for reference energies with two adjacent
    zeta-level bases. Used by :py:func:`~psi4.driver.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component (e.g., 'HF') used in summary printing.
    zLO
        Zeta number of the smaller basis set in 2-point extrapolation.
    valueLO
        Energy, gradient, or Hessian value at the smaller basis set in 2-point
        extrapolation.
    zHI
        Zeta number of the larger basis set in 2-point extrapolation.
        Must be `zLO + 1`.
    valueHI
        Energy, gradient, or Hessian value at the larger basis set in 2-point
        extrapolation.
    verbose
        Controls volume of printing.
    alpha
        Fitted 2-point parameter. Overrides the default :math:`\alpha = 1.63`

    Returns
    -------
    float or ~numpy.ndarray
        Eponymous function applied to input zetas and values; type from `valueLO`.

    Notes
    -----
    The extrapolation is calculated according to [1]_:
    :math:`E_{total}^X = E_{total}^{\infty} + \beta e^{-\alpha X}, \alpha = 1.63`

    References
    ----------

    .. [1] Halkier, Helgaker, Jorgensen, Klopper, & Olsen, Chem. Phys. Lett. 302 (1999) 437-446,
       DOI: 10.1016/S0009-2614(99)00179-7

    Examples
    --------
    >>> # [1] Hartree-Fock extrapolation
    >>> psi4.energy('cbs', scf_wfn='hf', scf_basis='cc-pV[DT]Z', scf_scheme='scf_xtpl_helgaker_2')

    """

    if type(valueLO) != type(valueHI):
        raise ValidationError(
            f"scf_xtpl_helgaker_2: Inputs must be of the same datatype! ({type(valueLO)}, {type(valueHI)})")

    if alpha is None:
        alpha = 1.63

    beta_division = 1 / (np.exp(-1 * alpha * zLO) * (np.exp(-1 * alpha) - 1))
    beta_mult = np.exp(-1 * alpha * zHI)

    if isinstance(valueLO, float):
        beta = (valueHI - valueLO) * beta_division
        value = valueHI - beta * beta_mult

        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = ''
            cbsscheme += """\n   ==> Helgaker 2-point exponential SCF extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
            cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
            cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % (alpha)
            cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n\n""" % (beta)

            #name_str = "%s/(%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zHI].upper())
            cbsscheme += """   @Extrapolated """
            #cbsscheme += name_str + ':'
            #cbsscheme += " " * (18 - len(name_str))
            cbsscheme += """% 16.12f\n\n""" % value
            #core.print_out(cbsscheme)
            print(cbsscheme)
            #logger.debug(cbsscheme)

        return value

    elif isinstance(valueLO, np.ndarray):

        beta = (valueHI - valueLO) * beta_division
        value = valueHI - beta * beta_mult

        if verbose > 2:
            cbsscheme = f"""\n   ==> Helgaker 2-point exponential SCF extrapolation for method: {functionname.upper()} <==\n"""
            cbsscheme += f"""\n   LO-zeta ({zLO}) Data\n"""
            cbsscheme += nppp(valueLO)
            cbsscheme += f"""\n   HI-zeta ({zHI}) Data\n"""
            cbsscheme += nppp(valueHI)

            cbsscheme += f"""\n   Alpha (exponent) Value:          {alpha:16.8f}"""
            cbsscheme += f"""\n   Beta Data\n"""
            cbsscheme += nppp(beta)
            cbsscheme += f"""\n   Extrapolated Data\n"""
            cbsscheme += nppp(value)
            cbsscheme += "\n"
            core.print_out(cbsscheme)
            logger.debug(cbsscheme)

        return value

    else:
        raise ValidationError(f"scf_xtpl_helgaker_2: datatype is not recognized '{type(valueLO)}'.")


def scf_xtpl_helgaker_3(functionname, zLO, valueLO, zMD, valueMD, zHI, valueHI, verbose=True, alpha=None):
    r"""Extrapolation scheme for reference energies with three adjacent zeta-level bases.
    Used by :py:func:`~psi4.cbs`.
    Parameters
    ----------
    functionname : str
        Name of the CBS component.
    zLO : int
        Lower zeta level.
    valueLO : float
        Lower value used for extrapolation.
    zMD : int
        Intermediate zeta level. Should be equal to zLO + 1.
    valueMD : float
        Intermediate value used for extrapolation.
    zHI : int
        Higher zeta level. Should be equal to zLO + 2.
    valueHI : float
        Higher value used for extrapolation.
    alpha : float, optional
        Not used.
    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.
    Notes
    -----
    The extrapolation is calculated according to [4]_:
    :math:`E_{total}^X = E_{total}^{\infty} + \beta e^{-\alpha X}, \alpha = 3.0`
    References
    ----------
    .. [4] Halkier, Helgaker, Jorgensen, Klopper, & Olsen, Chem. Phys. Lett. 302 (1999) 437-446,
       DOI: 10.1016/S0009-2614(99)00179-7
    """

    if (type(valueLO) != type(valueMD)) or (type(valueMD) != type(valueHI)):
        raise ValidationError("scf_xtpl_helgaker_3: Inputs must be of the same datatype! (%s, %s, %s)" %
                              (type(valueLO), type(valueMD), type(valueHI)))

   

    ratio = (valueHI - valueMD) / (valueMD - valueLO)
    alpha = -1 * np.log(ratio)
    beta = (valueHI - valueMD) / (np.exp(-1 * alpha * zMD) * (ratio - 1))
    value = valueHI - beta * np.exp(-1 * alpha * zHI)

    if verbose:
        # Output string with extrapolation parameters
        cbsscheme = ''
        cbsscheme += """\n   ==> Helgaker 3-point SCF extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
        cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
        cbsscheme += """   MD-zeta (%s) Energy:               % 16.12f\n""" % (str(zMD), valueMD)
        cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
        cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % (alpha)
        cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n\n""" % (beta)

        #name_str = "%s/(%s,%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zMD].upper(),
        #                                  _zeta_val2sym[zHI].upper())
        cbsscheme += """   @Extrapolated """
        #cbsscheme += name_str + ':'
        #cbsscheme += " " * (18 - len(name_str))
        cbsscheme += """% 16.12f\n\n""" % value
        #core.print_out(cbsscheme)
        print(cbsscheme)
    return value


def corl_xtpl_helgaker_2(functionname, zLO, valueLO, zHI, valueHI, verbose=True, alpha=None):
    r"""Extrapolation scheme for correlation energies with two adjacent zeta-level bases.
    Used by :py:func:`~psi4.cbs`.
    Parameters
    ----------
    functionname : str
        Name of the CBS component.
    zLO : int
        Lower zeta level.
    valueLO : float
        Lower value used for extrapolation.
    zHI : int
        Higher zeta level. Should be equal to zLO + 1.
    valueHI : float
        Higher value used for extrapolation.
    alpha : float, optional
        Overrides the default :math:`\alpha = 3.0`
    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.
    Notes
    -----
    The extrapolation is calculated according to [5]_:
    :math:`E_{corl}^X = E_{corl}^{\infty} + \beta X^{-alpha}`
    References
    ----------
    .. [5] Halkier, Helgaker, Jorgensen, Klopper, Koch, Olsen, & Wilson,
       Chem. Phys. Lett. 286 (1998) 243-252,
       DOI: 10.1016/S0009-2614(99)00179-7
    """
    import numpy as np
    if type(valueLO) != type(valueHI):
        raise ValidationError(
            "corl_xtpl_helgaker_2: Inputs must be of the same datatype! (%s, %s)" % (type(valueLO), type(valueHI)))

    if alpha is None:
        alpha = 3.0

 
    value = (valueHI * zHI**alpha - valueLO * zLO**alpha) / (zHI**alpha - zLO**alpha)
    beta = (valueHI - valueLO) / (zHI**(-alpha) - zLO**(-alpha))

    final = value
    if verbose:
            # Output string with extrapolation parameters
        cbsscheme = """\n\n   ==> Helgaker 2-point correlated extrapolation for method: %s <==\n\n""" % (
        functionname.upper())
            #            cbsscheme += """   HI-zeta (%1s) SCF Energy:           % 16.12f\n""" % (str(zHI), valueSCF)
        cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
        cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
        cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % alpha
        cbsscheme += """   Extrapolated Energy:              % 16.12f\n\n""" % value
            #cbsscheme += """   LO-zeta (%s) Correlation Energy:   % 16.12f\n""" % (str(zLO), valueLO)
            #cbsscheme += """   HI-zeta (%s) Correlation Energy:   % 16.12f\n""" % (str(zHI), valueHI)
            #cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n""" % beta
            #cbsscheme += """   Extrapolated Correlation Energy:  % 16.12f\n\n""" % value

        #name_str = "%s/(%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zHI].upper())
        cbsscheme += """   @Extrapolated """
        #cbsscheme += name_str + ':'
        #cbsscheme += " " * (19 - len(name_str))
        cbsscheme += """% 16.12f\n\n""" % final
        #core.print_out(cbsscheme)
        print(cbsscheme)
    return final
