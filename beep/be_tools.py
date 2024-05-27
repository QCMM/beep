import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
from typing import Union
import pandas as pd


def _vibanal_wfn(hess: np.ndarray = None, irrep: Union[int, str] = None, molecule=None, energy=None, project_trans: bool = True, project_rot: bool = True, molden=False,name=None, lt=None):
    """Function to perform analysis of a hessian or hessian block, specifically...
    calling for and printing vibrational and thermochemical analysis, setting thermochemical variables,
    and writing the vibrec and normal mode files.
    Parameters
    ----------
    wfn
        The wavefunction which had its Hessian computed.
    hess
        Hessian to analyze, if not the hessian in wfn.
        (3*nat, 3*nat) non-mass-weighted Hessian in atomic units, [Eh/a0/a0].
    irrep
        The irrep for which frequencies are calculated. Thermochemical analysis is skipped if this is given,
        as only one symmetry block of the hessian has been computed.
    molecule : :py:class:`~psi4.core.Molecule` or qcdb.Molecule, optional
        The molecule to pull information from, if not the molecule in wfn. Must at least have similar
        geometry to the molecule in wfn.
    project_trans
        Should translations be projected in the harmonic analysis?
    project_rot
        Should rotations be projected in the harmonic analysis?
    Returns
    -------
    vibinfo : dict
        A dictionary of vibrational information. See :py:func:`~psi4.driver.qcdb.vib.harmonic_analysis`
    """

    from psi4.driver import qcdb
    from psi4 import core, geometry
    

    
    if hess is None:
        print("no hessian")
        
    else:
        nmwhess = hess
    
 
    m=molecule.to_string('xyz')
    mol = geometry(m)
    geom = np.asarray(mol.geometry())
    symbols = [mol.symbol(at) for at in range(mol.natom())]
    vibrec = {'molecule': mol.to_dict(np_out=False), 'hessian': nmwhess.tolist()}
    m = np.asarray([mol.mass(at) for at in range(mol.natom())])
    irrep_labels = mol.irrep_labels()
    print('***')
    vibinfo, vibtext = qcdb.vib.harmonic_analysis(
        nmwhess, geom, m, None, irrep_labels, dipder=None, project_trans=project_trans, project_rot=project_rot)
    
    #core.print_out(vibtext)
    #core.print_out(qcdb.vib.print_vibs(vibinfo, shortlong=True, normco='x', atom_lbl=symbols))

    if core.has_option_changed('THERMO', 'ROTATIONAL_SYMMETRY_NUMBER'):
        rsn = core.get_option('THERMO', 'ROTATIONAL_SYMMETRY_NUMBER')
    else:
        rsn = mol.rotational_symmetry_number()

    if irrep is None:
        therminfo, thermtext = qcdb.vib.thermo(
            vibinfo,
            T=core.get_option("THERMO", "T"),  # 298.15 [K]
            P=core.get_option("THERMO", "P"),  # 101325. [Pa]
            multiplicity=mol.multiplicity(),
            molecular_mass=np.sum(m),
            sigma=rsn,
            rotor_type=mol.rotor_type(),
            rot_const=np.asarray(mol.rotational_constants()),
            E0=energy) 
      
        core.set_variable("ZPVE", therminfo['ZPE_corr'].data)  # P::e THERMO
        core.set_variable("THERMAL ENERGY CORRECTION", therminfo['E_corr'].data)  # P::e THERMO
        core.set_variable("ENTHALPY CORRECTION", therminfo['H_corr'].data)  # P::e THERMO
        core.set_variable("GIBBS FREE ENERGY CORRECTION", therminfo['G_corr'].data)  # P::e THERMO

        core.set_variable("ZERO K ENTHALPY", therminfo['ZPE_tot'].data)  # P::e THERMO
        core.set_variable("THERMAL ENERGY", therminfo['E_tot'].data)  # P::e THERMO
        core.set_variable("ENTHALPY", therminfo['H_tot'].data)  # P::e THERMO
        core.set_variable("GIBBS FREE ENERGY", therminfo['G_tot'].data)  # P::e THERMO

    else:
        core.print_out('  Thermochemical analysis skipped for partial frequency calculation.\n')
    return vibinfo, therminfo



def zpve_correction(name_be, be_method, lot_opt, client, scale_factor = 1.0):
    import qcelemental as qcel

    ds_be = client.get_collection("ReactionDataset", name_be)
    df_all = ds_be.get_entries()
    df_nocp = df_all[df_all['stoichiometry'] == 'be_nocp']

    zpve_corr_dict = {}
    todelete = []

    for i in ds_be.get_index():
        mols = []
        zpve_list = []

        #if 'D3BJ' in be_method:
        #    rec_be_method = be_method.split('-')[0]
        #else:
        #    rec_be_method = be_method
    
        mol_list = df_nocp[df_nocp['name'] == i]['molecule']

        for mol in mol_list:
            try:
                r = client.query_results(driver='hessian', molecule=mol, method=lot_opt.split("_")[0], basis=lot_opt.split("_")[1], keywords = None )[0]
            except IndexError:
                print("Molecule {} does not have hessian yet".format(str(i)))
                continue
    
            h = r.dict()['return_result']
            e = r.dict()['extras']['qcvars']['CURRENT ENERGY']
            print(f"Obtaining ZPVE correction for {mol} in {i}")
            vib,therm = _vibanal_wfn(hess=h, molecule=r.get_molecule(), name=i, energy=e, lt=lot_opt)
            zpve_list.append(therm['ZPE_vib'].data)
        
        print(len(zpve_list), zpve_list)
    
        if len(zpve_list) != 3:
            todelete.append(i)
            continue

        d, m_1, m_2 = zpve_list
        zpve_corr = (d  - m_1 - m_2) * qcel.constants.hartree2kcalmol
        zpve_corr_dict[i]= zpve_corr
    
    df_zpve = pd.DataFrame.from_dict(zpve_corr_dict, orient='index', columns=["Delta_ZPVE"])
    df_be = ds_be.get_values()
    
    for i in todelete:
        df_be.drop(i,inplace=True)
    
    df_all = pd.concat([df_be, df_zpve], axis=1)
    
    # The ZPVE correction values obtained using HF-3c/minix geometries need to be scaled by 0.86:
    if lot_opt.split("_")[0] == 'hf3c':
        scale_factor = 0.86
    
    df_all['Delta_ZPVE'] = scale_factor * df_all['Delta_ZPVE']


    # Calculate ZPVE corrected binding energies
    df_all['Eb_ZPVE'] = df_all.loc[:,[be_method+'/def2-tzvp','Delta_ZPVE']].sum(axis=1)

    
    x = df_all[be_method+'/def2-tzvp'].astype(str).astype(float)
    y = df_all['Eb_ZPVE'].astype(str).astype(float)

    mol_name = "_".join(name_be.split("_")[1:4])
    
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)
    m,b = np.polyfit(x, y, 1)
    
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0,1]
    r_sq = correlation_xy**2
    
    n_rows = 1
    n_col = 1
    fig = plt.figure(figsize=(12,10))
    
    plt.plot(x,y, 's', markersize = 13)
    plt.plot(x, poly1d_fn(x), '--k', label= '''y = {0:.3g}x +{1:.3g}
    $R^2$ = {2:.2g}'''.format(m,b,r_sq))
    
    plt.xlabel('$E_b$ / $kcal$ $mol^{-1}$ ', size = 22)
    plt.ylabel('$E_b$ + $\Delta$ $ZPVE$ / $kcal$ $mol^{-1}$', size = 22)
    plt.title("Linear Fit: {} , {}/def2-tzvp $E_b$ values".format(mol_name, be_method), size = 16)
    
    plt.tick_params(direction='in', which = 'both', labelsize = 20, pad = 15, length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.2)
    
    plt.legend(prop={'size': 20}, loc = 2, edgecolor = 'inherit')
    
    return df_all, [m,b,r_sq], fig

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2./(2.*sigma**2.))


def gauss_fitting(nbins, data, p0, nboot = 10000):

    # do histogram of experimental data
    ydata, bin_edges = np.histogram(data, bins=nbins)
    
    # error is Poisson
    err = np.sqrt(ydata)
    
    # bins center
    xbin = (bin_edges[:-1] + bin_edges[1:]) / 2e0
    xbin = xbin.astype("float64")
    
    # init arrays where to store fit variables at each bootstrap iteration
    
    sigma_boot = np.zeros(nboot)
    mu_boot = np.zeros(nboot)
    a_boot = np.zeros(nboot)
    
    print("Startig the curve fit within the Poisson error")

    
    for i in range(nboot):

        # randomize values with Poisson errror
        rdata = np.random.normal(loc=ydata, scale=err, size=len(ydata))
        # fit randomized values
        coeff, var_matrix = curve_fit(gauss, xbin, rdata, p0=p0)
        # store coefficients
        a_boot[i], mu_boot[i], sigma_boot[i] = coeff
    
    # fit stored coefficient assuming their distribution is Gaussian
    print("Fitting the Gaussian parameters A, mu and sigma")
    vbest = []
    labels = ["A", "mu", "sigma"]
    plt.figure(figsize=(16, 4)) 
    p0_param  = [[1500, p0[0], 9.], [1500, p0[1], 300], [1500., p0[2], 30]]
    param_list = [a_boot, mu_boot, sigma_boot]
    
    for i, vdata in enumerate(param_list):
        
        vydata, vedges = np.histogram(vdata, bins=30)
        vxbin = (vedges[1:] + vedges[:-1]) / 2e0
        vcoef, var_matrix = curve_fit(gauss, vxbin, vydata, p0=p0_param[i])
        vbest.append(vcoef[1])
    
        #print("%s, A:%.2f, mu:%.2f, sigma: %.2f" % (labels[i], vcoef[0], vcoef[1], vcoef[2]))
    
        # plot the distribiution of the coeficients
        xdata = np.linspace(min(vxbin), max(vxbin), 100)
        zdata = gauss(xdata, *vcoef)
        plt.subplot(1, 4, i+1)
        plt.hist(vdata, bins=30)
        plt.plot(xdata, zdata)
    
   
    # check if bootstrap fit original data properly
    xdata = np.linspace(min(xbin), max(xbin), 100)
    plt.subplot(1, 4, 4)
    plt.hist(data, bins=nbins, color='g', alpha = 0.6)
    plt.bar(xbin, ydata, width=0, yerr=err, color='b')
    plt.plot(xdata, gauss(xdata, *vbest), color='k')
    print("The best fit is: A: {} mu: {} sigma: {}".format(vbest[0], vbest[1], vbest[2]))
    return vbest

zpve_dictionary = {'nh3_W22' : [0.7621480061870387,0.14224862969428917], 
         'ch3oh_W22': [0.8185524741301762,0.17034213359299527], 
         'h2co_W22':[0.7584599636175994,-0.4461259341133827], 
         'hf_W22':[0.7982923884361879,0.], 
         'hcl_W22':[0.9517641119685634,0.], 
         'h2o_W22':[0.7807501395430856,0.465792612888217], 
         'nhch2_W22': [0.8435323965581132,0.27703448545624243],
         'hco_W22': [0.7232360891823393,0.29782533964515256],
         'c2h2_W22': [0.8034189241100004,0.],
        'ch3o_W22':[0.813865963674682,0.39370667096085993],
          'hcn_W22': [0.8256948682725285,0.],
         'hnc_W22': [0.9288452608256647, 0.0],
         'ch2o2_W22':[0.899177485314638, -0.5078050232849965],
         'h2s_W22':[0.8058127174485291 ,0.0]
    }

fit_data = {'h2co': {'fit_params': [97.89266175231982,
   2969.591062638354,
   302.26377506601636],
  'nboot': 10000,
  'nbins': 13},
'hf': {'fit_params': [51.68655051544094, 4794.12840869433, 672.4421903018834],
  'nboot': 10000,
  'nbins': 12,
  'n': 212,
      },
'c2h2': {'fit_params': [47.09410551393588,
   1590.253514273688,
   300.9581435173123],
  'nboot': 10000,
  'nbins': 12,
  'n': 209},
'co2_M': {'fit_params': [49.889003526122416,
   1408.1457580178949,
   286.219089699056],
  'nboot': 1000,
  'nbins': 11,
  'n': 188},
 'co2_m': {'fit_params': [4.385909065239708, 1819.3010971859412, 68.55973000688073],  # 4.48 changed to 14.48
  'nboot': 10000,
  'nbins': 10,
  'n': 8},
'h2s': {'fit_params': [66.06266407464847,
   1793.9064934966075,
   422.4854960729488],
  'nboot': 10000,
  'nbins': 13,
  'n': 308},
'ch2oh_a': {'fit_params': [18.777700852763598,
   2690.6769190414743,
   192.70363320713454],
  'nboot': 3000,
  'nbins': 8,
  'n': 64},
'ch2oh_m': {'fit_params': [21.11276579666503,
   2670.3477437038573,
   181.63765247487342],
  'nboot': 10000,
  'nbins': 7,
  'n': 56},
'ch2oh_M': {'fit_params': [29.4444457724875,
   4451.0050508311015,
   843.2458367261609],
  'nboot': 10000,
  'nbins': 10,
  'n': 133},
'ch2oh_b': {'fit_params': [21.18885331645014,
   4368.985895636072,
   864.7623945978844],
  'nboot': 10000,
  'nbins': 10,
  'n': 125},
'h2o_m': {'fit_params': [42.58225993083435,
   2725.224445008547,
   449.1433329440647],
 'nboot': 10000,
  'nbins': 11,
  'n': 174},
#'h2o_m': {'fit_params': [28.855068263222897,
#   2732.500057091265,
#   460.1189682026524],
#  'nboot': 10000,
#  'nbins': 12,
#  'n': 231},            
'h2o_M': {'fit_params': [21.5807032728416,
   4087.3309323842727,
   338.314423177633],
  'nboot': 10000,
  'nbins': 7,
  'n': 59},
'hnc_M': {'fit_params': [44.080173499160765,
   4627.834115717624,
   705.7338190027715],
  'nboot': 10000,
  'nbins': 11,
  'n': 191},
'hnc_m': {'fit_params': [18.58597372180027,
   2552.3606864428725,
   117.06001956892375],
  'nboot': 10000,
  'nbins': 7,
  'n': 46},
 'ch2o2_m': {'fit_params': [33.14473702439963,
   6026.696682532694,
   801.8633836984219],
  'nboot': 10000,
  'nbins': 11,
  'n': 156},
#'ch2o2_mM': {'fit_params': [87.86788794637292,
#   2207.910731411251,
#   4.677078888570552],
#  'nboot': 10000,
#  'nbins': 6,
#  'n': 31},
'ch2o2_mM': {'fit_params': [15.94089713864299,
   2207.910731411251,
   200.197813502992055],
  'nboot': 1,
  'nbins': 6,
  'n': 31},
#'ch2o2_M': {'fit_params': [18.809404036004317,
#   3081.677549059341,
#   287.31812260415614],
#  'nboot': 9000,
#  'nbins': 9,
#  'n': 108},
'ch2o2_M': {'fit_params': [33.24387089978787,
   3265.6582969941787,
   268.6992570063701],
  'nboot': 10000,
  'nbins': 8,
  'n': 77},
#'nhch2': {'fit_params': [52.12095651900029,
#   3526.899667681464,
#   442.50852066495514],
#  'nboot': 5000,
#  'nbins': 12,
#  'n': 218},
'nhch2_m': {'fit_params': [44.78300820611808,
   3536.456406224352,
   448.55439333691976],
  'nboot': 10000,
  'nbins': 11,
  'n': 194},
'nhch2_M': {'fit_params': [10.51651052615661,
   1516.1161033344672,
   123.00612370470586],
  'nboot': 10000,
  'nbins': 5,
  'n': 24},
#
#'nhch2': {'fit_params':
#           [47.248787128447105, 3692.788942508374, 212.51252826875577],
#           'nboot':10000,
#           'nbins': 9,
#          'n': 113,
#  'filter': ['nhch2_W22_01_0001', 'nhch2_W22_01_0002', 'nhch2_W22_01_0008', 'nhch2_W22_01_0012', 'nhch2_W22_01_0017', 'nhch2_W22_01_0024', 'nhch2_W22_01_0025', 'nhch2_W22_02_0001', 'nhch2_W22_02_0004', 'nhch2_W22_02_0006', 'nhch2_W22_02_0007', 'nhch2_W22_02_0010', 'nhch2_W22_02_0011', 'nhch2_W22_02_0017', 'nhch2_W22_02_0020', 'nhch2_W22_02_0024', 'nhch2_W22_06_0001', 'nhch2_W22_06_0002', 'nhch2_W22_06_0003', 'nhch2_W22_06_0004', 'nhch2_W22_06_0006', 'nhch2_W22_06_0007', 'nhch2_W22_06_0011', 'nhch2_W22_06_0012', 'nhch2_W22_06_0014', 'nhch2_W22_06_0015', 'nhch2_W22_06_0019', 'nhch2_W22_06_0022', 'nhch2_W22_06_0024', 'nhch2_W22_09_0001', 'nhch2_W22_09_0002', 'nhch2_W22_09_0004', 'nhch2_W22_09_0007', 'nhch2_W22_09_0009', 'nhch2_W22_09_0010', 'nhch2_W22_09_0011', 'nhch2_W22_09_0012', 'nhch2_W22_09_0015', 'nhch2_W22_09_0016', 'nhch2_W22_09_0019', 'nhch2_W22_09_0029', 'nhch2_W22_11_0007', 'nhch2_W22_12_0001', 'nhch2_W22_12_0008', 'nhch2_W22_12_0012', 'nhch2_W22_12_0013', 'nhch2_W22_12_0014', 'nhch2_W22_12_0015', 'nhch2_W22_12_0016', 'nhch2_W22_12_0017', 'nhch2_W22_12_0028', 'nhch2_W22_13_0001', 'nhch2_W22_13_0002', 'nhch2_W22_13_0003', 'nhch2_W22_13_0008', 'nhch2_W22_13_0010', 'nhch2_W22_13_0016', 'nhch2_W22_13_0021', 'nhch2_W22_13_0023', 'nhch2_W22_14_0002', 'nhch2_W22_14_0003', 'nhch2_W22_14_0007', 'nhch2_W22_14_0019', 'nhch2_W22_14_0020', 'nhch2_W22_14_0021', 'nhch2_W22_14_0022', 'nhch2_W22_14_0025', 'nhch2_W22_15_0002', 'nhch2_W22_15_0004', 'nhch2_W22_15_0005', 'nhch2_W22_15_0007', 'nhch2_W22_15_0008', 'nhch2_W22_15_0011', 'nhch2_W22_15_0018', 'nhch2_W22_15_0020', 'nhch2_W22_15_0021', 'nhch2_W22_15_0022', 'nhch2_W22_15_0027', 'nhch2_W22_16_0001', 'nhch2_W22_16_0002', 'nhch2_W22_16_0003', 'nhch2_W22_16_0004', 'nhch2_W22_16_0005', 'nhch2_W22_16_0007', 'nhch2_W22_16_0011', 'nhch2_W22_16_0014', 'nhch2_W22_16_0015', 'nhch2_W22_16_0016', 'nhch2_W22_16_0023', 'nhch2_W22_16_0024', 'nhch2_W22_16_0028', 'nhch2_W22_16_0029', 'nhch2_W22_17_0001', 'nhch2_W22_17_0006', 'nhch2_W22_17_0008', 'nhch2_W22_17_0009', 'nhch2_W22_17_0015', 'nhch2_W22_17_0016', 'nhch2_W22_17_0017', 'nhch2_W22_17_0018', 'nhch2_W22_17_0019', 'nhch2_W22_18_0004', 'nhch2_W22_18_0005', 'nhch2_W22_18_0007', 'nhch2_W22_18_0009', 'nhch2_W22_18_0011', 'nhch2_W22_18_0016', 'nhch2_W22_18_0018', 'nhch2_W22_18_0019', 'nhch2_W22_18_0021', 'nhch2_W22_18_0024', 'nhch2_W22_18_0025', 'nhch2_W22_18_0026', 'nhch2_W22_18_0030']
# },
'hco': {'fit_params': [68.56026373263063,
   1316.9206347977636,
   378.056584803791],
  'nboot': 10000,
  'nbins': 12,
        'n': 218
        },
'nh3': {'fit_params': [96.35325196671593,
   3387.5134948269238,
   242.4874502393706],
  'nboot': 10000,
  'nbins': 12,
  'n': 242,
  'filter':['nh3_W22_01_0001', 'nh3_W22_01_0002', 'nh3_W22_01_0003', 'nh3_W22_01_0005', 'nh3_W22_01_0006', 'nh3_W22_01_0007', 'nh3_W22_01_0010', 'nh3_W22_01_0015', 'nh3_W22_01_0016', 'nh3_W22_01_0030', 'nh3_W22_01_0031', 'nh3_W22_01_0037', 'nh3_W22_01_0038', 'nh3_W22_01_0050', 'nh3_W22_01_0056', 'nh3_W22_02_0001', 'nh3_W22_02_0002', 'nh3_W22_02_0003', 'nh3_W22_02_0004', 'nh3_W22_02_0005', 'nh3_W22_02_0006', 'nh3_W22_02_0008', 'nh3_W22_02_0010', 'nh3_W22_02_0011', 'nh3_W22_02_0014', 'nh3_W22_02_0018', 'nh3_W22_02_0020', 'nh3_W22_02_0023', 'nh3_W22_02_0026', 'nh3_W22_02_0029', 'nh3_W22_02_0040', 'nh3_W22_02_0041', 'nh3_W22_02_0049', 'nh3_W22_02_0063', 'nh3_W22_03_0001', 'nh3_W22_03_0002', 'nh3_W22_03_0003', 'nh3_W22_03_0006', 'nh3_W22_03_0008', 'nh3_W22_03_0010', 'nh3_W22_03_0011', 'nh3_W22_03_0014', 'nh3_W22_03_0016', 'nh3_W22_03_0022', 'nh3_W22_03_0025', 'nh3_W22_03_0028', 'nh3_W22_03_0039', 'nh3_W22_03_0040', 'nh3_W22_03_0041', 'nh3_W22_03_0042', 'nh3_W22_03_0043', 'nh3_W22_03_0047', 'nh3_W22_03_0051', 'nh3_W22_04_0001', 'nh3_W22_04_0002', 'nh3_W22_04_0003', 'nh3_W22_04_0006', 'nh3_W22_04_0007', 'nh3_W22_04_0010', 'nh3_W22_04_0011', 'nh3_W22_04_0012', 'nh3_W22_04_0013', 'nh3_W22_04_0014', 'nh3_W22_04_0018', 'nh3_W22_04_0019', 'nh3_W22_04_0021', 'nh3_W22_04_0022', 'nh3_W22_04_0027', 'nh3_W22_04_0031', 'nh3_W22_04_0033', 'nh3_W22_04_0034', 'nh3_W22_04_0037', 'nh3_W22_04_0039', 'nh3_W22_05_0001', 'nh3_W22_05_0004', 'nh3_W22_05_0005', 'nh3_W22_05_0006', 'nh3_W22_05_0008', 'nh3_W22_05_0009', 'nh3_W22_05_0010', 'nh3_W22_05_0014', 'nh3_W22_05_0015', 'nh3_W22_05_0021', 'nh3_W22_05_0029', 'nh3_W22_05_0037', 'nh3_W22_06_0001', 'nh3_W22_06_0002', 'nh3_W22_06_0003', 'nh3_W22_06_0004', 'nh3_W22_06_0006', 'nh3_W22_06_0010', 'nh3_W22_06_0011', 'nh3_W22_06_0013', 'nh3_W22_06_0014', 'nh3_W22_06_0017', 'nh3_W22_06_0027', 'nh3_W22_06_0030', 'nh3_W22_06_0031', 'nh3_W22_06_0032', 'nh3_W22_06_0033', 'nh3_W22_06_0049', 'nh3_W22_09_0001', 'nh3_W22_09_0002', 'nh3_W22_09_0004', 'nh3_W22_09_0005', 'nh3_W22_09_0006', 'nh3_W22_09_0007', 'nh3_W22_09_0009', 'nh3_W22_09_0012', 'nh3_W22_09_0017', 'nh3_W22_09_0018', 'nh3_W22_09_0022', 'nh3_W22_09_0023', 'nh3_W22_09_0024', 'nh3_W22_09_0026', 'nh3_W22_09_0034', 'nh3_W22_09_0036', 'nh3_W22_09_0041', 'nh3_W22_09_0048', 'nh3_W22_09_0051', 'nh3_W22_09_0059', 'nh3_W22_09_0061', 'nh3_W22_10_0001', 'nh3_W22_10_0002', 'nh3_W22_10_0003', 'nh3_W22_10_0004', 'nh3_W22_10_0006', 'nh3_W22_10_0007', 'nh3_W22_10_0008', 'nh3_W22_10_0009', 'nh3_W22_10_0010', 'nh3_W22_10_0012', 'nh3_W22_10_0014', 'nh3_W22_10_0025', 'nh3_W22_10_0027', 'nh3_W22_10_0028', 'nh3_W22_10_0031', 'nh3_W22_10_0032', 'nh3_W22_10_0034', 'nh3_W22_10_0035', 'nh3_W22_10_0036', 'nh3_W22_10_0040', 'nh3_W22_10_0041', 'nh3_W22_10_0042', 'nh3_W22_14_0001', 'nh3_W22_14_0002', 'nh3_W22_14_0004', 'nh3_W22_14_0007', 'nh3_W22_14_0008', 'nh3_W22_14_0013', 'nh3_W22_14_0015', 'nh3_W22_14_0020', 'nh3_W22_14_0022', 'nh3_W22_14_0024', 'nh3_W22_14_0025', 'nh3_W22_14_0034', 'nh3_W22_14_0036', 'nh3_W22_14_0038', 'nh3_W22_14_0050', 'nh3_W22_14_0052', 'nh3_W22_14_0053', 'nh3_W22_15_0001', 'nh3_W22_15_0003', 'nh3_W22_15_0005', 'nh3_W22_15_0007', 'nh3_W22_15_0009', 'nh3_W22_15_0011', 'nh3_W22_15_0013', 'nh3_W22_15_0014', 'nh3_W22_15_0015', 'nh3_W22_15_0017', 'nh3_W22_15_0019', 'nh3_W22_15_0020', 'nh3_W22_15_0022', 'nh3_W22_15_0023', 'nh3_W22_15_0024', 'nh3_W22_15_0033', 'nh3_W22_15_0037', 'nh3_W22_15_0038', 'nh3_W22_16_0001', 'nh3_W22_16_0002', 'nh3_W22_16_0003', 'nh3_W22_16_0004', 'nh3_W22_16_0006', 'nh3_W22_16_0007', 'nh3_W22_16_0008', 'nh3_W22_16_0012', 'nh3_W22_16_0014', 'nh3_W22_16_0018', 'nh3_W22_16_0036', 'nh3_W22_16_0046', 'nh3_W22_16_0049', 'nh3_W22_16_0052', 'nh3_W22_16_0058']
      },
'nh3_m': {'fit_params': [96.93074566873253,
   3346.564406284833,
   219.05091027332963],
  'nboot': 10000,
  'nbins': 12,
  'n': 245},
'nh3_M': {'fit_params': [6.16788794637292, 1103.910731411251, 197],
  'nboot': 100,
  'nbins': 5,
  'n': 16},
'ch4': {'fit_params': [64.06397627910968,
   773.3004121814498,
   125.55212386345937],
  'nboot': 10000,
  'nbins': 13,
  'n': 257},
'ch3oh_m': {'fit_params': [16.732599478511325,
   2344.1008219699393,
   200.9823895635272],
  'nboot': 10000,
  'nbins': 8,
  'n': 67},
'ch3oh_M': {'fit_params': [41.20094477305211,
   3235.3193800712456,
   701.8636963685753],
  'nboot': 10000,
  'nbins': 11,
  'n': 172},
#'ch3oh': {'fit_params': [79.89439252968273,
#   3077.620980784922,
#   405.69671040831986],
#  'nboot': 10000,
#  'nbins': 12,
#  'n': 241},
#'ch3oh_m': {'fit_params': [33.19664544769346,
#   3008.3345499552565,
#   325.9875439212056],
#  'nboot': 10000,
#  'nbins': 8,
#  'n': 69},
#'ch3oh_M': {'fit_params': [41.208133585718485,
#   3236.576881295405,
#   700.027102382577],
#  'nboot': 10000,
#  'nbins': 11,
#  'n': 172},
'n2': {'fit_params': [67.07506553883705,
   637.3656640577545,
   113.88159727864256],
  'nboot': 10000,
  'nbins': 11,
  'n': 188},
'h2': {'fit_params': [54.03386809190605,
   301.04073383074353,
   81.20972504067258],
  'nboot': 5000,
  'nbins': 9,
  'n': 162},
'hcn': {'fit_params': [90.07752228655136,
   2341.5194243685846,
   589.2538592386865],
  'nboot': 10000,
  'nbins': 12,
  'n': 218},
'ch3': {'fit_params': [29.830419983398595,
   1062.165274555384,
   193.62002313271296],
  'nboot': 10000,
  'nbins': 10,
  'n': 131},

'co': {'fit_params': [22.398564391391822,
   415.02529356231946,
   366.7536395871657],
  'nboot': 1000,
  'nbins': 10,
  'n': 135,
  'filter': ['co_W22_01_0001', 'co_W22_01_0002', 'co_W22_01_0003', 'co_W22_01_0004', 'co_W22_01_0005', 'co_W22_01_0009', 'co_W22_01_0012', 'co_W22_01_0014', 'co_W22_01_0023', 'co_W22_01_0025', 'co_W22_01_0031', 'co_W22_01_0045', 'co_W22_02_0001', 'co_W22_02_0002', 'co_W22_02_0003', 'co_W22_02_0004', 'co_W22_02_0008', 'co_W22_02_0009', 'co_W22_02_0010', 'co_W22_02_0012', 'co_W22_02_0017', 'co_W22_02_0023', 'co_W22_02_0027', 'co_W22_02_0030', 'co_W22_03_0001', 'co_W22_03_0003', 'co_W22_03_0004', 'co_W22_03_0005', 'co_W22_03_0006', 'co_W22_03_0007', 'co_W22_03_0009', 'co_W22_03_0011', 'co_W22_03_0015', 'co_W22_03_0021', 'co_W22_03_0025', 'co_W22_03_0030', 'co_W22_03_0038', 'co_W22_03_0045', 'co_W22_03_0046', 'co_W22_04_0001', 'co_W22_04_0002', 'co_W22_04_0003', 'co_W22_04_0004', 'co_W22_04_0005', 'co_W22_04_0007', 'co_W22_04_0009', 'co_W22_04_0010', 'co_W22_04_0012', 'co_W22_04_0013', 'co_W22_04_0014', 'co_W22_04_0016', 'co_W22_04_0018', 'co_W22_04_0021', 'co_W22_04_0030', 'co_W22_04_0034', 'co_W22_04_0045', 'co_W22_05_0001', 'co_W22_05_0002', 'co_W22_05_0004', 'co_W22_05_0006', 'co_W22_05_0007', 'co_W22_05_0010', 'co_W22_05_0012', 'co_W22_05_0013', 'co_W22_05_0017', 'co_W22_05_0021', 'co_W22_05_0029', 'co_W22_06_0001', 'co_W22_06_0003', 'co_W22_06_0004', 'co_W22_06_0006', 'co_W22_06_0007', 'co_W22_06_0008', 'co_W22_06_0011', 'co_W22_06_0018', 'co_W22_06_0027', 'co_W22_06_0030', 'co_W22_06_0031', 'co_W22_06_0037', 'co_W22_06_0042', 'co_W22_06_0043', 'co_W22_06_0062', 'co_W22_06_0067', 'co_W22_09_0001', 'co_W22_09_0002', 'co_W22_09_0003', 'co_W22_09_0004', 'co_W22_09_0007', 'co_W22_09_0008', 'co_W22_09_0012', 'co_W22_09_0015', 'co_W22_09_0018', 'co_W22_09_0019', 'co_W22_09_0021', 'co_W22_09_0022', 'co_W22_09_0035', 'co_W22_09_0046', 'co_W22_10_0001', 'co_W22_10_0002', 'co_W22_10_0003', 'co_W22_10_0006', 'co_W22_10_0007', 'co_W22_10_0008', 'co_W22_10_0009', 'co_W22_10_0011', 'co_W22_10_0012', 'co_W22_10_0017', 'co_W22_10_0019', 'co_W22_10_0025', 'co_W22_10_0028', 'co_W22_10_0029', 'co_W22_10_0031', 'co_W22_10_0036', 'co_W22_10_0038', 'co_W22_10_0041', 'co_W22_10_0045', 'co_W22_10_0046', 'co_W22_10_0047', 'co_W22_11_0001', 'co_W22_11_0003', 'co_W22_11_0004', 'co_W22_11_0006', 'co_W22_11_0007', 'co_W22_11_0009', 'co_W22_11_0012', 'co_W22_11_0016', 'co_W22_11_0035', 'co_W22_11_0040', 'co_W22_11_0043', 'co_W22_11_0048', 'co_W22_11_0052', 'co_W22_11_0063', 'co_W22_11_0090', 'co_W22_12_0001', 'co_W22_12_0002', 'co_W22_12_0004', 'co_W22_12_0006', 'co_W22_12_0007', 'co_W22_12_0009', 'co_W22_12_0010', 'co_W22_12_0011', 'co_W22_12_0013', 'co_W22_12_0014', 'co_W22_12_0015', 'co_W22_12_0016', 'co_W22_12_0032', 'co_W22_12_0037', 'co_W22_12_0039', 'co_W22_12_0046']

  },
#'co': {'fit_params': [35.62789119143636,
#   1019.3612479627807,
#   187.866163186845],
#  'nboot': 10000,
#  'nbins': 10,
#  'n': 150,
#       'filter': ['co_W22_01_0001', 'co_W22_01_0002', 'co_W22_01_0003', 'co_W22_01_0004', 'co_W22_01_0005', 'co_W22_01_0009', 'co_W22_01_0012', 'co_W22_01_0014', 'co_W22_01_0023', 'co_W22_01_0025', 'co_W22_01_0031', 'co_W22_01_0045', 'co_W22_02_0001', 'co_W22_02_0002', 'co_W22_02_0003', 'co_W22_02_0004', 'co_W22_02_0008', 'co_W22_02_0009', 'co_W22_02_0010', 'co_W22_02_0012', 'co_W22_02_0017', 'co_W22_02_0023', 'co_W22_02_0027', 'co_W22_02_0030', 'co_W22_03_0001', 'co_W22_03_0003', 'co_W22_03_0004', 'co_W22_03_0005', 'co_W22_03_0006', 'co_W22_03_0007', 'co_W22_03_0009', 'co_W22_03_0011', 'co_W22_03_0015', 'co_W22_03_0021', 'co_W22_03_0025', 'co_W22_03_0030', 'co_W22_03_0038', 'co_W22_03_0045', 'co_W22_03_0046', 'co_W22_04_0001', 'co_W22_04_0002', 'co_W22_04_0003', 'co_W22_04_0004', 'co_W22_04_0005', 'co_W22_04_0007', 'co_W22_04_0009', 'co_W22_04_0010', 'co_W22_04_0012', 'co_W22_04_0013', 'co_W22_04_0014', 'co_W22_04_0016', 'co_W22_04_0018', 'co_W22_04_0021', 'co_W22_04_0030', 'co_W22_04_0034', 'co_W22_04_0045', 'co_W22_05_0001', 'co_W22_05_0002', 'co_W22_05_0004', 'co_W22_05_0006', 'co_W22_05_0007', 'co_W22_05_0010', 'co_W22_05_0012', 'co_W22_05_0013', 'co_W22_05_0017', 'co_W22_05_0021', 'co_W22_05_0029', 'co_W22_06_0001', 'co_W22_06_0003', 'co_W22_06_0004', 'co_W22_06_0006', 'co_W22_06_0007', 'co_W22_06_0008', 'co_W22_06_0011', 'co_W22_06_0018', 'co_W22_06_0027', 'co_W22_06_0030', 'co_W22_06_0031', 'co_W22_06_0037', 'co_W22_06_0042', 'co_W22_06_0043', 'co_W22_06_0062', 'co_W22_06_0067', 'co_W22_09_0001', 'co_W22_09_0002', 'co_W22_09_0003', 'co_W22_09_0004', 'co_W22_09_0007', 'co_W22_09_0008', 'co_W22_09_0012', 'co_W22_09_0015', 'co_W22_09_0018', 'co_W22_09_0019', 'co_W22_09_0021', 'co_W22_09_0022', 'co_W22_09_0035', 'co_W22_09_0046', 'co_W22_10_0001', 'co_W22_10_0002', 'co_W22_10_0003', 'co_W22_10_0006', 'co_W22_10_0007', 'co_W22_10_0008', 'co_W22_10_0009', 'co_W22_10_0011', 'co_W22_10_0012', 'co_W22_10_0017', 'co_W22_10_0019', 'co_W22_10_0025', 'co_W22_10_0028', 'co_W22_10_0029', 'co_W22_10_0031', 'co_W22_10_0036', 'co_W22_10_0038', 'co_W22_10_0041', 'co_W22_10_0045', 'co_W22_10_0046', 'co_W22_10_0047', 'co_W22_11_0001', 'co_W22_11_0003', 'co_W22_11_0004', 'co_W22_11_0006', 'co_W22_11_0007', 'co_W22_11_0009', 'co_W22_11_0012', 'co_W22_11_0016', 'co_W22_11_0035', 'co_W22_11_0040', 'co_W22_11_0043', 'co_W22_11_0048', 'co_W22_11_0052', 'co_W22_11_0063', 'co_W22_11_0090', 'co_W22_12_0001', 'co_W22_12_0002', 'co_W22_12_0004', 'co_W22_12_0006', 'co_W22_12_0007', 'co_W22_12_0009', 'co_W22_12_0010', 'co_W22_12_0011', 'co_W22_12_0013', 'co_W22_12_0014', 'co_W22_12_0015', 'co_W22_12_0016', 'co_W22_12_0032', 'co_W22_12_0037', 'co_W22_12_0039', 'co_W22_12_0046']
#      },
'nh2': {'fit_params': [56.94485651275069,
   3488.1560257779433,
   265.21691303573715],
  'nboot': 1000,
  'nbins': 10,
  'n': 146},
'ch3o': {'fit_params': [58.050203358036164,
   2274.011035003585,
   258.88746663540763],
  'nboot': 10000,
  'nbins': 11,
  'n': 184}
           }


fit_data_hyb = {
'co_m': {'fit_params': [6.18885331645014, 520.985895636072, 45.7623945978844],  #6.18 changed to 16.18
  'nboot': 10000,
  'nbins': 10,
  'n': 16},
'co_M': {'fit_params': [35.193212749369266,
   1034.6818436559981,
   175.7400277776676],
  'nboot': 10000,
  'nbins': 10,
  'n': 143},
'co': {'fit_params': [35.62789119143636,
   1019.3612479627807,
   187.866163186845],
  'nboot': 10000,
  'nbins': 10,
  'n': 150,
  'filter': ['co_W22_01_0001', 'co_W22_01_0002', 'co_W22_01_0003', 'co_W22_01_0004', 'co_W22_01_0005', 'co_W22_01_0008', 'co_W22_01_0011', 'co_W22_01_0012', 'co_W22_01_0014', 'co_W22_01_0015', 'co_W22_01_0018', 'co_W22_01_0023', 'co_W22_01_0025', 'co_W22_01_0031', 'co_W22_01_0033', 'co_W22_01_0038', 'co_W22_01_0045', 'co_W22_02_0001', 'co_W22_02_0002', 'co_W22_02_0003', 'co_W22_02_0004', 'co_W22_02_0005', 'co_W22_02_0007', 'co_W22_02_0008', 'co_W22_02_0009', 'co_W22_02_0010', 'co_W22_02_0012', 'co_W22_02_0017', 'co_W22_02_0020', 'co_W22_02_0023', 'co_W22_02_0024', 'co_W22_02_0027', 'co_W22_02_0030', 'co_W22_02_0033', 'co_W22_03_0001', 'co_W22_03_0002', 'co_W22_03_0003', 'co_W22_03_0004', 'co_W22_03_0005', 'co_W22_03_0006', 'co_W22_03_0007', 'co_W22_03_0009', 'co_W22_03_0011', 'co_W22_03_0012', 'co_W22_03_0015', 'co_W22_03_0021', 'co_W22_03_0025', 'co_W22_03_0027', 'co_W22_03_0034', 'co_W22_03_0046', 'co_W22_04_0001', 'co_W22_04_0002', 'co_W22_04_0003', 'co_W22_04_0004', 'co_W22_04_0005', 'co_W22_04_0007', 'co_W22_04_0009', 'co_W22_04_0010', 'co_W22_04_0012', 'co_W22_04_0013', 'co_W22_04_0014', 'co_W22_04_0016', 'co_W22_04_0018', 'co_W22_04_0020', 'co_W22_04_0021', 'co_W22_04_0030', 'co_W22_04_0033', 'co_W22_04_0034', 'co_W22_04_0045', 'co_W22_05_0001', 'co_W22_05_0002', 'co_W22_05_0004', 'co_W22_05_0005', 'co_W22_05_0006', 'co_W22_05_0007', 'co_W22_05_0010', 'co_W22_05_0012', 'co_W22_05_0013', 'co_W22_05_0015', 'co_W22_05_0017', 'co_W22_05_0021', 'co_W22_05_0023', 'co_W22_05_0027', 'co_W22_05_0028', 'co_W22_05_0029', 'co_W22_05_0031', 'co_W22_06_0001', 'co_W22_06_0002', 'co_W22_06_0003', 'co_W22_06_0004', 'co_W22_06_0005', 'co_W22_06_0006', 'co_W22_06_0007', 'co_W22_06_0008', 'co_W22_06_0011', 'co_W22_06_0012', 'co_W22_06_0015', 'co_W22_06_0018', 'co_W22_06_0026', 'co_W22_06_0027', 'co_W22_06_0030', 'co_W22_06_0031', 'co_W22_06_0037', 'co_W22_06_0039', 'co_W22_06_0042', 'co_W22_06_0052', 'co_W22_06_0062', 'co_W22_06_0067', 'co_W22_06_0069', 'co_W22_09_0001', 'co_W22_09_0002', 'co_W22_09_0003', 'co_W22_09_0004', 'co_W22_09_0005', 'co_W22_09_0007', 'co_W22_09_0008', 'co_W22_09_0010', 'co_W22_09_0012', 'co_W22_09_0015', 'co_W22_09_0018', 'co_W22_09_0021', 'co_W22_09_0022', 'co_W22_09_0023', 'co_W22_09_0035', 'co_W22_10_0001', 'co_W22_10_0002', 'co_W22_10_0003', 'co_W22_10_0004', 'co_W22_10_0006', 'co_W22_10_0007', 'co_W22_10_0008', 'co_W22_10_0009', 'co_W22_10_0010', 'co_W22_10_0011', 'co_W22_10_0012', 'co_W22_10_0017', 'co_W22_10_0019', 'co_W22_10_0020', 'co_W22_10_0022', 'co_W22_10_0025', 'co_W22_10_0028', 'co_W22_10_0029', 'co_W22_10_0031', 'co_W22_10_0034', 'co_W22_10_0036', 'co_W22_10_0038', 'co_W22_10_0042', 'co_W22_10_0046', 'co_W22_10_0047', 'co_W22_10_0058']
      
  },
    'nh3': {'fit_params': [61.257110357287864,
   3196.679067696749,
   240.4621793734706],
  'nboot': 10000,
  'nbins': 11,
  'n': 177,
'filter': ['nh3_W22_01_0001', 'nh3_W22_01_0003', 'nh3_W22_01_0005', 'nh3_W22_01_0006', 'nh3_W22_01_0007', 'nh3_W22_01_0010', 'nh3_W22_01_0015', 'nh3_W22_01_0017', 'nh3_W22_01_0026', 'nh3_W22_01_0028', 'nh3_W22_01_0030', 'nh3_W22_01_0037', 'nh3_W22_01_0038', 'nh3_W22_01_0039', 'nh3_W22_01_0041', 'nh3_W22_01_0043', 'nh3_W22_01_0050', 'nh3_W22_01_0056', 'nh3_W22_01_0058', 'nh3_W22_02_0001', 'nh3_W22_02_0003', 'nh3_W22_02_0004', 'nh3_W22_02_0005', 'nh3_W22_02_0006', 'nh3_W22_02_0008', 'nh3_W22_02_0009', 'nh3_W22_02_0010', 'nh3_W22_02_0011', 'nh3_W22_02_0014', 'nh3_W22_02_0015', 'nh3_W22_02_0018', 'nh3_W22_02_0020', 'nh3_W22_02_0026', 'nh3_W22_02_0029', 'nh3_W22_02_0033', 'nh3_W22_02_0036', 'nh3_W22_02_0040', 'nh3_W22_02_0041', 'nh3_W22_02_0049', 'nh3_W22_02_0062', 'nh3_W22_02_0071', 'nh3_W22_02_0077', 'nh3_W22_02_0082', 'nh3_W22_03_0001', 'nh3_W22_03_0002', 'nh3_W22_03_0003', 'nh3_W22_03_0006', 'nh3_W22_03_0008', 'nh3_W22_03_0011', 'nh3_W22_03_0012', 'nh3_W22_03_0014', 'nh3_W22_03_0016', 'nh3_W22_03_0022', 'nh3_W22_03_0025', 'nh3_W22_03_0043', 'nh3_W22_03_0047', 'nh3_W22_03_0051', 'nh3_W22_04_0001', 'nh3_W22_04_0002', 'nh3_W22_04_0003', 'nh3_W22_04_0006', 'nh3_W22_04_0007', 'nh3_W22_04_0010', 'nh3_W22_04_0011', 'nh3_W22_04_0012', 'nh3_W22_04_0013', 'nh3_W22_04_0014', 'nh3_W22_04_0019', 'nh3_W22_04_0021', 'nh3_W22_04_0025', 'nh3_W22_04_0027', 'nh3_W22_04_0031', 'nh3_W22_04_0033', 'nh3_W22_04_0034', 'nh3_W22_04_0037', 'nh3_W22_04_0039', 'nh3_W22_04_0041', 'nh3_W22_05_0001', 'nh3_W22_05_0003', 'nh3_W22_05_0004', 'nh3_W22_05_0005', 'nh3_W22_05_0006', 'nh3_W22_05_0007', 'nh3_W22_05_0008', 'nh3_W22_05_0009', 'nh3_W22_05_0010', 'nh3_W22_05_0014', 'nh3_W22_05_0015', 'nh3_W22_05_0019', 'nh3_W22_05_0023', 'nh3_W22_05_0029', 'nh3_W22_05_0037', 'nh3_W22_06_0002', 'nh3_W22_06_0003', 'nh3_W22_06_0004', 'nh3_W22_06_0006', 'nh3_W22_06_0010', 'nh3_W22_06_0013', 'nh3_W22_06_0017', 'nh3_W22_06_0018', 'nh3_W22_06_0019', 'nh3_W22_06_0027', 'nh3_W22_06_0030', 'nh3_W22_06_0031', 'nh3_W22_06_0032', 'nh3_W22_06_0049', 'nh3_W22_09_0001', 'nh3_W22_09_0002', 'nh3_W22_09_0004', 'nh3_W22_09_0005', 'nh3_W22_09_0006', 'nh3_W22_09_0007', 'nh3_W22_09_0010', 'nh3_W22_09_0012', 'nh3_W22_09_0013', 'nh3_W22_09_0017', 'nh3_W22_09_0018', 'nh3_W22_09_0022', 'nh3_W22_09_0024', 'nh3_W22_09_0026', 'nh3_W22_09_0033', 'nh3_W22_09_0034', 'nh3_W22_09_0036', 'nh3_W22_09_0044', 'nh3_W22_09_0048', 'nh3_W22_09_0051', 'nh3_W22_09_0059', 'nh3_W22_10_0001', 'nh3_W22_10_0002', 'nh3_W22_10_0003', 'nh3_W22_10_0004', 'nh3_W22_10_0006', 'nh3_W22_10_0007', 'nh3_W22_10_0008', 'nh3_W22_10_0009', 'nh3_W22_10_0010', 'nh3_W22_10_0012', 'nh3_W22_10_0013', 'nh3_W22_10_0025', 'nh3_W22_10_0027', 'nh3_W22_10_0028', 'nh3_W22_10_0031', 'nh3_W22_10_0032', 'nh3_W22_10_0033', 'nh3_W22_10_0034', 'nh3_W22_10_0035', 'nh3_W22_10_0040', 'nh3_W22_10_0042', 'nh3_W22_11_0001', 'nh3_W22_11_0002', 'nh3_W22_11_0004', 'nh3_W22_11_0005', 'nh3_W22_11_0007', 'nh3_W22_11_0008', 'nh3_W22_11_0009', 'nh3_W22_11_0010', 'nh3_W22_11_0012', 'nh3_W22_11_0018', 'nh3_W22_11_0023', 'nh3_W22_11_0026', 'nh3_W22_11_0034', 'nh3_W22_11_0037', 'nh3_W22_11_0041', 'nh3_W22_11_0046', 'nh3_W22_11_0051', 'nh3_W22_12_0001', 'nh3_W22_12_0002', 'nh3_W22_12_0003', 'nh3_W22_12_0004', 'nh3_W22_12_0007', 'nh3_W22_12_0009', 'nh3_W22_12_0010', 'nh3_W22_12_0012', 'nh3_W22_12_0015', 'nh3_W22_12_0017', 'nh3_W22_12_0019', 'nh3_W22_12_0020']
           },

   'hf': {'fit_params': [46.57080082678657,
   4476.769787511505,
   738.7749961142687],
  'nboot': 10000,
  'nbins': 11,
  'n': 196},

    'ch4': {'fit_params': [39.46693827199936,
   845.6475930480823,
   148.96080579236929],
  'nboot': 10000,
  'nbins': 11,
  'n': 197},
'ch3oh': {'fit_params': [89.80969465082849,
   3830.4501470406117,
   391.25356952807306],
  'nboot': 10000,
  'nbins': 12,
  'n': 232},
'h2co': {'fit_params': [35.01637754692832,
   3662.8331546870045,
   550.834745772044],
  'nboot': 10000,
  'nbins': 11,
  'n': 157},
'n2': {'fit_params': [64.87330000824913,
   681.1875987558071,
   100.60727148066195],
  'nboot': 10000,
  'nbins': 12,
  'n': 213}
}



fit_dict_h_geom = {'h2co': [32.87405352200029, 3640.237828823547, 533.636943718558],
 'nh3': [59.34970269062614, 4331.6058907829, 291.90969432383844],
 'ch3oh': [89.71838750436365, 3831.040731201496, 392.1546706593619],
 'hf': [38.928977978134384, 5578.9910037283125, 856.4912600056758],
 'ch4': [36.67448406984524, 842.8170509748614, 145.73635420617856],
 'co' : [31.29147685835279, 1022.4361062197701, 169.30772427491596],
  'n2': [64.87330000824913,
   681.1875987558071,
   100.60727148066195],
'c2h2':[41.16453092310805, 1646.5358837674871, 323.7230188442273]
}

fit_dict_geom = {'h2co': [40.935492603610655, 3653.7123924140024, 352.92290567911215],
 'nh3': [72.51189818918093, 4579.2568490057265, 296.84314353171357],
 'ch3oh': [73.60664311397525, 3880.5752206580405, 522.9381714488848],
 'hf': [41.382973448336, 5971.305314759656, 843.33683925724],
 'ch4': [89.99038786293167, 760.3777815203639, 112.0348779093874],
'co': [27.240817077523076, 418.08234991337025, 351.5538423801491],
 'n2': [67.07506553883705,
   637.3656640577545,
   113.88159727864256],
'c2h2':  [47.09410551393588,
   1590.253514273688,
   300.9581435173123]
}

fit_dict_w = {'W12': [51.30740063728372, 4666.187978722857, 253.53215429877974], 'W60': [34.596243542419344, 5415.724988658218, 949.9919348261457], 'W22': [111.68562016355713, 4684.906435900515, 338.25935569702665], 'W37': [61.7599683285835, 5051.408460999617, 864.3155610952904]}

nh3_filter_hf3c = {'W12': ['nh3_W12_01_0004', 'nh3_W12_01_0006', 'nh3_W12_01_0009', 'nh3_W12_01_0015', 'nh3_W12_02_0002', 'nh3_W12_02_0006', 'nh3_W12_02_0020', 'nh3_W12_02_0029', 'nh3_W12_03_0001', 'nh3_W12_03_0003', 'nh3_W12_03_0008', 'nh3_W12_03_0009', 'nh3_W12_03_0010', 'nh3_W12_03_0012', 'nh3_W12_03_0015', 'nh3_W12_03_0019', 'nh3_W12_04_0005', 'nh3_W12_04_0009', 'nh3_W12_04_0011', 'nh3_W12_04_0013', 'nh3_W12_04_0019', 'nh3_W12_05_0003', 'nh3_W12_05_0016', 'nh3_W12_05_0021', 'nh3_W12_05_0022', 'nh3_W12_06_0001', 'nh3_W12_06_0002', 'nh3_W12_06_0018', 'nh3_W12_06_0019', 'nh3_W12_06_0022', 'nh3_W12_06_0023', 'nh3_W12_06_0024', 'nh3_W12_07_0001', 'nh3_W12_08_0004', 'nh3_W12_08_0009', 'nh3_W12_08_0020', 'nh3_W12_08_0026', 'nh3_W12_08_0029', 'nh3_W12_08_0036', 'nh3_W12_08_0041', 'nh3_W12_09_0004', 'nh3_W12_09_0006', 'nh3_W12_09_0008', 'nh3_W12_10_0003', 'nh3_W12_10_0005', 'nh3_W12_10_0006', 'nh3_W12_10_0010', 'nh3_W12_11_0001', 'nh3_W12_11_0002', 'nh3_W12_11_0004', 'nh3_W12_11_0008', 'nh3_W12_11_0012', 'nh3_W12_11_0014', 'nh3_W12_11_0023', 'nh3_W12_12_0003', 'nh3_W12_12_0004', 'nh3_W12_12_0006', 'nh3_W12_12_0011']
,
        'W22': ['nh3_W22_01_0003', 'nh3_W22_01_0005', 'nh3_W22_01_0007', 'nh3_W22_01_0017', 'nh3_W22_01_0031', 'nh3_W22_01_0050', 'nh3_W22_01_0056', 'nh3_W22_02_0002', 'nh3_W22_02_0004', 'nh3_W22_02_0010', 'nh3_W22_02_0011', 'nh3_W22_02_0016', 'nh3_W22_02_0020', 'nh3_W22_02_0025', 'nh3_W22_02_0049', 'nh3_W22_03_0001', 'nh3_W22_03_0002', 'nh3_W22_03_0003', 'nh3_W22_03_0006', 'nh3_W22_03_0008', 'nh3_W22_03_0010', 'nh3_W22_03_0011', 'nh3_W22_03_0012', 'nh3_W22_03_0022', 'nh3_W22_03_0025', 'nh3_W22_03_0028', 'nh3_W22_03_0043', 'nh3_W22_03_0051', 'nh3_W22_04_0001', 'nh3_W22_04_0002', 'nh3_W22_04_0003', 'nh3_W22_04_0007', 'nh3_W22_04_0010', 'nh3_W22_04_0018', 'nh3_W22_04_0021', 'nh3_W22_04_0031', 'nh3_W22_04_0033', 'nh3_W22_04_0039', 'nh3_W22_05_0003', 'nh3_W22_05_0005', 'nh3_W22_05_0010', 'nh3_W22_05_0014', 'nh3_W22_05_0029', 'nh3_W22_06_0010', 'nh3_W22_06_0011', 'nh3_W22_06_0013', 'nh3_W22_06_0014', 'nh3_W22_09_0009', 'nh3_W22_09_0012', 'nh3_W22_09_0018', 'nh3_W22_09_0022', 'nh3_W22_09_0023', 'nh3_W22_09_0024', 'nh3_W22_09_0034', 'nh3_W22_09_0036', 'nh3_W22_09_0041', 'nh3_W22_09_0048', 'nh3_W22_09_0059', 'nh3_W22_09_0061', 'nh3_W22_10_0001', 'nh3_W22_10_0004', 'nh3_W22_10_0008', 'nh3_W22_10_0009', 'nh3_W22_10_0012', 'nh3_W22_10_0014', 'nh3_W22_10_0028', 'nh3_W22_10_0031', 'nh3_W22_10_0034', 'nh3_W22_10_0035', 'nh3_W22_10_0038', 'nh3_W22_11_0002', 'nh3_W22_11_0004', 'nh3_W22_11_0006', 'nh3_W22_11_0007', 'nh3_W22_11_0013', 'nh3_W22_11_0014', 'nh3_W22_11_0018', 'nh3_W22_11_0023', 'nh3_W22_11_0034', 'nh3_W22_11_0046', 'nh3_W22_12_0005', 'nh3_W22_12_0007', 'nh3_W22_12_0010', 'nh3_W22_12_0016', 'nh3_W22_12_0017', 'nh3_W22_13_0005', 'nh3_W22_13_0009', 'nh3_W22_13_0012', 'nh3_W22_13_0018', 'nh3_W22_13_0022', 'nh3_W22_13_0024', 'nh3_W22_13_0027', 'nh3_W22_14_0001', 'nh3_W22_14_0007', 'nh3_W22_14_0008', 'nh3_W22_14_0013', 'nh3_W22_14_0020', 'nh3_W22_14_0050', 'nh3_W22_14_0053', 'nh3_W22_15_0003', 'nh3_W22_15_0009', 'nh3_W22_15_0011', 'nh3_W22_15_0013', 'nh3_W22_15_0014', 'nh3_W22_15_0015', 'nh3_W22_15_0017', 'nh3_W22_15_0019', 'nh3_W22_15_0020', 'nh3_W22_15_0022', 'nh3_W22_15_0023', 'nh3_W22_15_0024', 'nh3_W22_15_0037', 'nh3_W22_16_0001', 'nh3_W22_16_0008', 'nh3_W22_16_0012', 'nh3_W22_16_0018'], 
        'W37': ['nh3_W37_01_0001', 'nh3_W37_01_0003', 'nh3_W37_01_0004', 'nh3_W37_01_0006', 'nh3_W37_01_0010', 'nh3_W37_01_0017', 'nh3_W37_01_0019', 'nh3_W37_01_0021', 'nh3_W37_01_0024', 'nh3_W37_01_0028', 'nh3_W37_01_0033', 'nh3_W37_01_0035', 'nh3_W37_01_0040', 'nh3_W37_01_0046', 'nh3_W37_01_0051', 'nh3_W37_01_0053', 'nh3_W37_01_0054', 'nh3_W37_01_0056', 'nh3_W37_01_0057', 'nh3_W37_01_0065', 'nh3_W37_02_0002', 'nh3_W37_02_0004', 'nh3_W37_02_0006', 'nh3_W37_02_0007', 'nh3_W37_02_0010', 'nh3_W37_02_0011', 'nh3_W37_02_0015', 'nh3_W37_02_0027', 'nh3_W37_02_0030', 'nh3_W37_02_0031', 'nh3_W37_02_0032', 'nh3_W37_02_0044', 'nh3_W37_02_0047', 'nh3_W37_03_0001', 'nh3_W37_03_0004', 'nh3_W37_03_0011', 'nh3_W37_03_0016', 'nh3_W37_03_0017', 'nh3_W37_03_0020', 'nh3_W37_03_0021', 'nh3_W37_03_0022', 'nh3_W37_03_0023', 'nh3_W37_03_0030', 'nh3_W37_03_0032', 'nh3_W37_03_0043', 'nh3_W37_03_0049', 'nh3_W37_03_0050', 'nh3_W37_03_0054', 'nh3_W37_03_0059', 'nh3_W37_04_0001', 'nh3_W37_04_0008', 'nh3_W37_04_0009', 'nh3_W37_04_0014', 'nh3_W37_04_0019', 'nh3_W37_04_0020', 'nh3_W37_04_0024', 'nh3_W37_04_0026', 'nh3_W37_04_0029', 'nh3_W37_04_0032', 'nh3_W37_04_0037', 'nh3_W37_04_0038', 'nh3_W37_05_0004', 'nh3_W37_05_0005', 'nh3_W37_05_0011', 'nh3_W37_05_0013', 'nh3_W37_05_0017', 'nh3_W37_05_0020', 'nh3_W37_05_0021', 'nh3_W37_05_0026', 'nh3_W37_05_0028', 'nh3_W37_05_0029', 'nh3_W37_05_0030', 'nh3_W37_05_0033', 'nh3_W37_05_0035', 'nh3_W37_05_0037', 'nh3_W37_05_0040', 'nh3_W37_06_0009', 'nh3_W37_06_0012', 'nh3_W37_06_0013', 'nh3_W37_06_0020', 'nh3_W37_06_0022', 'nh3_W37_06_0034', 'nh3_W37_06_0039', 'nh3_W37_06_0070', 'nh3_W37_07_0003', 'nh3_W37_07_0004', 'nh3_W37_07_0005', 'nh3_W37_07_0006', 'nh3_W37_07_0007', 'nh3_W37_07_0010', 'nh3_W37_07_0011', 'nh3_W37_07_0017', 'nh3_W37_07_0021', 'nh3_W37_07_0026', 'nh3_W37_07_0027', 'nh3_W37_07_0028', 'nh3_W37_07_0029', 'nh3_W37_07_0034', 'nh3_W37_07_0036', 'nh3_W37_07_0042', 'nh3_W37_07_0047', 'nh3_W37_07_0054', 'nh3_W37_07_0058', 'nh3_W37_08_0002', 'nh3_W37_08_0004', 'nh3_W37_08_0005', 'nh3_W37_08_0006', 'nh3_W37_08_0008', 'nh3_W37_08_0010', 'nh3_W37_08_0011', 'nh3_W37_08_0018', 'nh3_W37_08_0019', 'nh3_W37_08_0023', 'nh3_W37_08_0037', 'nh3_W37_08_0045', 'nh3_W37_08_0046', 'nh3_W37_08_0072', 'nh3_W37_08_0074', 'nh3_W37_08_0081', 'nh3_W37_08_0082', 'nh3_W37_08_0100', 'nh3_W37_09_0003', 'nh3_W37_09_0009', 'nh3_W37_09_0011', 'nh3_W37_09_0018', 'nh3_W37_09_0019', 'nh3_W37_09_0023', 'nh3_W37_09_0026', 'nh3_W37_09_0029', 'nh3_W37_09_0034', 'nh3_W37_09_0041', 'nh3_W37_09_0049', 'nh3_W37_09_0061', 'nh3_W37_09_0062']
, 
        'W60': ['nh3_W60_01_0001', 'nh3_W60_01_0003', 'nh3_W60_01_0004', 'nh3_W60_01_0006', 'nh3_W60_01_0008', 'nh3_W60_01_0009', 'nh3_W60_01_0017', 'nh3_W60_01_0022', 'nh3_W60_01_0028', 'nh3_W60_01_0033', 'nh3_W60_01_0036', 'nh3_W60_01_0040', 'nh3_W60_01_0050', 'nh3_W60_01_0056', 'nh3_W60_01_0068', 'nh3_W60_02_0003', 'nh3_W60_02_0004', 'nh3_W60_02_0006', 'nh3_W60_02_0007', 'nh3_W60_02_0009', 'nh3_W60_02_0011', 'nh3_W60_02_0012', 'nh3_W60_02_0015', 'nh3_W60_02_0020', 'nh3_W60_02_0025', 'nh3_W60_02_0028', 'nh3_W60_02_0029', 'nh3_W60_02_0032', 'nh3_W60_02_0034', 'nh3_W60_02_0036', 'nh3_W60_02_0037', 'nh3_W60_02_0039', 'nh3_W60_02_0042', 'nh3_W60_02_0044', 'nh3_W60_02_0048', 'nh3_W60_02_0049', 'nh3_W60_02_0051', 'nh3_W60_02_0055', 'nh3_W60_02_0059', 'nh3_W60_02_0062', 'nh3_W60_02_0067', 'nh3_W60_02_0072', 'nh3_W60_02_0077', 'nh3_W60_03_0003', 'nh3_W60_03_0005', 'nh3_W60_03_0008', 'nh3_W60_03_0009', 'nh3_W60_03_0011', 'nh3_W60_03_0012', 'nh3_W60_03_0016', 'nh3_W60_03_0018', 'nh3_W60_03_0022', 'nh3_W60_03_0023', 'nh3_W60_03_0026', 'nh3_W60_03_0027', 'nh3_W60_03_0028', 'nh3_W60_03_0030', 'nh3_W60_03_0036', 'nh3_W60_03_0040', 'nh3_W60_03_0046', 'nh3_W60_03_0070', 'nh3_W60_03_0074', 'nh3_W60_03_0078', 'nh3_W60_03_0085', 'nh3_W60_04_0003', 'nh3_W60_04_0004', 'nh3_W60_04_0005', 'nh3_W60_04_0006', 'nh3_W60_04_0009', 'nh3_W60_04_0010', 'nh3_W60_04_0011', 'nh3_W60_04_0024', 'nh3_W60_04_0025', 'nh3_W60_04_0026', 'nh3_W60_04_0037', 'nh3_W60_04_0052', 'nh3_W60_04_0054', 'nh3_W60_04_0060', 'nh3_W60_04_0062', 'nh3_W60_04_0067', 'nh3_W60_04_0100', 'nh3_W60_05_0001', 'nh3_W60_05_0003', 'nh3_W60_05_0005', 'nh3_W60_05_0006', 'nh3_W60_05_0009', 'nh3_W60_05_0010', 'nh3_W60_05_0013', 'nh3_W60_05_0017', 'nh3_W60_05_0020', 'nh3_W60_05_0023', 'nh3_W60_05_0025', 'nh3_W60_05_0026', 'nh3_W60_05_0032', 'nh3_W60_05_0034', 'nh3_W60_05_0048', 'nh3_W60_05_0065', 'nh3_W60_05_0072', 'nh3_W60_05_0089', 'nh3_W60_05_0101', 'nh3_W60_06_0002', 'nh3_W60_06_0004', 'nh3_W60_06_0005', 'nh3_W60_06_0008', 'nh3_W60_06_0009', 'nh3_W60_06_0010', 'nh3_W60_06_0015', 'nh3_W60_06_0016', 'nh3_W60_06_0017', 'nh3_W60_06_0024', 'nh3_W60_06_0025', 'nh3_W60_06_0030', 'nh3_W60_06_0031', 'nh3_W60_06_0049', 'nh3_W60_06_0050', 'nh3_W60_06_0053']

 } 

nh3_filter_wpbe = { 'W12': ['nh3_W12_01_0004', 'nh3_W12_01_0006', 'nh3_W12_01_0009', 'nh3_W12_01_0015', 'nh3_W12_02_0002', 'nh3_W12_02_0006', 'nh3_W12_02_0020', 'nh3_W12_02_0029', 'nh3_W12_03_0001', 'nh3_W12_03_0003', 'nh3_W12_03_0008', 'nh3_W12_03_0009', 'nh3_W12_03_0010', 'nh3_W12_03_0012', 'nh3_W12_03_0015', 'nh3_W12_03_0019', 'nh3_W12_04_0005', 'nh3_W12_04_0009', 'nh3_W12_04_0011', 'nh3_W12_04_0013', 'nh3_W12_04_0019', 'nh3_W12_05_0003', 'nh3_W12_05_0016', 'nh3_W12_05_0021', 'nh3_W12_05_0022', 'nh3_W12_06_0001', 'nh3_W12_06_0002', 'nh3_W12_06_0018', 'nh3_W12_06_0019', 'nh3_W12_06_0022', 'nh3_W12_06_0023', 'nh3_W12_06_0024', 'nh3_W12_07_0001', 'nh3_W12_08_0004', 'nh3_W12_08_0009', 'nh3_W12_08_0020', 'nh3_W12_08_0026', 'nh3_W12_08_0029', 'nh3_W12_08_0036', 'nh3_W12_08_0041', 'nh3_W12_09_0004', 'nh3_W12_09_0006', 'nh3_W12_09_0008', 'nh3_W12_10_0003', 'nh3_W12_10_0005', 'nh3_W12_10_0006', 'nh3_W12_10_0010', 'nh3_W12_11_0001', 'nh3_W12_11_0002', 'nh3_W12_11_0004', 'nh3_W12_11_0008', 'nh3_W12_11_0012', 'nh3_W12_11_0014', 'nh3_W12_11_0023', 'nh3_W12_12_0003', 'nh3_W12_12_0004', 'nh3_W12_12_0006', 'nh3_W12_12_0011']
,
'W22':['nh3_W22_01_0003', 'nh3_W22_01_0005', 'nh3_W22_01_0007', 'nh3_W22_01_0017', 'nh3_W22_01_0031', 'nh3_W22_01_0050', 'nh3_W22_01_0056', 'nh3_W22_02_0002', 'nh3_W22_02_0004', 'nh3_W22_02_0010', 'nh3_W22_02_0011', 'nh3_W22_02_0016', 'nh3_W22_02_0020', 'nh3_W22_02_0025', 'nh3_W22_02_0049', 'nh3_W22_03_0001', 'nh3_W22_03_0002', 'nh3_W22_03_0003', 'nh3_W22_03_0006', 'nh3_W22_03_0008', 'nh3_W22_03_0010', 'nh3_W22_03_0011', 'nh3_W22_03_0012', 'nh3_W22_03_0022', 'nh3_W22_03_0025', 'nh3_W22_03_0028', 'nh3_W22_03_0043', 'nh3_W22_03_0051', 'nh3_W22_04_0001', 'nh3_W22_04_0002', 'nh3_W22_04_0003', 'nh3_W22_04_0007', 'nh3_W22_04_0010', 'nh3_W22_04_0018', 'nh3_W22_04_0021', 'nh3_W22_04_0031', 'nh3_W22_04_0033', 'nh3_W22_04_0039', 'nh3_W22_05_0003', 'nh3_W22_05_0005', 'nh3_W22_05_0010', 'nh3_W22_05_0014', 'nh3_W22_05_0029', 'nh3_W22_06_0010', 'nh3_W22_06_0011', 'nh3_W22_06_0013', 'nh3_W22_06_0014', 'nh3_W22_09_0009', 'nh3_W22_09_0012', 'nh3_W22_09_0018', 'nh3_W22_09_0022', 'nh3_W22_09_0023', 'nh3_W22_09_0024', 'nh3_W22_09_0034', 'nh3_W22_09_0036', 'nh3_W22_09_0041', 'nh3_W22_09_0048', 'nh3_W22_09_0059', 'nh3_W22_09_0061', 'nh3_W22_10_0001', 'nh3_W22_10_0004', 'nh3_W22_10_0008', 'nh3_W22_10_0009', 'nh3_W22_10_0012', 'nh3_W22_10_0014', 'nh3_W22_10_0028', 'nh3_W22_10_0031', 'nh3_W22_10_0034', 'nh3_W22_10_0035', 'nh3_W22_10_0038', 'nh3_W22_11_0002', 'nh3_W22_11_0004', 'nh3_W22_11_0006', 'nh3_W22_11_0007', 'nh3_W22_11_0013', 'nh3_W22_11_0014', 'nh3_W22_11_0018', 'nh3_W22_11_0023', 'nh3_W22_11_0034', 'nh3_W22_11_0046', 'nh3_W22_12_0005', 'nh3_W22_12_0007', 'nh3_W22_12_0010', 'nh3_W22_12_0016', 'nh3_W22_12_0017', 'nh3_W22_13_0005', 'nh3_W22_13_0009', 'nh3_W22_13_0012', 'nh3_W22_13_0018', 'nh3_W22_13_0022', 'nh3_W22_13_0024', 'nh3_W22_13_0027', 'nh3_W22_14_0001', 'nh3_W22_14_0007', 'nh3_W22_14_0008', 'nh3_W22_14_0013', 'nh3_W22_14_0020', 'nh3_W22_14_0050', 'nh3_W22_14_0053', 'nh3_W22_15_0003', 'nh3_W22_15_0009', 'nh3_W22_15_0011', 'nh3_W22_15_0013', 'nh3_W22_15_0014', 'nh3_W22_15_0015', 'nh3_W22_15_0017', 'nh3_W22_15_0019', 'nh3_W22_15_0020', 'nh3_W22_15_0022', 'nh3_W22_15_0023', 'nh3_W22_15_0024', 'nh3_W22_15_0037', 'nh3_W22_16_0001', 'nh3_W22_16_0008', 'nh3_W22_16_0012', 'nh3_W22_16_0018']
 ,
'W37':['nh3_W37_01_0001', 'nh3_W37_01_0003', 'nh3_W37_01_0004', 'nh3_W37_01_0006', 'nh3_W37_01_0010', 'nh3_W37_01_0017', 'nh3_W37_01_0019', 'nh3_W37_01_0021', 'nh3_W37_01_0024', 'nh3_W37_01_0028', 'nh3_W37_01_0033', 'nh3_W37_01_0035', 'nh3_W37_01_0040', 'nh3_W37_01_0046', 'nh3_W37_01_0051', 'nh3_W37_01_0053', 'nh3_W37_01_0054', 'nh3_W37_01_0056', 'nh3_W37_01_0057', 'nh3_W37_01_0065', 'nh3_W37_02_0002', 'nh3_W37_02_0004', 'nh3_W37_02_0006', 'nh3_W37_02_0007', 'nh3_W37_02_0010', 'nh3_W37_02_0011', 'nh3_W37_02_0015', 'nh3_W37_02_0027', 'nh3_W37_02_0030', 'nh3_W37_02_0031', 'nh3_W37_02_0032', 'nh3_W37_02_0044', 'nh3_W37_02_0047', 'nh3_W37_03_0001', 'nh3_W37_03_0004', 'nh3_W37_03_0011', 'nh3_W37_03_0016', 'nh3_W37_03_0017', 'nh3_W37_03_0020', 'nh3_W37_03_0021', 'nh3_W37_03_0022', 'nh3_W37_03_0023', 'nh3_W37_03_0030', 'nh3_W37_03_0032', 'nh3_W37_03_0043', 'nh3_W37_03_0049', 'nh3_W37_03_0050', 'nh3_W37_03_0054', 'nh3_W37_03_0059', 'nh3_W37_04_0001', 'nh3_W37_04_0008', 'nh3_W37_04_0009', 'nh3_W37_04_0014', 'nh3_W37_04_0019', 'nh3_W37_04_0020', 'nh3_W37_04_0024', 'nh3_W37_04_0026', 'nh3_W37_04_0029', 'nh3_W37_04_0032', 'nh3_W37_04_0037', 'nh3_W37_04_0038', 'nh3_W37_05_0004', 'nh3_W37_05_0005', 'nh3_W37_05_0011', 'nh3_W37_05_0013', 'nh3_W37_05_0017', 'nh3_W37_05_0020', 'nh3_W37_05_0021', 'nh3_W37_05_0026', 'nh3_W37_05_0028', 'nh3_W37_05_0029', 'nh3_W37_05_0030', 'nh3_W37_05_0033', 'nh3_W37_05_0035', 'nh3_W37_05_0037', 'nh3_W37_05_0040', 'nh3_W37_06_0009', 'nh3_W37_06_0012', 'nh3_W37_06_0013', 'nh3_W37_06_0020', 'nh3_W37_06_0022', 'nh3_W37_06_0034', 'nh3_W37_06_0039', 'nh3_W37_06_0070', 'nh3_W37_07_0003', 'nh3_W37_07_0004', 'nh3_W37_07_0005', 'nh3_W37_07_0006', 'nh3_W37_07_0007', 'nh3_W37_07_0010', 'nh3_W37_07_0011', 'nh3_W37_07_0017', 'nh3_W37_07_0021', 'nh3_W37_07_0026', 'nh3_W37_07_0027', 'nh3_W37_07_0028', 'nh3_W37_07_0029', 'nh3_W37_07_0034', 'nh3_W37_07_0036', 'nh3_W37_07_0042', 'nh3_W37_07_0047', 'nh3_W37_07_0054', 'nh3_W37_07_0058']

 
 }
 
