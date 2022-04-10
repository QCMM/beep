BEEP (Binding Energy Evaluation Platform)
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/beep/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/beep/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/BEEP/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/BEEP/branch/master)


A binding energy evaluation platform and database for molecules on interstellar ice-grain mantels.

If you would like to Query the platform using the BindingParadies class or QCPortal you 
will need to install the  QCPortal python module. The easiest way to install it it is using the 
[Mamba Python pacakge manager](https://mamba.readthedocs.io/en/latest/index.html). First install mamba using conda into
your base directory.

`conda install mamba -n base -c conda-forge`

Then you can install QCPortal by running. 

`mamba install -c conda-forge qcportal`

If you would like to create your own database using the BEEP protocol or if you would like 
to contribute to the BEEP database you will need to install the QCFractal module. It is best intall
QCFractal into a fresh environment.

`mamba install --name qcfrc -c conda-forge -c psi4 qcfractal psi4`

Once you installed QCFractal or QCPortal, to install the BEEP libraries run:

`python setup.py install`

For examples of data query and data generation head to the tutorials folder in this repository.


### Copyright

Copyright (c) 2022, Stefan Vogt-Geisse, Giulia Bovolenta


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
