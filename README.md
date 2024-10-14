![logo_BEEP](https://user-images.githubusercontent.com/7481702/178007641-1f4260b3-dd34-4e39-9a51-286b076c5ea8.png)



[//]: # (Badges)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/BEEP/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/BEEP/branch/master)



BEEP is a binding energy evaluation platform and database for molecules on interstellar ice-grain mantels.

For data query for the BEEP database you only need to install the QCPortal python module and follow the tutorial

`conda install -c conda-forge  qcportal=0.15  pydantic=1.9*`

If you would like to create your own database using the BEEP protocol or if you would like 
to contribute to the BEEP database you will need to install the QCFractal module. It is best intall
QCFractal into a fresh environment.

`conda create -n beep -c psi4 -c main qcfractal=0.15 psi4=1.6* pydantic=1.9* intel-openmp geometric`

Once you installed QCFractal or QCPortal, to install the BEEP libraries run:

`python -m pip install .`

For examples of data query and data generation head to the tutorials folder in this repository.


### Copyright

Copyright (c) 2022, Stefan Vogt-Geisse, Giulia M. Bovolenta


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
