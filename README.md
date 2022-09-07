# equiv_med
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6504217.svg)](https://doi.org/10.5281/zenodo.6504217)<br />
Python functions for clinical **equivalence testing**. An overview is provided below:<br />

![Function_Overview](/schema2.png)

The repository contains Python functions to produce novel graphs for biomedical equivalence testing. Visualization enhances the interpretation and storytelling of statistical tests. Each function checks the preliminary assumptions of the tests automatically. The scripts were subdivided into four folders following the scheme in the figure and paired with console outputs. Users interested in running bio-similarity analysis can download the code and follow the instructions contained in the referenced manuscript. 

# Installation
```Python
pip install git+https://github.com/m89p067/equivmed.git
```
Please check pip, git and setuptools are properly installed on your system.

# Explanations, references, and examples included in:<br />
M. Nascimben and L. Rimondini <br />
*Visually enhanced python function for equality of measurement assessment*<br />
Presented during the IEEE Fedcsis 2022 conference (4-7 September, Sofia, Bulgaria)<br />

# Minimal working examples
## Bland-Altman revised plot
Evalutes the laboratory outputs of two devices or values of two organic samples and checks the agreement
```Python
import numpy as np
from equivmed.EQU import eq_BA
mu1,sigma1=100.2,62.6
mu2,sigma2=130.9,80.2
var1= np.random.normal(mu1, sigma1, 300)
var2= np.random.normal(mu2, sigma2, 300)
my_BA=eq_BA.BA_analysis(var1,var2)
#Bland-Altman plot
my_BA.run_analysis() # default 95% of the difference will lie in this interval [revised plot]
#In case of repeated measures
my_BA.minimal_detectable_change() #output also Minimal Detectable Change 
```

## Regression residuals diagnostics 
Creates a linear model old vs. new methodology and evaluates the residuals
```Python
import numpy as np
from equivmed.EQU import eq_Regr
mu1,sigma1=100.2,62.6
mu2,sigma2=130.9,80.2
var1= np.random.normal(mu1, sigma1, 300)
var2= np.random.normal(mu2, sigma2, 300)
my_regr=eq_Regr.Regr_diagn(var1,var2)
my_regr.run_diagnostic([0.05,0.1,0.2]) # Cook values
my_regr.influential_points() #DIFFITS & DFBETAS with default thresholds
```
