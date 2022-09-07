# equiv_med
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6504217.svg)](https://doi.org/10.5281/zenodo.6504217)<br />
Python functions for clinical **equivalence testing**. An overview is provided below:<br />

![Function_Overview](/schema2.png)

The repository contains Python functions to produce novel graphs for biomedical equivalence testing. Visualization enhances the interpretation and storytelling of statistical tests. Each function checks the preliminary assumptions of the tests automatically. The scripts were subdivided into four folders following the scheme in the figure and paired with console outputs. Users interested in running bio-similarity analysis can download the code and follow the instructions contained in the referenced manuscript. 

# Installation
```Python
pip install git+https://github.com/m89p067/equiv_med.git
```
Please check pip, git and setuptools are properly installed on your system.

# Explanations, references, and examples included in:<br />
M. Nascimben and L. Rimondini <br />
*Visually enhanced python function for equality of measurement assessment*<br />
Presented during the IEEE Fedcsis 2022 conference (4-7 September, Sofia, Bulgaria)<br />

# Minimal working examples
Create two random vectors simulating the outputs of a new and an old device (could be also data from two assays, two drugs, two equipments or organic samples). One might be reference method or a gold-standard
```Python
import numpy as np
mu1,sigma1=100.2,62.6
mu2,sigma2=130.9,80.2
var1= np.random.normal(mu1, sigma1, 300)
var2= np.random.normal(mu2, sigma2, 300)
```
## Bland-Altman revised plot
Evalutes the laboratory outputs of two devices or values of two organic samples and checks the agreement
```Python
from equiv_med.EQU import eq_BA
my_BA=eq_BA.BA_analysis(var1,var2)
#Bland-Altman plot
my_BA.run_analysis() # default 95% of the difference will lie in this interval [revised plot]
# Evaluate sample size and assurance probability of exact agreement limist (as in [10])
#Exact limits of agreement sample size (as in [10])
out1=my_BA.exact_Bound_sample_size(mu1,sigma1,len(var1),95,0.05)
out2=my_BA.exact_Bound_sample_size(mu2,sigma2,len(var2),95,0.05)
#Exact limits of agreement assurance (as in [10])
out3=my_BA.exact_Bound_assurance(mu1,sigma1,len(var1),95,0.05,0.9)
out4=my_BA.exact_Bound_assurance(mu2,sigma2,len(var2),95,0.05,0.9)
#In case of repeated measures
my_BA.minimal_detectable_change() #output also Minimal Detectable Change 
```

## Regression residuals diagnostics 
Creates a linear model old vs. new methodology and evaluates the residuals
```Python
from equiv_med.EQU import eq_Regr
my_regr=eq_Regr.Regr_diagn(var1,var2)
my_regr.run_diagnostic([0.05,0.1,0.2]) # Cook values
my_regr.influential_points() #DIFFITS & DFBETAS with default thresholds
```
