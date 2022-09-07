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

## Equivalence and agreement between measurements
### Bland-Altman revised plot
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

### Regression residuals diagnostics 
Creates a linear model old vs. new methodology and evaluates the residuals
```Python
from equiv_med.EQU import eq_Regr
my_regr=eq_Regr.Regr_diagn(var1,var2)
my_regr.run_diagnostic([0.05,0.1,0.2]) # Cook values
my_regr.influential_points() #DIFFITS & DFBETAS with default thresholds
```

### CatEyes plots 
Visualization and comparison graph of 95% confidence intervals for the two measurements
```Python
from equiv_med.CI import ci_Cateyes
my_ce=ci_Cateyes.Cat_Eye_2var(var1,var2)
my_ce.run_ce(95) # C.I. value
my_ce.run_ce_unbiased(95) # C.I. value
my_ce.single_cat_eye(var1,95) # C.I. value
```

### Cohen's d
Effect size (automatic calculation of the right formula based on input data) and visualization
```Python
from equiv_med.ES import Cohen_family
D_meas=Cohen_family.Cohen_es(var1,var2,design='indep')
print('Cohen D :', D_meas.Cohen_d())
print('Lambda parameter (non centrality) :', D_meas.lambda_par())
print('Variance of Cohen D :',D_meas.standard_error_cohen() )
print('CI :',D_meas.CI_cohen())
D_meas.nonoverlap_measures()
D_meas.plotting()
```

### Standard TOST
Equivalence testing
```Python
from equiv_med.EIS import Standard_Tost
eq_tost=Standard_Tost.EQU(var1,var2,-5.5,5.5) # [-5.5;5.5] are the user-defined regulatory boundaries simmetric to zero
eq_tost.run_Tost_indep() # Independent samples
eq_tost=Standard_Tost.EQU(var1,var2,-5.5,5.5) # [-5.5;5.5] are the user-defined regulatory boundaries simmetric to zero
eq_tost.run_Tost_dep() # Dependent samples
```

### Special implementation of TOST tests
TOST implemented as in [20]
```Python
from equiv_med.EIS import Tost_WS
out1=Tost_WS.WS_eq(var1,var2,5)
out1.run_TOST()
out1.power_TOST()
out1.opt_sample_size()
```
TOST implemented as in [17,18]
```Python
from equiv_med.EIS import Tost_Alt
test_a=Tost_Alt.TOST_T(var1,var2)
test_a.run_TOST_T()
test_a.run_TOST_MW()
```
TOST implemented as in ()
```Python
from equiv_med.EIS import Tost_NCP
tost_res=Tost_NCP.Tost_paired(var1,var2,5.5)
tost_res.run_tost()
tost_res.stat_power()
tost_res.opt_sample_size()
```
## Special case functions
### Stacked representations of confidence intervals 
The graph simulates a certain number of confidence intervals: it helps determine the percentage of values falling below a regulatory boundary
```Python
import numpy as np
from equiv_med.CI import ci_find_margin
variab=np.random.normal(loc = 10, scale = 2.3, size = (100,))
margin=np.mean(variab)
margin_help=ci_find_margin.Id_margin(variab)
# Simulating Coefficient of Variation (known variability between measurements)
margin_help.decision_perc(9.75,noise_variability=0.05) # regulatory boundary inserted by user is 9.75
```
