# Confronting Macro-Finance model with Data (2020)
This GitHub repository contains code to solve the class of models from the paper Confronting Macro-Finance model with Data (2020). Please refer to the paper for the model framework. 

## Requirements
You need Python 3.5 or later to run the files. In Ubuntu, Mint and Debian you can install Python 3 like this:
```
$ sudo apt-get install python3 python3-pip 
```
For other Linux flavors, macOS and Windows, packages are available at
https://www.python.org/getit/

The following python packages need to be installed and imported: numpy 1.19.1, pandas 1.0.1, matplotlib 3.3.0, scipy 1.5.2, statsmodels 0.11.0, dill 0.3.2, seaborn 0.10.0. The packages can be installed using pip like this
```
pip install pandas
```

## Code description
1) main.py: Imports the simulation file and plots main figures in the paper.
2) model_class.py: Solves the incomplete-market capital mis-allocation model with CRRA utility function. Dependencies: finite_difference.py. 
3) model_recursive_class.py: Solves the incomplete-market capital mis-allocation model with recursive utiltity and IES=1. Risk aversion can be any positive value. Dependencies: finite_difference.py
4) model_recursive_general_class.py: Solves the incomplete-market capital mis-allocation model with recursive utiltity and IES different from unity. Risk aversion can be any positive value. Dependencies: finite_difference.py
5) finite_difference.py: Contains modules to solve 1-D partial differential equations using implicit and explicit methods with up-winding scheme. 
6) simulation_model_class.py: Performs model simulation and computes moments and stationary distribution. This is a generic file that can be used for any type of utility functions (log, CRRA, Recursive with IES=1, Recursive with IES different from 1). Dependencies: model_class.py, model_recursive_class.py, model_recursive_general.py, finite_difference.py, interpolate_var.py
7) interpolate_var.py: Contains modules to interpolate functions using different methods.
8) empirical.py: Computes empirical risk premium moments from the data and plots autocorrelation. 


## Data description
1) pd_shiller.csv: Historical price-dividend ratio data from Robert Shiller's website (http://www.econ.yale.edu/~shiller/data.htm)
2) predictor_data.csv: Equity risk premium predictor data from Amit Goyal's website (http://www.hec.unil.ch/agoyal/)
3) rf.csv: Historical T-bill rate 
4) USREC.csv: US recessionary periods data from NBER website (https://www.nber.org/cycles.html)

## Questions
If you have any questions with the code, please contact goutham.gopalakrishna@epfl.ch
