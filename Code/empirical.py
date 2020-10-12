import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.api as tsa
import statsmodels.regression.linear_model as sm
import statsmodels.api as sm_filters
from statsmodels.api import add_constant
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
import statsmodels.regression.linear_model as sm
import statsmodels.api as sm_filters
from statsmodels.api import add_constant
from statsmodels.iolib.summary2 import summary_col
import statistics
from statsmodels.graphics import tsaplots

'''
This file performs risk premium estimation and computes first and second moments of premia from the data
Input: price-dividend (Shiller's website), risk premia predictors (Amit Goyal's website), NBER recessionary periods
Output: autocorrelation plots, first and second moments of equity risk premium.       
'''
#load risk free rate
rf = pd.read_csv('../Data/rf.csv',sep=';')
rf = rf.set_index(pd.to_datetime(pd.to_datetime(rf.date,format='%Y%m'),format='%Y-%m'))
rf.drop(columns=['date'],inplace=True)
rf.rf = rf.rf
freq = 12

#load Shiller data
pd_data = pd.read_csv('../Data/pd_shiller.csv',sep=';')
pd_data = pd_data.set_index(pd.to_datetime(pd.to_datetime(pd_data.Date,format='%Y-%m'),format='%Y-%m'))
pd_data.drop(columns=['Date'],inplace=True)

#load NBER recessionary data
usRec = pd.read_csv('../Data/USREC.csv')
usRec = usRec.set_index(pd.to_datetime(pd.to_datetime(usRec.DATE,format='%Y-%m-%d'),format='%Y-%m'))
usRec.drop(columns=['DATE'],inplace=True)
usRec['crisis'] = 0

#Manually set crises periods
usRec.loc['1894-01-01':'1894-06-01','crisis'] = 1
usRec.loc['1930-01-01':'1830-12-01','crisis'] = 1
usRec.loc['1982-01-01':'1982-11-01'] = 1
usRec.loc['2008-09-01':'2009-06-01','crisis'] = 1

#predictor variables from Amit Goyal's website
predictor_data = pd.read_csv('../Data/predictor_data.csv',sep=';')
predictor_data = predictor_data.set_index(pd.to_datetime(pd.to_datetime(predictor_data.yyyymm,format='%Y%m'),format='%Y-%m'))

#create price-dividend ratio
pd_data['dp'] = pd_data.D/pd_data.P
pd_data = pd_data.merge(predictor_data['CRSP_SPvwx'], how='left', right_index=True, left_index=True)
pd_data['Ret'] = np.log(pd_data.P).diff()
#pd_data['Ret'] = pd_data['CRSP_SPvwx']
pd_data['Ret_1qtr'] = pd_data.Ret.rolling(1).sum().shift(-1) 
pd_data['Ret_1yr'] = pd_data.Ret.rolling(freq).sum().shift(-freq) 
pd_data['Ret_5yr'] = pd_data.Ret.rolling(freq*5).sum().shift(-freq*5) 
tsaplots.plot_acf(pd_data.Ret[1:],lags=40)
lags = np.arange(1,50)

pd_data = pd_data.merge(rf,how='left',left_index = True, right_index=True)
pd_data = pd_data.merge(usRec,how='left',left_index = True, right_index=True)
pd_data['exRet'] = pd_data.Ret - pd_data.rf

#create autocorrelation plots
start_date_acf = '1926'
end_date_acf = '2013'
acf_coeff1, acf_coeff2, acf_coeff3 = [],[],[]
acf_tstat1, acf_tstat2, acf_tstat3 = [], [], []
for j in range(len(lags)):
    #acf_indep = np.column_stack([pd_data.Ret.values-rf.rf.values, (pd_data.Ret.values-rf.rf.values) * usRec.loc['1871':'2018','USREC'], (pd_data.Ret.values-rf.rf.values) * usRec.loc['1871':'2018','crisis'] ])
    acf_indep = np.column_stack([pd_data[start_date_acf:end_date_acf].exRet.values, pd_data[start_date_acf:end_date_acf].exRet.values * pd_data[start_date_acf:end_date_acf].USREC, pd_data[start_date_acf:end_date_acf].exRet.values * pd_data[start_date_acf:end_date_acf].crisis])
    acf_exog = sm_filters.add_constant(acf_indep)
    acf_reg = sm.OLS(pd_data[start_date_acf:end_date_acf].exRet[lags[j]:],acf_exog[0:-lags[j]],missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : int(0.75*acf_exog.shape[0]**0.33)}) #int(0.75* acf_exog.shape[0]**0.33)
    acf_coeff1.append(acf_reg.params[1])
    acf_tstat1.append(acf_reg.params[1]/acf_reg.bse[1])
    if len(acf_reg.params)>2:
        acf_coeff2.append(acf_reg.params[2])
        acf_tstat2.append(acf_reg.params[2]/acf_reg.bse[2])
    if len(acf_reg.params)>3:
        acf_coeff3.append(acf_reg.params[3])
        acf_tstat3.append(acf_reg.params[3]/acf_reg.bse[3])
    
plt.figure()
plt.bar(np.arange(1,50),acf_tstat1)
plt.xlabel('lags (months)')
plt.ylabel(r'$\beta_{1}(h): t-stat$',fontsize=15)
plt.ylim(-3,12)
plt.axhline(y=1.96,linewidth=1)
plt.axhline(y=-1.96,linewidth=1)
plt.title('Normal period', fontsize=20)
plt.savefig('../output/acf1.png')
plt.show()

plt.figure()
plt.bar(np.arange(1,50),np.array(acf_tstat1)+ np.array(acf_tstat3))
plt.xlabel('lags (months)',)
plt.ylabel(r'$\beta_{1}(h) + \beta_{3}(h): t-stat$',fontsize=15)
plt.ylim(-3,10)
plt.axhline(y=1.96,linewidth=1)
plt.axhline(y=-1.96,linewidth=1)
plt.title('Crisis period', fontsize=20)
plt.savefig('../output/acf2.png')
plt.show()


#estimation of risk premia
start_date = '1926'
end_date = '2018'
pd_data = pd_data[start_date:end_date]
usRec = usRec[start_date:end_date]
rf = rf[start_date:end_date]

#reg plain
indep_ = np.column_stack([pd_data.dp])
exog_ = sm_filters.add_constant(indep_)
reg1 = sm.OLS((pd_data['Ret_1yr'] - rf.rf).values, exog_,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : 1})
reg5 = sm.OLS((pd_data['Ret_5yr'] - rf.rf).values, exog_,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : 1})
print(reg1.summary())
print(reg5.summary())

#reg with rec only
indep1 = np.column_stack([pd_data.dp, pd_data.dp * usRec.USREC.values])
exog1 = sm_filters.add_constant(indep1)
reg_1yr_rec = sm.OLS((pd_data['Ret_1yr']-rf.rf).values,exog1,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : 1})
reg_5yr_rec = sm.OLS((pd_data['Ret_5yr']- rf.rf).values,exog1,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : 1})
print(reg_1yr_rec.summary())
print(reg_5yr_rec.summary())

# reg with rec and fin
indep2 = np.column_stack([pd_data.dp, pd_data.dp * usRec.USREC.values, pd_data.dp * usRec.crisis.values])
exog2 = sm_filters.add_constant(indep2)
reg_1yr = sm.OLS((pd_data['Ret_1yr']-rf.rf).values,exog2,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : 1})
reg_5yr = sm.OLS((pd_data['Ret_5yr']- rf.rf).values,exog2,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : 1})
print(reg_1yr.summary())
print(reg_5yr.summary())

sum_reg1 = summary_col([reg1, reg_1yr_rec, reg_1yr], stars = True,
                float_format = '%0.2f', info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.2f}".format(x.rsquared)})
sum_reg2 = summary_col([reg5, reg_5yr_rec, reg_5yr], stars = True,
                float_format = '%0.2f', info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.2f}".format(x.rsquared)})

sum_reg = summary_col([reg1, reg_1yr_rec, reg_1yr, reg5, reg_5yr_rec, reg_5yr], stars = True,
                float_format = '%0.2f', info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.2f}".format(x.rsquared)})
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"
f = open('../output/ret_pred_1yr.tex', 'w')
f.write(beginningtex)
f.write(sum_reg1.as_latex())
f.write(endtex)
f.close()
f = open('../output/ret_pred_5yr.tex', 'w')
f.write(beginningtex)
f.write(sum_reg2.as_latex())
f.write(endtex)
f.close()
f = open('../output/ret_pred.tex', 'w')
f.write(beginningtex)
f.write(sum_reg.as_latex())
f.write(endtex)
f.close()

usRec['flag'] = np.where(usRec.USREC==1,1,np.nan)
usRec['flag_crisis'] = np.where(usRec.crisis==1,1,np.nan)

print(np.nanstd(reg_1yr.fittedvalues))
print(np.nanstd(reg_1yr.fittedvalues * usRec.flag.iloc[0:reg_1yr.fittedvalues.shape[0]]))
print(np.nanstd(reg_1yr.fittedvalues * np.where((1-usRec.USREC.iloc[0:reg_1yr.fittedvalues.shape[0]])==0.0,0,1)))
print(np.nanstd(reg_1yr.fittedvalues * usRec.flag_crisis.iloc[0:reg_1yr.fittedvalues.shape[0]]))
print(np.nanstd(reg_1yr.fittedvalues * np.where((1-usRec.crisis.iloc[0:reg_1yr.fittedvalues.shape[0]])==0.0,0,1)))
print('#################################')
print(np.nanmean(pd_data.Ret_1yr))
print(np.nanmean(pd_data.Ret_1yr * usRec.flag))
print(np.nanmean(pd_data.Ret_1yr * np.where((1-usRec.USREC)==0.0,0,1)))
print(np.nanmean(pd_data.Ret_1yr * usRec.flag_crisis))
print(np.nanmean(pd_data.Ret_1yr * np.where((1-usRec.crisis)==0.0,0,1)))
print('#################################')
print(np.nanstd(reg1.fittedvalues))
print(np.nanstd(reg1.fittedvalues * usRec.flag.iloc[0:reg1.fittedvalues.shape[0]]))
print(np.nanstd(reg1.fittedvalues * np.where((1-usRec.USREC.iloc[0:reg1.fittedvalues.shape[0]])==0.0,0,1)))
print(np.nanstd(reg1.fittedvalues * usRec.flag_crisis.iloc[0:reg1.fittedvalues.shape[0]]))
print(np.nanstd(reg1.fittedvalues * np.where((1-usRec.crisis.iloc[0:reg1.fittedvalues.shape[0]])==0.0,0,1)))
print('#################################')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

print(np.nanstd(reg_5yr.fittedvalues))
print(np.nanstd(reg_5yr.fittedvalues * usRec.flag.iloc[0:reg_5yr.fittedvalues.shape[0]]))
print(np.nanstd(reg_5yr.fittedvalues * np.where((1-usRec.USREC.iloc[0:reg_5yr.fittedvalues.shape[0]])==0.0,0,1)))
print(np.nanstd(reg_5yr.fittedvalues * usRec.flag_crisis.iloc[0:reg_5yr.fittedvalues.shape[0]]))
print(np.nanstd(reg_5yr.fittedvalues * np.where((1-usRec.crisis.iloc[0:reg_5yr.fittedvalues.shape[0]])==0.0,0,1)))
print('#################################')
print(np.nanmean(reg_5yr.fittedvalues))
print(np.nanmean(reg_5yr.fittedvalues * usRec.flag.iloc[0:reg_5yr.fittedvalues.shape[0]]))
print(np.nanmean(reg_5yr.fittedvalues * np.where((1-usRec.USREC.iloc[0:reg_5yr.fittedvalues.shape[0]])==0.0,0,1)))
print('#################################')
print(np.nanstd(reg5.fittedvalues))
print(np.nanstd(reg5.fittedvalues * usRec.flag.iloc[0:reg5.fittedvalues.shape[0]]))
print(np.nanstd(reg5.fittedvalues * np.where((1-usRec.USREC.iloc[0:reg5.fittedvalues.shape[0]])==0.0,0,1)))
print(np.nanstd(reg5.fittedvalues * usRec.flag_crisis.iloc[0:reg5.fittedvalues.shape[0]]))
print(np.nanstd(reg5.fittedvalues * np.where((1-usRec.crisis.iloc[0:reg5.fittedvalues.shape[0]])==0.0,0,1)))


