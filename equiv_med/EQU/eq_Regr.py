import numpy as np
from . import eq_base
import statsmodels.stats.stattools as sms
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from seaborn import kdeplot
from math import sqrt
class Regr_diagn(eq_base.init_eq):
    is_normal=True
    def __init__(self, x, y):
        ''' 
        Residuals diagnostics and influential points evaluation
        Args:
        x,y two laboratory measurements
        '''
        super().__init__(x, y)
        self.results=self.run_regression()
    def get_cook(self,D):     
        influence = self.results.get_influence()
        res1=influence.resid_studentized_internal
        limiti=[abs(np.min(res1)),abs(np.max(res1))]
        scale_fact=3
        valori_y=np.linspace(-max(limiti)*scale_fact,max(limiti)*scale_fact,num=len(res1)*scale_fact,endpoint=True)
        part3=[]
        for valori in valori_y:
            part4=(2*D)/((valori**2)+(2*D))
            part3.append(part4)
        return valori_y,part3
    def linea(self, x_1,y_1,s1,k1,type_spline=1):         
        xmin, xmax = np.min(x_1), np.max(x_1)
        ymin, ymax = np.min(y_1), np.max(y_1)
        xx = np.linspace(xmin, xmax, num=len(x_1),endpoint=True)
        spline = interpolate.UnivariateSpline(xx, y_1,s=s1,k=k1)                
        return xx,spline
    def run_diagnostic(self,array_of_cook):
        residuals = self.results.resid
        fitted_value = self.results.fittedvalues
        stand_resids = self.results.resid_pearson
        influence = self.results.get_influence()
        print('\nRegression diagnostic with Jarque Bera on residuals:')
        test = sms.jarque_bera(residuals)
        if test[1]>0.05:
            print('Keep the null hypothesis: sample data have the skewness and kurtosis matching a normal distribution')
            Regr_diagn.is_normal=True
        else:
            print('Reject the null hypothesis that the data is normally distributed')
            print('We have sufficient evidence to say that the data in this example is not normally distributed')
            Regr_diagn.is_normal=False
        print( "Durbin-Watson test statistics is " + str(sms.durbin_watson(residuals)))
        leverage =influence.hat_matrix_diag
        cooks =influence.cooks_distance        
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx],idx
        # Residuals vs fitted  
        fig = plt.figure()
        plt.plot( fitted_value,residuals,'k.')
        plt.axhline(y=0, color='grey', linestyle='--')
        #sns.kdeplot(fitted_value,residuals, shade = True)
        kdeplot(x=fitted_value,y=residuals, cmap = "winter_r")
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        
        plt.show()
        
        out1,out2=self.linea( fitted_value,stand_resids,None,3,type_spline=1)
        plt.plot(fitted_value,stand_resids,'go')
        plt.plot(out1,out2(out1),'r-',label='Fitting')
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.ylabel('Standardized residuals')
        plt.xlabel('Fitted values')
        plt.title('Spread-Location') 
        plt.show()
        limite1=[]
        limite2=[]
        plt.plot(leverage,stand_resids,'m.')
        plt.axhline(y=0, color='grey', linestyle='--')
        for i in range(len(array_of_cook)):
            x_a,a=self.get_cook(array_of_cook[i])
            plt.plot(a,x_a,'--',color='orange')
            limite1.append(a)
            limite2.append(x_a)
        plt.ylabel('Standardized Residuals')
        plt.xlabel('Leverage')
        scaler_x,scaler_y=0.8,0.8
        plt.xlim(min(leverage),max(leverage)+(scaler_x*max(leverage)))
        limiti=[abs(np.min(stand_resids)),abs(np.max(stand_resids))]
        plt.ylim(-max(limiti)-(scaler_y*max(limiti)),max(limiti)+(scaler_y*max(limiti)))
        left, right = plt.xlim()
        for i in range(len(array_of_cook)):
            out1,out2=find_nearest(limite1[i], right)
            out3=limite2[i]
            plt.text(right+(right*0.01), out3[out2], 'd:'+str(array_of_cook[i]))
        
        plt.show()
    def influential_points(self):

        influence = self.results.get_influence()
        dffits, p1 = influence.dffits #general influence of an observation
        dfbeta=influence.dfbetas# specific impact of an observation on the regression coefficients. 
        dffits_th=2 * sqrt(2.0 / len(dffits))
        dffits_th2=3 * sqrt(2.0 / len(dffits)) #more conservative
        dfbetas_th=2/sqrt(len(dfbeta))
        fig1 = plt.figure()
        ax1 = plt.subplot(111)
        plt.plot(np.arange(0,len(dffits)),dffits,color='k',marker='o',linestyle='None')
        plt.xticks(color='w')
        plt.ylabel('DFFITS')
        plt.xlabel('Observations')     
        handles, _ = ax1.get_legend_handles_labels()
        handles.append(plt.axhline(y = dffits_th, color = 'lightsalmon', linestyle =":",label='Empir. Th.') )
        handles.append(plt.axhline(y = dffits_th2, color = 'red', linestyle =":",label='Conserv. Th.')    )
        ax1.legend(handles = handles,loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        
        plt.show()
        
        fig2 = plt.figure()
        ax2 = plt.subplot(111)
        plt.plot(np.arange(0,len(dfbeta[:,0])),dfbeta[:,0],color='forestgreen',marker='o',linestyle='None',label='Param. 1')
        plt.plot(np.arange(0,len(dfbeta[:,1])),dfbeta[:,1],color='royalblue',marker='o',linestyle='None',label='Param. 2')
        plt.xticks(color='w')
        plt.ylabel('DFBETAS')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        plt.axhline(y = dfbetas_th, color = 'red', linestyle =":")         
        
        plt.show()        
