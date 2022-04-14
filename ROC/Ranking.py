# Alternative plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statistics import stdev
from math import sqrt
from matplotlib.patches import Wedge
from statsmodels.distributions.empirical_distribution import ECDF
from . import Roc_youden
class Ranking_plots(Roc_youden.Youden_Roc):
    def __init__(self, x2,true_labels1):               
        self.the_labels=true_labels1
        self.the_controls = x2[true_labels1 == 0]
        self.the_cases = x2[true_labels1 == 1]
        self.the_predictions = x2 
    def ranking_plot(self):
        the_zeros=self.the_controls.copy()
        the_ones=self.the_cases.copy()
        n_0 = len(the_zeros)
        n_1 = len(the_ones) 
        tot_val= np.concatenate((np.repeat(0, n_0), np.repeat(1, n_1)))
        ppc_values,auc_value,roc_side=self.calculate_roc(self.the_predictions.copy(), self.the_labels.copy(),autodetect=True)
        print('Estimated ROC side : ',roc_side)
        print('Used to derive further ranking data')
        specificity=np.asarray(1-ppc_values['FPR'])
        sensitivity=np.asarray(ppc_values['TPR']) #aka recall
        FPR=    np.asarray(  ppc_values['FPR'])
        th=ppc_values['c']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(th,FPR,color='indigo',marker='o',label='FPR')
        plt.plot(th,sensitivity,color='gold',marker='o',label='TPR') 
        plt.legend()
        plt.xlabel("Thresholds")
        plt.ylabel("TPR and FPR")
        plt.ylim(0,1)        

        KS=max(abs(sensitivity-FPR))
        print('Kolmogorov - Smirnov statistic :',KS)
        KSt=np.sum(sensitivity[1:-1]-FPR[1:-1]) / (len(sensitivity)-2)
        print('Kolmogorov - Smirnov statistic [truncated] :',KSt)
        text1 = 'KS '+str(round(KS,2))
        text2 = 'Tr. KS '+str(round(KSt,2))
        plt.title('Statistic: '+text1+','+text2)

        #plt.savefig('my_plot200.tif', dpi = 300, bbox_inches='tight')
        plt.show()
    def calculate_AP(self,plot_title=''):
        the_zeros=self.the_controls.copy()
        the_ones=self.the_cases.copy()
        n_0 = len(the_zeros)  # N 
        n_1 = len(the_ones)  # P
        tot_val= np.concatenate((np.repeat(0, n_0), np.repeat(1, n_1)))
        ppc_values,auc_value,roc_side=self.calculate_roc(self.the_predictions.copy(), self.the_labels.copy(),autodetect=True)
        specificity=1-ppc_values['FPR']
        sensitivity=ppc_values['TPR']
        FPR=      ppc_values['FPR']
        PPV=[]
        for i,val_tpr in enumerate(sensitivity):
            tmp=(n_1*val_tpr)/((n_1*val_tpr)+(n_0*FPR[i]))
            PPV.append(tmp)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        AP= np.trapz(PPV,sensitivity)
        plt.plot(sensitivity,PPV,color='forestgreen',marker='.',label = 'AUCPR = %0.2f' % AP) #aka recall        
        plt.xlabel("Recall (TPR)")
        plt.ylabel("Precision")
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.legend()
        if len(plot_title)>0:
            plt.title(plot_title)
        #plt.savefig('my_plot201.tif', dpi = 300, bbox_inches='tight')
        plt.show()
        N=n_0
        
        print('Average Precision :',AP)
