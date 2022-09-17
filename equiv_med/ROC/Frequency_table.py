import numpy as np
import scipy.stats as stats
import pandas as pd
from math import sqrt,pi
import matplotlib.pyplot as plt
import seaborn as sn
import pdb
class Freq_table:      
    def __init__(self,Diagnostic_result,Disease_status): # dichotomous values
        ''' 
        Creates a 2x2 frequency table (aka confusion matrix) and extracts performance indexes
        Args:
        Diagnostic_result   binary vector with results of the laboratory test
        Disease_status  binary vector with sick/healthy status of patients
        '''
        check1=((Diagnostic_result==0) | (Diagnostic_result==1)).all()
        check2=((Disease_status==0) | (Disease_status==1)).all()
        if check1==False or check2==False:
            print('Please convert both vectors in binary values')
        else:
            self.DR1=Diagnostic_result
            self.DS1=Disease_status  
    def performance_indexes(self,do_verbose=True):
        DR=np.asarray(self.DR1)
        DS=np.asarray(self.DS1     )  
        check1=((DR==0) | (DR==1)).all()
        check2=((DS==0) | (DS==1)).all()
        if check1 and check2:
            print('Assuming 0s mean negative results and 1s mean postive results')
            N=len(DR)
            freq_tab=pd.crosstab(DS, DR) #index, columns
            TN=freq_tab.iloc[0,0]
            TP=freq_tab.iloc[1,1]
            FN=freq_tab.iloc[0,1] # Type II error
            FP=freq_tab.iloc[1,0] # Type I error
            if do_verbose:
                print('Subjects with the disease TP+FN ',TP+FN) # Positives
                print('Subjects without the disease TN+FP ',TN+FP) # Negatives
            TPF=TP/(TP+FN) # True positive fraction aka Sensitivity
            FNF=FN/(TP+FN) # False Negative Fraction aka 1-Sensitivity
            TNF=TN/(TN+FP) # True Negative Fraction aka Specificity
            FPF=FP/(TN+FP) # False Positive Fraction aka 1-Specificity
            PPV=TP/(TP+FP) # Positive predicted value
            NPV=TN/(TN+FN) # Negative predicted value
            Likelihood_ratio_plus=TPF/FPF # Positive likelihood ratio index
            Likelihood_ratio_negative=FNF/TNF # Negative likelihood ratio index
            DOR=(TP/FN)/(FP/TN) #Diagnostic odds ratio
            ACC= (TP+TN) / (TP+TN+FP+FN) #Diagnostic effectiveness aka diagnostic accuracy
            Error_rate=(FP+FN)/(TP+FP+FN+TN) # error rate
            YOUD= (TPF + TNF) - 1 # Youden's index            
            BACC=(TPF+TNF)/2 # Balanced Accuracy
            Precision=TP/(TP+FP) # Precision
            Recall=TP/(TP+FN) # Recall
            F1= 2*(1/(  (1/Precision)+ (1/Recall) ) ) # F-measure (also known as F1-score or simply F-score)
            #F1 = 2*TP/(2*TP+FP+FN);         # F1 score
            G=sqrt(Precision*Recall)
            MCC=((TP*TN)-(FP*FN))/sqrt(  (TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)  )
            Cohen_K=(2*((TP*TN) -(FN*FP)  ) )/(  ( (TP+FP)*(FP+TN))+((TP+FN)*(FN+TN) ))
            FDR=FP/(FP+TP)
            BM=TPF+TNF-1 # Bookmaker informedness
            MK=PPV+NPV-1 # Markedness
            FNR   = FN/(FN+TP)
            FPR   = FP/(FP+TN)
            FOR   = FN/(FN+TN)
            NNT=(TP+TN+FP+FN)/TN
            NND=1/YOUD
            CSI=TP/(TP+FN+FP)
            PT=(sqrt(TPF*FPF)-FPF)/(TPF-FPF)
            TNR=1-FPF
            FMI=sqrt(PPV*TPF)
            PR=(TP+FN)/N
            B=(TP+FP)/N
            if do_verbose:
                print('True positive fraction aka Sensitivity ',TPF)
                print('False Negative Fraction aka 1-Sensitivity ',FNF)
                print('True Negative Fraction aka Specificity ',TNF)
                print('False Positive Fraction aka 1-Specificity ',FPF)
                print('Positive Predictive Value ',PPV)
                print('Negative Predictive Value ',NPV)
                print('Positive Likelihood Ratio ',Likelihood_ratio_plus)
                print('Negative Likelihood Ratio ', Likelihood_ratio_negative)                
                print('Diagnostic Odds Ratio ',DOR)
                print('Diagnostic effectiveness (accuracy) ',ACC)
                print('Youden index ',YOUD )
                print('Cohen kappa coefficient ',Cohen_K)
                print('Balanced accuracy ',BACC)
                print('Error rate ',Error_rate)
                print('False negative rate ',FNR)
                print('True negative rate ',TNR)
                print('False positive rate ',FPR)
                print('False discovery rate ',FDR)
                print('False omission rate ',FOR)
                print('Precision ',Precision)
                print('Recall ',Recall)
                print('F-measure (aka F1-score or simply F-score) ',F1)
                print('G measure ',G)
                print('Matthews correlation coefficient ',MCC) 
                print('Bookmaker informedness ',BM)
                print('Markedness ',MK)
                print('Number necessary to test ',NNT)
                print('Number necessary to diagnose ',NND)
                print('Critical success index ',CSI)
                print('Prevalence ',PR)
                print('Prevalence threshold ',PT)
                print('Fowlkes–Mallows index ',FMI)
                print('Bias ',B)
            outcomes={'True positive fraction':TPF,
                'False Negative Fraction':FNF,
                'True Negative Fraction':TNF,
                'False Positive Fraction':FPF,
                'Positive Predictive Value':PPV,
                'Negative Predictive Value':NPV,
                'Positive Likelihood Ratio':Likelihood_ratio_plus,
                'Negative Likelihood Ratio': Likelihood_ratio_negative,
                'Diagnostic Odds Ratio':DOR,
                'Diagnostic effectiveness (accuracy)':ACC,
                'Youden index':YOUD ,
                'Cohen kappa coefficient':Cohen_K,
                'Balanced accuracy':BACC,
                'Error rate':Error_rate,
                'False discovery rate':FDR,
                'Precision':Precision,
                'Recall':Recall,
                'F-measure (aka F1-score or simply F-score)':F1,
                'G measure':G,'Bias':B,
                'Matthews correlation coefficient':MCC,
                'Bookmaker informedness':BM,
                'Markedness':MK,'False negative rate':FNR,'False positive rate':FPR,
                'False omission rate':FOR,'Sensitivity':TPF,'Specificity':TNF,
                'Number necessary to test':NNT,'Number necessary to diagnose':NND,
                'Critical success index':CSI,'Prevalence threshold':PT,'True negative rate':TNR,
                'Fowlkes–Mallows index':FMI,'Prevalence':PR}
            return outcomes
        else:
            print('Labels are not 0s and 1s')
    def frequency_plot(self,percentage=False):
        DR=np.asarray(self.DR1)
        DS=np.asarray(self.DS1     )  
        check1=((DR==0) | (DR==1)).all()
        check2=((DS==0) | (DS==1)).all()
        if check1 and check2:
            print('Assuming 0s mean negative results and 1s mean postive results')
            if percentage==False:
                freq_tab=pd.crosstab(DS, DR,rownames=['Diagnostic result'],colnames=['Disease status']) #index, columns        
                sn.heatmap(freq_tab, annot=True, fmt='.2f', cmap="YlGnBu", cbar=False,xticklabels=['Absent','Present'], yticklabels=['Negative','Positive'])                        
            else:
                freq_tab2=pd.crosstab(DS, DR,rownames=['Diagnostic result [%]'],colnames=['Disease status [%]'],normalize ='all').round(2)*100 
                sn.heatmap(freq_tab2, annot=True, fmt='.2f', cmap="YlGnBu", cbar=False,xticklabels=['Absent','Present'], yticklabels=['Negative','Positive'])                                    
            
            plt.show()

        
