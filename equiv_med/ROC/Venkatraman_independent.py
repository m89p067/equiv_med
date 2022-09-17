import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d 
import seaborn as sns
import pandas as pd
class Venkatraman_indep:      
    def __init__(self, true_labels1,measurement1,true_labels2,measurement2):
        ''' 
        Venkatraman indepedent ROC statistics
        Args:
        true_labels1,true_labels2   two sets of 0s and 1s 
        measurement1,measurement2   mesurements to be tested
        '''
        check1=((true_labels1==0) | (true_labels1==1)).all()
        check2=((true_labels2==0) | (true_labels1==2)).all()
        if check1==False or check2==False:
            print('Please convert both labels in binary values')
        self.labels1 = np.array(true_labels1) # LABELS
        self.meas1 =np.array( measurement1 )
        self.labels2 = np.array(true_labels2) #LABELS
        self.meas2 =np.array( measurement2 )
    def venkatraman_unpaired_stat(self,R, S, D1, D2, mp):
        R_controls = R[D1 == 0]
        R_cases = R[D1 == 1]
        S_controls = S[D2 == 0]
        S_cases = S[D2 == 1]
        n = len(D1)
        m = len(D2)        
        R_fx=np.array([np.count_nonzero(R_cases <= x) for x in range(n)])     /len(R_cases)   
        R_gx=np.array([np.count_nonzero(R_controls <= x) for x in range(n)])   /len(R_controls)     
        S_fx=np.array([np.count_nonzero(S_cases <= x) for x in range(m)]) /len(S_cases)
        S_gx=np.array([np.count_nonzero(S_controls <= x) for x in range(m)]) /len(S_controls)
        R_p = mp * R_fx + (1 - mp) * R_gx
        S_p = mp * S_fx + (1 - mp) * S_gx
        R_exp = mp * R_fx + (1 - mp) * (1 - R_gx)
        S_exp = mp * S_fx + (1 - mp) * (1 - S_gx)
        x = np.sort(np.append(R_p,S_p) )
        R_f = interp1d(R_p, R_exp, fill_value="extrapolate")
        S_f = interp1d(S_p, S_exp, fill_value="extrapolate")
        f = lambda x_val: abs(R_f(x_val) - S_f(x_val))
        y = f(x)
        #plt.plot(x,y)
        integral=[]
        for idx in range(1,len(x)):
            integral.append( np.nansum(((y[idx] + y[idx - 1]) * (x[idx] - x[idx -   1]))/2))
        return sum(integral)
    def venkatraman_unpaired_permutation(self, R, S, D1, D2, mp):   
        R=R+np.random.uniform(low=0, high=1, size=len(D1))- 0.5
        S=S+np.random.uniform(low=0, high=1, size=len(D1))- 0.5
        R_controls = R[D1 == 0]
        R_cases = R[D1 == 1]
        S_controls = S[D2 == 0]
        S_cases = S[D2 == 1]
        temp=np.append(R_controls, S_controls)
        controls = np.random.choice(temp, size=len(temp), 
                                    replace=False, p=None)
        temp=np.append(R_cases, S_cases)
        cases = np.random.choice(temp, size=len(temp), 
                                 replace=False, p=None)
        R[D1 == 0] = controls[0:len(R_controls)]
        S[D2 == 0] = controls[len(R_controls) :len(controls)]
        R[D1 == 1] = cases[0:len(R_cases)]
        S[D2 == 1]= cases[len(R_cases) :len(cases)]
        out=self.venkatraman_unpaired_stat(stats.rankdata(R, 'ordinal'),
                                           stats.rankdata(S, 'ordinal'), D1, D2, mp)
        return out
    def Venkatraman_unpaired_test(self,n_boot):
        X=self.meas1
        Y=self.meas2
        D1=self.labels1 
        D2=self.labels2 
        S=stats.rankdata(Y,'ordinal')
        R=stats.rankdata(X,'ordinal')
        mp = (np.count_nonzero(D1 == 1) + np.count_nonzero(D2 == 1))/(len(D1) + len(D2))# D1?
        E=self.venkatraman_unpaired_stat(R, S, D1, D2, mp)
        out=[]
        for i in range(n_boot):
            out.append(self.venkatraman_unpaired_permutation(R,S,D1,D2,mp)    )   
        EP=np.array(out)
        #alternative hypothesis: true difference in ROC operating points is not equal to 0
        return [E,EP]
    def v_indep(self,num_boots=2000):
        stat_values=self.Venkatraman_unpaired_test(n_boot=num_boots)
        pval = sum(stat_values[1] >= stat_values[0])/num_boots        
        print('Venkatraman method for testing two unpaired ROC curves\n')
        print('Statistics : ',stat_values[0], 'with ',num_boots,' permutations')
        print('This method tests the equality of the two ROC curves at all operating points')
        print('Statistics, p value :',pval,' [two sided]')
        if pval>0.05:
            print('Do not reject the hypothesis that the ROC curves are equal.') 
            print('Equality of the two ROC curves at all operating points')
            #print('and consequently sensitivities and specificities are equal')
        else:
            print('Reject the hypothesis that the ROC curves are equal.')
            print('Difference in at least one ROC operating point')
        self.plot_distrib()    
    def plot_distrib(self):
        X=self.meas1
        Y=self.meas2
        D1=self.labels1 
        D2=self.labels2 
        df1=pd.DataFrame({'Data':X,'Labels':D1})
        df2=pd.DataFrame({'Data':Y,'Labels':D2})
        group1=['Group 1'] * df1.shape[0]
        group2=['Group 2'] * df2.shape[0]
        df1['Group']=group1
        df2['Group']=group2
        df3=pd.concat([df1, df2], ignore_index=True)
        sns.displot(data=df3, x="Data", hue="Labels", col="Group", kind="kde")
