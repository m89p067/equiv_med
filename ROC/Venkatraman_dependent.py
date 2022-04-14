import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
class Venkatraman_dep:      
    def __init__(self, true_labels,measurement1,measurement2):
        # vectors of 0s and 1s
        check1=((true_labels==0) | (true_labels==1)).all()
        if check1==False:
            print('Please convert the labels in binary values')
        self.labels = np.array(true_labels) # LABELS
        self.meas1 =np.array( measurement1 )        
        self.meas2 =np.array( measurement2 )    
    def Venkatraman_paired_stat(self,R, S, D):
        R_controls = R[D == 0]
        R_cases = R[D == 1]
        S_controls = S[D == 0]
        S_cases = S[D == 1]
        n = len(D)
        R_fn = np.array([np.count_nonzero(R_cases <= x) for x in range(n)])
        R_fp = np.array([np.count_nonzero(R_controls > x) for x in range(n)])
        S_fn = np.array([np.count_nonzero(S_cases <= x) for x in range(n)])
        S_fp = np.array([np.count_nonzero(S_controls > x) for x in range(n)])
        out=np.subtract(np.add(S_fn , S_fp) , np.add(R_fn , R_fp) )
        return sum(abs(out ) )
    def Venkatraman_paired_permutation(self,R, S, D):
        R2 = R + np.random.uniform(low=0, high=1, size=len(D)) - 0.5
        S2 = S + np.random.uniform(low=0, high=1, size=len(D)) - 0.5
        q = 1 - np.round_(np.random.uniform(low=0, high=1, size=len(D)))
        R3 = R2 * q + (1 - q) * S
        S3 = S2 * q + (1 - q) * R        
        out1=stats.rankdata(R3, 'ordinal')
        out2=stats.rankdata(S3, 'ordinal')
        out3=self.Venkatraman_paired_stat(out1,out2 , D)
        return out3
    def Venkatraman_paired_test(self,n_boot):
        X=self.meas1
        Y=self.meas2
        D=self.labels         
        S=stats.rankdata(Y,'ordinal')
        R=stats.rankdata(X,'ordinal') 
        E = self.Venkatraman_paired_stat(R, S, D)
        out=[]
        for i in range(n_boot):
            out.append(self.Venkatraman_paired_permutation(R,S,D)    )   
        EP=np.array(out)
        return [E,EP]
    def v_dep(self,num_boots=2000):
        stat_values=self.Venkatraman_paired_test(n_boot=num_boots)
        pval = sum(stat_values[1] >= stat_values[0])/num_boots        
        print('Venkatraman method for testing two paired ROC curves')
        print('This method tests the equality of the two ROC curves at all operating points')
        print('Statistics : ',stat_values[0], 'with ',num_boots,' permutations')        
        print('Statistics, p value :',pval,' [two sided]')
        if pval>0.05:
            print('Do not reject the hypothesis that the ROC curves are equal.') 
            print('Equality of the two ROC curves at all operating points')
            #print('and consequently sensitivities and specificities are equal')
        else:
            print('Reject the hypothesis that the ROC curves are equal.')
            print('Difference in at least one ROC operating point')        
    def do_plot(self,bootstrap_replicates=500,do_auc_p_val=True):
        val1,auc1=self.calculate_roc(self.meas1, self.labels)
        val2,auc2=self.calculate_roc(self.meas2, self.labels)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        
        ax1.plot(val1['FPR'],val1['TPR'], lw=2, color="forestgreen",label = 'AUC = %0.2f' % auc1)
        ax1.set_ylabel('True Positive Rate')
        ax1.set_xlabel('False Positive Rate')
        ax1.plot([0, 1], [0, 1],'k--')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.legend(loc = 'lower right')
        
        ax2.plot(val2['FPR'],val2['TPR'], lw=2, color="crimson",label = 'AUC = %0.2f' % auc2)
        ax2.set_ylabel('True Positive Rate')
        ax2.set_xlabel('False Positive Rate')
        ax2.plot([0, 1], [0, 1],'k--')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.legend(loc = 'lower right')        
        
        plt.show()
        
        if do_auc_p_val:            
            #p values of AUC (first measurement)        
            the_controls = self.meas1[self.labels == 0]
            the_cases = self.meas1[self.labels == 1]
            n0 = len(the_controls)
            n1 = len(the_cases)
            valori = np.concatenate((np.repeat(0, n0), np.repeat(1, n1)))
            p_auc1=[]
            for i in range(bootstrap_replicates):
                pD = np.random.choice(valori, size= n0 + n1, replace=False)
                _,auc_res=self.calculate_roc(self.meas1, pD)
                p_auc1.append(auc_res)        
            pval_auc1 = np.mean(np.array(p_auc1 )> auc1)
            print('Measurement 1 AUC [',round(auc1,3),'] pvalue:',pval_auc1,' based on ',bootstrap_replicates,' replicates')
            #p values of AUC (second measurement)  
            the_controls = self.meas2[self.labels == 0]
            the_cases = self.meas2[self.labels == 1]
            n0 = len(the_controls)
            n1 = len(the_cases)
            valori = np.concatenate((np.repeat(0, n0), np.repeat(1, n1)))
            p_auc2=[]
            for i in range(bootstrap_replicates):
                pD = np.random.choice(valori, size= n0 + n1, replace=False)
                _,auc_res=self.calculate_roc(self.meas2, pD)
                p_auc2.append(auc_res)       
            pval_auc2 = np.mean(np.array(p_auc2 )> auc2)
            print('Measurement 2 AUC [',round(auc2,3),'] pvalue:',pval_auc2,' based on ',bootstrap_replicates,' replicates')
        
    def calculate_roc(self,myX,myD):          
        the_controls = myX[myD == 0]
        the_cases = myX[myD == 1]
        n0 = len(the_controls)
        n1 = len(the_cases)
        m=n0
        t = np.arange(0, 1+(1/m), 1/m) 
        N = len(t)
        tmp=np.concatenate((the_controls, the_cases))
        XX = np.sort( tmp)
        XX_un=np.unique(XX)
        e = np.where(len(XX_un) > 1, np.amin(XX_un[1:] - np.delete(XX_un,len(XX_un)-1))/2, np.sqrt(np.finfo(float).eps))

        A=[] #assuming General/right-sided ROC curve
        for i in range(1,N+1):
            if i==N:
                roc =1
                xu = np.amax(the_controls)
                xl=np.amax(the_controls)
            else:
                gamma = np.arange(0,i,1)
                s_c=np.sort(the_controls)
                ecdf = ECDF(the_cases)
                index_gamma_t = np.argmax(ecdf(s_c[gamma] -e) + 1 - ecdf(s_c[m - i + gamma]))
                gamma_t = gamma[index_gamma_t]
                xl = s_c[gamma_t]
                xu = s_c[m - i + gamma_t]
                roc = ecdf(xl - e) + 1 - ecdf(xu)
            A.append([roc,xl,xu])
        results=np.array(A)
        
            
        xl = results[:,1]
        xu = results[:,2]
        roc=results[:,0]
        pairpoints_coordinates = pd.DataFrame({"xl":xl,  "xu":xu,"FPR": t,"TPR": roc})
        auc= np.sum(np.multiply(np.delete(roc,N-1) , (t[1:] - np.delete(t,N-1)))  )  
        return pairpoints_coordinates,auc