import numpy as np
from scipy import stats
from math import sqrt
class DeLong_dep:      
    def __init__(self, true_labels1,measurement1,measurement2):
        ''' 
        DeLong stats for dependent measurements
        true_labels1    a vector of 0s and 1s as true labels 
        measurement1,measurement2   two measurement vectors
        '''
        check1=((true_labels1==0) | (true_labels1==1)).all()        
        if check1==False :
            print('Please check both labels contain binary values')
        self.labels1 = np.array(true_labels1) # LABELS
        self.meas1 =np.array( measurement1 )        
        self.meas2 =np.array( measurement2 )
    def delongPlacements(self,the_controls,the_cases,change_direction=True):
        m = len(the_cases)
        n = len(the_controls)
        L = m + n    
        if change_direction: # > need to reverse the values
            for i in range(m):
                the_cases[i] = -the_cases[i]
            for i in  range(n):
                the_controls[i] = -the_controls[i]
        Z=np.empty([L,2])
        labels=[]
        for i in range(m):
            Z[i,0]=i
            Z[i,1]=the_cases[i]
            labels.append(True)
        for j in range(n):
            Z[m+j,0]=m+j
            Z[m+j,1]=the_controls[j]
            labels.append(False)
        Z_sorted_asc = Z[Z[:, 1].argsort()]       
        XY=np.repeat(0.0,L)
        m = 0
        n = 0
        i = 0
        while i<L:
            X_inds =[]
            Y_inds=[]
            mdupl = 0
            ndupl = 0
            #if i % 10000 == 0:
            #    print('Division')
            while True:
                j=int(Z_sorted_asc[i,0])
                if labels[j]:
                    mdupl=mdupl+1
                    X_inds.append(j)
                else:
                    ndupl=ndupl+1
                    Y_inds.append(j)
                if i == (L-1):
                    break
                if Z_sorted_asc[i,1] != Z_sorted_asc[i+1,1]:
                    break
                i=i+1
            for k in range(mdupl):
                XY[X_inds[k] ]= n + ndupl/2.0
            for k in range(ndupl):
                XY[Y_inds[k ] ]= m + mdupl/2.0
            n += ndupl
            m += mdupl  
            i=i+1
        the_sum=0.0
        X=[]
        Y=[]
        for i  in range(L):
            if labels[i]:
                the_sum += XY[i]
                X.append(XY[i]/n)
            else:
                Y.append(1.0 - XY[i] / m)
        theta=the_sum / m / n
        ret={'theta':theta,'X':X,'Y':Y}
        return ret 
    def delong_paired_calculations(self,R,S,D) :    
        R_controls = R[D == 0]
        R_cases = R[D == 1]
        S_controls = S[D == 0]
        S_cases = S[D == 1] 
        n = len(R_controls)
        m = len(R_cases)
        # DeLong's test should not be applied to ROC curves with a different direction.
        VR = self.delongPlacements(R_controls.copy(),R_cases.copy())
        VS = self.delongPlacements(S_controls.copy(),S_cases.copy(),change_direction=False)
        SX = np.full((2,2), np.nan)
        SX[0,0] = np.sum(np.multiply(VR['X'] - VR['theta'] , VR['X'] - VR['theta']))/(m -             1)
        SX[0, 1]= np.sum(np.multiply(VR['X'] - VR['theta'] , VS['X'] - VS['theta']))/(m -             1)
        SX[1,0] = np.sum(np.multiply(VS['X'] - VS['theta'] , VR['X'] - VR['theta']))/(m -             1)
        SX[1, 1] = np.sum(np.multiply(VS['X'] - VS['theta'] , VS['X'] - VS['theta']))/(m -             1)
        SY =np.full((2,2), np.nan)
        SY[0, 0] = np.sum(np.multiply(VR['Y'] - VR['theta'] , VR['Y'] - VR['theta']))/(n -             1)
        SY[0, 1] = np.sum(np.multiply(VR['Y'] - VR['theta'] , VS['Y'] - VS['theta']))/(n -             1)
        SY[1, 0] = np.sum(np.multiply(VS['Y'] - VS['theta'] , VR['Y'] - VR['theta']))/(n -             1)
        SY[1, 1] = np.sum(np.multiply(VS['Y'] - VS['theta'] , VS['Y'] - VS['theta']))/(n -             1)
        Sout = SX/m + SY/n
        L =np.asarray( [1, -1])
        part1=np.dot(L,Sout.T)
        part2=np.dot(part1,L.T)
        sig = sqrt(part2)
        d = VR['theta'] - VS['theta']
        out={'d' : d, 'sig' : sig }
        return out
    def delong_paired_test(self,calcs):
        zscore=calcs['d']/calcs['sig']
        if np.isnan(zscore) and calcs['d'] == 0 and calcs['sig'] == 0:
            zscore=0
        return zscore
    def ci_delong_paired(self,calcul,ci_limits=0.95):
        crit_z = stats.norm.ppf(1 - ((1 - ci_limits)/2))
        out = dict()
        out['upper'] = calcul['d'] + crit_z * calcul['sig']
        out['lower'] = calcul['d'] - crit_z * calcul['sig']
        out['level'] =ci_limits
        return out
    def dl_dep(self):
        D2=self.labels1
        R2=self.meas1        
        S2=self.meas2
        calcs2=self.delong_paired_calculations(R2,S2,D2)
        print("DeLong's test for two correlated ROC curves")
        stat=self.delong_paired_test(calcs2)
        conf_int=self.ci_delong_paired(calcs2)
        pval = 2 * stats.norm.cdf(-abs(stat))
        print('Statistics [',stat,'], p value :',pval,' [two sided]')
        print('Confidence intervals [',conf_int['lower'],':',conf_int['upper'],'] at ',conf_int['level']*100,' %')
        if pval>0.05:
            print('Do not reject the hypothesis that the ROC curves are equal.') 
        else:
            print('Reject the hypothesis that the ROC curves are equal.')
