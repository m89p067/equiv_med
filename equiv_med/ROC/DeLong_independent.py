import numpy as np
from math import sqrt
from scipy.stats import t
class DeLong_indep:      
    def __init__(self, true_labels1,measurement1,true_labels2,measurement2):
        ''' 
        DeLong stats on independent measurements
        Args:
        true_labels1,true_labels2   vectors of 0s and 1s as true labels for the first and second set of binary outcomes
        measurement1,measurement2   measurements
        '''
        check1=((true_labels1==0) | (true_labels1==1)).all()
        check2=((true_labels2==0) | (true_labels1==1)).all()
        if check1==False or check2==False:
            print('Please convert both labels in binary values')
        self.labels1 = np.array(true_labels1) # LABELS
        self.meas1 =np.array( measurement1 )
        self.labels2 = np.array(true_labels2) #LABELS
        self.meas2 =np.array( measurement2 )
    def delongPlacements(self,the_controls,the_cases,direction=True):
        m = len(the_cases)
        n = len(the_controls)
        L = m + n    
        if direction: # > need to reverse the values
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

    def delong_unpaired_test(self,R,S,D1,D2):
        R_controls = R[D1 == 0]
        R_cases = R[D1 == 1]
        S_controls = S[D2 == 0]
        S_cases = S[D2 == 1]    
        nR = len(R_controls)
        mR = len(R_cases)
        nS = len(S_controls)
        mS = len(S_cases)
        VR = self.delongPlacements(R_controls,R_cases)
        VS = self.delongPlacements(S_controls,S_cases)
        SRX = np.sum(np.multiply(VR['X'] - VR['theta'] , VR['X'] - VR['theta']))/(mR - 1)
        SSX = np.sum(np.multiply(VS['X'] - VS['theta'] , VS['X'] - VS['theta']))/(mS - 1)
        SRY = np.sum(np.multiply(VR['Y'] - VR['theta'] , VR['Y'] - VR['theta']))/(nR - 1)
        SSY = np.sum(np.multiply(VS['Y'] - VS['theta'] , VS['Y'] - VS['theta']))/(nS - 1)
        SR = SRX/mR + SRY/nR
        SS = SSX/mS + SSY/nS
        ntotR = nR + mR
        ntotS = nS + mS
        SSR = sqrt((SR) + (SS))
        t = (VR['theta'] - VS['theta'])/SSR
        df = ((SR + SS)**2)/(((SR**2)/(ntotR - 1)) + ((SS**2)/(ntotS -  1)))
        return [t, df]
    def dl_indep(self):
        Labels1=self.labels1 
        Meas1=self.meas1 
        Labels2=self.labels2 
        Meas2=self.meas2 
        stats= self.delong_unpaired_test(Meas1,Meas2,Labels1,Labels2) 
        print("Two sided DeLong's test for two unpaired ROC curves")
        quantil=stats[0]
        dof=stats[1]
        pval = 2 * t.cdf(-abs(quantil),  int(dof) )
        print('Statistics [',quantil,'], p value :',pval,' [two sided]')
        if pval>0.05:
            print('Do not reject the hypothesis that the ROC curves are equal.') 
        else:
            print('Reject the hypothesis that the ROC curves are equal.')
            
        

