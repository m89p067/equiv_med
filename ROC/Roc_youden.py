# Youden index and ROC plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statistics import stdev
from math import sqrt
from matplotlib.patches import Wedge
from statsmodels.distributions.empirical_distribution import ECDF
class Youden_Roc:
    def __init__(self, x1,true_labels):               
        self.label=true_labels
        self.the_controls = x1[true_labels == 0]
        self.the_cases = x1[true_labels == 1]
        self.preds1 = x1 
    def density_estimation(self):
        gkde_controls = stats.gaussian_kde(self.the_controls.copy())
        gkde_cases = stats.gaussian_kde(self.the_cases.copy())
        min_controls = min(self.the_controls.copy())
        max_controls = max(self.the_controls.copy())
        min_cases = min(self.the_cases.copy())
        max_cases = max(self.the_cases.copy())
        x_controls=np.linspace(min_controls,max_controls,1000,endpoint=True)
        x_cases=np.linspace(min_cases,max_cases,1000,endpoint=True)
        estimated_controls = gkde_controls.evaluate(x_controls)
        estimated_cases = gkde_cases.evaluate(x_cases)
        plt.figure()        
        plt.plot(x_controls, estimated_controls, label="Controls [0s]", color="mediumslateblue",lw=2)
        plt.plot(x_cases, estimated_cases, label="Cases [1s]", color="seagreen",lw=2)
        plt.xlabel("Observations")
        plt.ylabel("Prob. density")
        plt.legend()
        plt.grid(True)
        plt.show()
    def check_side(self):
        n0 = len(self.the_controls.copy())
        n1 = len(self.the_cases.copy())
        wilcox = stats.mannwhitneyu(self.the_controls.copy(), self.the_cases.copy(), 
                                    use_continuity=True, alternative='less')
        if wilcox.statistic<= n0 * n1/2:
            the_side = "right"            
        else:
            the_side="left"        
        print('The ROC curve will be ',the_side,' sided')
        return the_side,wilcox.pvalue
    def calculate_roc(self,myX,myD,autodetect=False ):
        ind0=np.ravel(myD == 0)
        ind1=np.ravel(myD == 1)
        controls1 = myX[ind0]
        cases1 = myX[ind1]
        n0 = len(controls1)
        n1 = len(cases1)        
        m=n0
        t= np.arange(0, 1+(1/m), 1/m) 
        N = len(t)
        XX = np.sort( np.concatenate([controls1,cases1]))
        XX_un=np.unique(XX)
        if autodetect:
            side,p_value=self.check_side()
        else:
            side='general'
        the_method='closest_observation'
        e = np.where(len(XX_un) > 1, np.amin(XX_un[1:] - np.delete(XX_un,len(XX_un)-1))/2, np.sqrt(np.finfo(float).eps))
        if side == "right":
            c = np.quantile(controls1, 1 - t, method = the_method)
            tmp=ECDF(cases1)
            roc = 1 - tmp(c)            
           
        elif side == "left":
            tmp1=np.quantile(controls1, t, method = the_method)
            tmp2=ECDF(controls1)
            c = np.where(tmp1 - e * tmp2(tmp1) > t,1,0)
            tmp3=ECDF(cases1)
            roc =tmp3(c)            
            
        elif side == "general":
            A=[]
            for i in range(1,N+1):
                if i==N:
                    roc =1
                    xu = np.amax(controls1)
                    xl=np.amax(controls1)
                else:
                    gamma = np.arange(0,i,1)
                    s_c=np.sort(controls1)
                    ecdf = ECDF(cases1)
                    #index_gamma_t = np.argmax(ecdf(s_c[gamma] -e) + 1 - ecdf(s_c[m - i + gamma]))
                    index_gamma_t = np.argmax(ecdf(s_c[gamma] -e) + 1 - ecdf(s_c[m -i + gamma]))
                    gamma_t = gamma[index_gamma_t]
                    xl = s_c[gamma_t]
                    #xu = s_c[m - i + gamma_t]
                    xu = s_c[m - i + gamma_t]
                    roc = ecdf(xl - e) + 1 - ecdf(xu)
                A.append([roc,xl,xu])
            results=np.array(A)
        if side=="general":    
            xl = results[:,1]
            xu = results[:,2]
            roc=results[:,0]
            pairpoints_coordinates = pd.DataFrame({"xl":xl,  "xu":xu,"FPR": t,"TPR": roc})
            auc= np.sum(np.multiply(np.delete(roc,N-1) , (t[1:] - np.delete(t,N-1)))  )  
        else:            
            pairpoints_coordinates = pd.DataFrame({"c":c,  "FPR": t,"TPR": roc})
            auc= np.sum(np.multiply(np.delete(roc,N-1) , (t[1:] - np.delete(t,N-1)))  )
        return pairpoints_coordinates,auc  ,side  
    def get_roc(self,bootstrap_replicates=500):
        controls2=self.the_controls.copy()
        cases2=self.the_cases.copy()
        n0 = len(controls2)
        n1 = len(cases2) 
        valori = np.concatenate((np.repeat(0, n0), np.repeat(1, n1)))
        ppc,auc1,the_side=self.calculate_roc(self.preds1.copy(), self.label.copy())
        p_auc1=[]
        for i in range(bootstrap_replicates):
            pD = np.random.choice(valori, size= n0 + n1, replace=False)
            _,auc_res,_=self.calculate_roc(self.preds1.copy(), pD)
            p_auc1.append(auc_res)        
        pval_auc1 = np.mean(np.array(p_auc1 )> auc1)
        print('Measurement AUC [',round(auc1,3),'] pvalue:',pval_auc1,' based on ',bootstrap_replicates,' replicates')
        return ppc,auc1,the_side
    def roc_for_ci(self,controls3,cases3,side3):
        n0 = len(controls3)
        n1 = len(cases3)        
        m=n0
        t= np.arange(0, 1+(1/m), 1/m) 
        N = len(t)
        XX = np.sort( np.concatenate([controls3,cases3]))
        XX_un=np.unique(XX)
        the_method='closest_observation'
        e = np.where(len(XX_un) > 1, np.amin(XX_un[1:] - np.delete(XX_un,len(XX_un)-1))/2, np.sqrt(np.finfo(float).eps))
        if side3 == "right":
            c = np.quantile(controls3, 1 - t, method = the_method)
            tmp=ECDF(cases3)
            roc = 1 - tmp(c)            
           
        elif side3 == "left":
            tmp1=np.quantile(controls3, t, method = the_method)
            tmp2=ECDF(controls3)
            c = np.where(tmp1 - e * tmp2(tmp1) > t,1,0)
            tmp3=ECDF(cases3)
            roc =tmp3(c)            
            
        elif side3 == "general":
            A=[]
            for i in range(1,N+1):
                if i==N:
                    roc =1
                    xu = np.amax(controls3)
                    xl=np.amax(controls3)
                else:
                    gamma = np.arange(0,i,1)
                    s_c=np.sort(controls3)
                    ecdf = ECDF(cases3)
                    index_gamma_t = np.argmax(ecdf(s_c[gamma] -e) + 1 - ecdf(s_c[m - i + gamma]))
                    gamma_t = gamma[index_gamma_t]
                    xl = s_c[gamma_t]
                    xu = s_c[m - i + gamma_t]
                    roc = ecdf(xl - e) + 1 - ecdf(xu)
                A.append([roc,xl,xu])
            results=np.array(A)
        if side3=="general":    
            xl = results[:,1]
            xu = results[:,2]
            roc=results[:,0]
            pairpoints_coordinates = pd.DataFrame({"xl":xl,  "xu":xu,"FPR": t,"TPR": roc})
            auc= np.sum(np.multiply(np.delete(roc,N-1) , (t[1:] - np.delete(t,N-1)))  )  
        else:            
            pairpoints_coordinates = pd.DataFrame({"c":c,  "FPR": t,"TPR": roc})
            auc= np.sum(np.multiply(np.delete(roc,N-1) , (t[1:] - np.delete(t,N-1)))  )
        return pairpoints_coordinates,auc  ,side3        
    def roc_conf_interv(self,confid_level=0.95,bootstrap_replicates=500,alpha1=np.nan):
        s=1 #scale parameter used to compute the smoothed kernel distribution functions
        if confid_level > 1 and confid_level <= 100:
            confid_level = confid_level/100
        elif confid_level < 0 or confid_level > 1 :
            print("confidence level should be in the 0-1 interval.")
        alpha = 1 - confid_level
        if alpha1< 0 or alpha1 > 1:
            print("alpha1 level should be in the 0-1 interval.")
        val2,auc2,sided=self.get_roc()
        controls4=self.the_controls.copy()
        cases4=self.the_cases.copy()
        Ni = len(controls4)    
        grid = np.arange(0, 1+(1/Ni), 1/Ni)
        n = len(cases4)
        m = len(controls4)
        hn = s * (min(n, m)**(-1/5)) * stdev(cases4)
        hm = s * (min(n, m)**(-1/5)) * stdev(controls4)
        cases_b=np.empty([len(cases4),bootstrap_replicates])
        controls_b=np.empty([len(controls4),bootstrap_replicates])
        #np.random.seed(123)
        for i in range(bootstrap_replicates):
            tmp1=np.random.choice(cases4, size= len(cases4), replace=True)
            tmp2=np.random.normal( 0, hn,n)
            cases_b[:,i]=np.add(tmp1,tmp2)
            tmp1=np.random.choice(controls4, size= len(controls4), replace=True)
            tmp2=np.random.normal( 0, hm,m)
            controls_b[:,i]=np.add(tmp1,tmp2  )
        X_B=np.empty([len(val2["TPR"]),bootstrap_replicates])        
        print('It may take some time, depending on the number of bootstrap replicates considered...')
        for i in range(bootstrap_replicates):
            roc_b,_,_=self.roc_for_ci(controls_b[:,i],cases_b[:,i],sided)    
            out_tmp=np.subtract(roc_b["TPR"] ,val2["TPR"]) 
            X_B[:,i]=out_tmp            
        out_tmp=np.multiply(sqrt(n) , X_B)
        sigma_B=np.empty([out_tmp.shape[0],1])
        for i in range(out_tmp.shape[0]):
            sigma_B[i]=stdev(out_tmp[i,:])
        #indexes=np.ravel(sigma_B==0)        
        #sigma_B[indexes]=np.finfo(float).eps
        #for i in range(X_B.shape[1]):
        #    X_B[indexes,i]=np.finfo(float).eps          
        tmp_div=np.true_divide(X_B,sigma_B)
        tmp_div[np.isnan(tmp_div)] = 0
        out_tmp= np.multiply( sqrt(n),tmp_div) #la divisione fa uscire 1
        U_B=np.empty([out_tmp.shape[1],1])
        L_B=np.empty([out_tmp.shape[1],1])
        for i in range(out_tmp.shape[1]):            
            U_B[i] = np.nanmax(out_tmp[:,i])
            L_B[i] = np.nanmin(out_tmp[:,i])
        if np.isnan(alpha1):
            valori=np.arange(0, alpha+0.005, 0.005)
            c1_v=[]
            c2_v=[]
            for i in valori:
                c1_v.append( np.nanquantile(U_B,  1 - i))
            valori=np.arange(alpha,-0.005, -0.005)
            valori[-1]=0            
            for i in valori:
                c2_v.append( np.nanquantile(L_B,  i))
            c1_v=np.asarray(c1_v)#U
            c2_v=np.asarray(c2_v)#L
            pos_min = np.argmin( np.subtract(c1_v, c2_v))
            alpha1 = np.arange(0, alpha+0.005, 0.005)
            alpha2 = np.arange(alpha, -0.005, -0.005) 
            alpha2[-1]=0 
            alpha1=alpha1[pos_min]
            alpha2=alpha2[pos_min]
            c1 = c1_v[pos_min]#U
            c2 = c2_v[pos_min]#L
            fixed_alpha1 = False
        
        else :
            alpha1 = alpha1
            alpha2 = alpha - alpha1
            c1_v = np.nanquantile(U_B,  1 - alpha1)
            c2_v = np.nanquantile(L_B,  alpha2)
            c1 =c1_v
            c2 =c2_v
            fixed_alpha1 = True #fixed by the user
        tmp1=np.multiply(c1 , np.true_divide(sigma_B,sqrt(n))) #L
        tmp2=np.multiply(c2 ,np.true_divide( sigma_B,sqrt(n))) #U
        L = np.subtract(val2["TPR"] ,tmp1.ravel())
        U = np.subtract(val2["TPR"] ,tmp2.ravel())
        indL=L<0
        L[indL]=0
        indU=U>1
        U[indU]=1
        indL=L>0.95
        L[indL]=0.95
        indU=U<0.05
        U[indU]=0.05
        practical_area = np.mean(U[1:] - L[1:] + U[:-1] -             L[:-1])/2
        #theoretical_area = (c1 - c2)/sqrt(n) * np.nanmean(sigma_B[1:] +      np.delete(sigma_B,Ni - 2) )/2
        theoretical_area = (c1 - c2)/sqrt(n) * np.nanmean(sigma_B[1:] +      np.delete(sigma_B,Ni - 1) )/2
        print('Theoretical area between confidence bands by trapezoidal rule ',theoretical_area)
        print('Area between lower and upper c.i. computed by trapezoidal rule ',practical_area )
        print('Confidence intervals obtained with ',alpha1,' and ',alpha2,' chance probability values [alphas]')
        return L,U
    def get_mid(self,A1):
        A=np.column_stack((A1['FPR'],A1['TPR']))
        p0 = [1, 0] # point 0
        p1 = [0, 1] # point 1

        b = (p1[1] - p0[1]) / (p1[0] - p0[0]) # gradient
        a = p0[1] - b * p0[0] # intercept
        B = (a + A[:,0] * b) - A[:,1] # distance of y value from line
        ix = np.where(B[1:] * B[:-1] < 0)[0] # index of points where the next point is on the other side of the line
        d_ratio = B[ix] / (B[ix] - B[ix + 1]) # similar triangles work out crossing points
        cross_points = np.zeros((len(ix), 2)) # empty array for crossing points
        cross_points[:,0] = A[ix,0] + d_ratio * (A[ix+1,0] - A[ix,0]) # x crossings
        cross_points[:,1] = A[ix,1] + d_ratio * (A[ix+1,1] - A[ix,1]) # y crossings

        return cross_points
    def plot_roc_youden(self):
        val1,auc2,sided=self.get_roc()
        self.density_estimation()
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)        
        ax.plot(val1['FPR'],val1['TPR'], lw=2, color="dodgerblue",label = 'AUC = %0.2f' % auc2)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.plot([0, 1], [0, 1],'k--',label='No diagnostic power')
        ax.plot([1, 0], [0, 1],linestyle='--',color='gray',label='Theoretical optimum',lw=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])        
        differential_values=val1['TPR']-val1['FPR']
        sensit=val1['TPR']
        specif=1-val1['FPR']
        youden=sensit+specif-1
        maxElem = np.amax(youden)
        youden1=max(differential_values)
        if maxElem == youden1:
            print('Youden index (or Youden׳s J statistic) ',maxElem)
            ind_max = np.where(youden == maxElem)
        else:
            print('Youden index (or Youden׳s J statistic) ',youden1)
            maxElem=youden1
            ind_max = np.where(differential_values== youden1)
        ax.plot(val1['FPR'][ind_max[0]],val1['TPR'][ind_max[0]], color="orange",marker="v", markersize=15)
        ax.vlines(x = val1['FPR'][ind_max[0]], ymin = val1['TPR'][ind_max[0]]-maxElem, ymax = val1['TPR'][ind_max[0]], colors = 'orange', label = 'Youden Index',linestyle='--')         
        mid_anchor=self.get_mid(val1)
        ax.plot(mid_anchor[:,0], mid_anchor[:,1],'k.',label='MID')
        print('MID in terms of FPR :',mid_anchor[:,0])
        print('MID in terms of TPR :',mid_anchor[:,1])
        Lower_bound,Upper_bound=self.roc_conf_interv()
        ax.fill_between(val1['FPR'], Upper_bound, Lower_bound, alpha=0.2)
        d_ns=1-val1['TPR'][ind_max[0]]
        radius=sqrt(val1['FPR'][ind_max[0]] **2 + d_ns**2)
        p=Wedge((0, 1), radius, 270, 360, facecolor='r',alpha=0.1,edgecolor='darkred',linestyle='--')
        ax.add_artist(p)
        ax.legend(loc = 'lower right')
        #plt.savefig('my_plot100.tif', dpi = 300, bbox_inches='tight')
        plt.show()
        
