import numpy as np
import scipy.stats as stats
from statistics import stdev
from math import sqrt,exp,lgamma
def qt(p1,df1,ncp=0):
       
    if ncp==0:
        result=stats.t.ppf(q=p1,df=df1,loc=0,scale=1)
    else:
        result=stats.nct.ppf(q=p1,df=df1,nc=ncp,loc=0,scale=1)
    return result
def pt(q1,df1,ncp=0):
      
    if ncp==0:
        result=stats.t.cdf(x=q1,df=df1,loc=0,scale=1)
    else:
        result=stats.nct.cdf(x=q1,df=df1,nc=ncp,loc=0,scale=1)
    return result
def dof_Satterthwaite(s1, n1, n1s, s2, n2, n2s):
    part1=(s1**2/n1s + s2**2/n2s)**2
    part2=(s1**2/n1s)**2/(n1 - 1)
    part3=(s2**2/n2s)**2/(n2 - 1)
    return part1/(part2+part3)
def polin_coeff(coef ): 

    a = np.asarray(coef)
    la = len(a)
    while  la> 1 and a[la] == 0:
        a = a[-la]
    return a
def qnorm(p,mean=0,sd=1):
     
    result=stats.norm.ppf(q=p,loc=mean,scale=sd)
    return result
def RMLE_equivTest(nT, nR, smplMuT, smplMuR, smplSigmaT, smplSigmaR, vecT, vecR, eta):
    muT = smplMuT
    muR = smplMuR
    sigmaT = smplSigmaT
    sigmaR = smplSigmaR
    oldPrmtr = np.asarray([muT, muR, sigmaT, sigmaR])
    dif = 1
    iterMax = 200
    iterat = 1
    while (dif > 1e-04 and iterat < iterMax) :
        muT = muR + eta * sigmaR
        sigmaT = sqrt(np.mean((vecT - muT)**2))
        muR = (nT * (smplMuT - eta * sigmaR)/sigmaT**2 + nR *  smplMuR/sigmaR**2)/(nT/sigmaT**2 + nR/sigmaR**2)
        tmp=[-np.mean((vecR - muR)**2), 0, 1, -nT/nR * eta * (smplMuT - muR)/sigmaT**2, nT/nR * eta**2/sigmaT**2]
        poly = np.flip(tmp)
        pz = np.roots(poly)
        
        rootFound = False
        for i, valore in reversed(list(enumerate(pz))):
            if (pz[i].imag == 0 and pz[i].real > 0) :
                sigmaR = pz[i].real
                rootFound = True
                break
        if rootFound==False:
            print("Cannot find real root for sigmaR")
        newPrmtr = np.asarray([muT, muR, sigmaT, sigmaR])
        if np.mean(np.abs(oldPrmtr - newPrmtr)) < 1e-04: 
            break
        oldPrmtr = newPrmtr
        iterat = iterat + 1
    return np.asarray([ muT, muR,  sigmaT,  sigmaR])
class TOST_T():
    is_normal=True
    same_var=True
    def __init__(self, x1, y1):
        print('Testing hypothesis of equivalence')
        self.x = np.asarray(x1)
        self.y = np.asarray(y1)


        _, p = stats.levene(x1,y1)
        if p<0.05:
            print('Input measurements do not have equal variances')
            TOST_T.same_var=False
            
        else:
            print("Input measurements have equal variances")
        shapiro_test1 = stats.shapiro(x1)
        shapiro_test2 = stats.shapiro(y1)
        if shapiro_test1.pvalue>0.05 and shapiro_test2.pvalue>0.05:
            TOST_T.is_normal=True
            print("Dealing with normal data")
        else:
            TOST_T.is_normal=False
    def run_TOST_T(self,alpha = 0.05, marginX = 1.5, sampleSizeX = 1.5):
        smplT = self.x
        smplR = self.y #Reference
        margin = marginX *stdev(smplR)
        dm = np.mean(smplT) - np.mean(smplR)
        nRLs = np.where(len(smplR) > sampleSizeX * len(smplT), sampleSizeX * len(smplT), len(smplR))
        nTLs = np.where(len(smplT) > sampleSizeX * len(smplR), sampleSizeX * len(smplR), len(smplT))
        dsigma = sqrt(stdev(smplR)**2/nRLs + stdev(smplT)**2/nTLs)

        # DF Satterthwaite
        tdf=dof_Satterthwaite(stdev(smplT),len(smplT),nTLs,stdev(smplR), len(smplR),nRLs)
        tq = qt( 1 - alpha,  tdf)
        ci = [dm - tq * dsigma, dm + tq * dsigma]
        rslt = np.where(all(abs(np.asarray(ci))/margin <=  1), 1,0)        
        prec=3
        print('NULL HYPOTHESIS:the difference between the means of two samples are not within a margin')
        print("Diff in means : ", round(dm, prec), ", std ",round(dsigma, prec), ", df ", round(tdf, prec),  ", tq ", round(tq, prec))
        print('Mean Diff. (90% CI) [',round(ci[0], prec),':',round(ci[1], prec),']')
        print('Equivalence bounds [',round(-margin,prec),':',round(margin,prec),']')
        if rslt==1:
            print('-> Found equivalence')
            print('Reject the NULL hypothesis')
        else:
            print('No equivalence')
            print('Do not reject the NULL hypothesis')
    def run_TOST_MW(self,alpha = 0.05, marginX = 1.7):
        from_append=True
        vecT=self.x
        vecR=self.y
        eta = marginX
        nT = len(vecT)
        nR = len(vecR)
        smplMuT = np.mean(vecT)
        smplMuR = np.mean(vecR)
        smplSigmaT = stdev(vecT)
        smplSigmaR = stdev(vecR)
        k = sqrt((nR - 1)/2) * exp(lgamma((nR - 1)/2) - lgamma(nR/2))
        if from_append:
            Vn=2* ( (nR-1)/2 - exp( 2*(lgamma(nR/2) - lgamma((nR-1)/2)) ) )

            nRnew = np.where(nR > 1.5*nT, 1.5*nT, nR)
            nTnew = np.where(nT > 1.5*nR, 1.5*nR, nT)

        est = RMLE_equivTest(nT, nR, smplMuT, smplMuR, smplSigmaT, smplSigmaR, vecT, vecR, -eta)
        tstatL = (smplMuT - smplMuR + eta * smplSigmaR)/sqrt(est[2]**2/nT + (1/nR + eta**2 * (1 - 1/k**2)) * est[3]**2)
        if from_append:
            tstatL = (smplMuT - smplMuR + eta * smplSigmaR)/sqrt(est[2]**2/nTnew + (1/nRnew + eta**2 * Vn/(nR-1)) * est[3]**2)
        estL = est

        est = RMLE_equivTest(nT, nR, smplMuT, smplMuR, smplSigmaT, smplSigmaR, vecT, vecR, eta)
        tstatU = (smplMuT - smplMuR - eta * smplSigmaR)/sqrt(est[2]**2/nT + (1/nR + eta**2 * (1 - 1/k**2)) * est[3]**2)
        if from_append:
            tstatU = (smplMuT - smplMuR - eta * smplSigmaR)/sqrt(est[2]**2/nTnew + (1/nRnew + eta**2 * Vn/(nR-1)) * est[3]**2)
        estU = est

        qntl = qnorm(1-alpha)
        rslt = np.where(tstatL > qntl and tstatU < -qntl, 1, 0)
        if from_append:
            ci_L=smplMuT - smplMuR - qntl*sqrt(estL[2]**2/nTnew + (1.0/nRnew + eta**2*Vn/(nR-1))*estL[3]**2)
            ci_U=smplMuT - smplMuR + qntl*sqrt(estU[2]**2/nTnew+ (1.0/nRnew + eta**2*Vn/(nR-1))*estU[3]**2)
        
        prec=2
        print('NULL HYPOTHESIS:the difference between the means of two samples are not within a margin')
        print("Estimated L [New] : mean ", round(estL[0], prec), ", std ",round(estL[2], prec))
        print("Estimated L [Ref] : mean ", round(estL[1], prec), ", std ",round(estL[3], prec))
        print("Estimated U [New] : mean ", round(estU[0], prec), ", std ",round(estU[2], prec))
        print("Estimated U [Ref] : mean ", round(estU[1], prec), ", std ",round(estU[3], prec))
        print('Statistics [L:',round(tstatL,prec),'],[U:',round(tstatU,prec),'] and Critical value:',round(qntl,prec))
        if from_append:
            print('Estimated CI Lower ',round(ci_L, prec),' CI Upper ',round(ci_U, prec))
        if rslt==1:
            print('-> Found equivalence')
            print('Reject the NULL hypothesis')
        else:
            print('No equivalence')
            print('Do not reject the NULL hypothesis')
