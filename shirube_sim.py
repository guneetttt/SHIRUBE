"""Shirube acceptance-test harness (illustrative inputs). Run: python3 shirube_sim.py"""
import numpy as np
from scipy.stats import beta
rng = np.random.default_rng(18102026)

# ---- Launch Fit Score (illustrative title A) ----
W  = np.array([.30,.20,.20,.15,.15])
S  = np.array([85,75,80,70,70.])
C  = np.array([.7,.7,1.0,.4,1.0])          # per-factor confidence
d  = 20*(1-C)                               # half-width in points
score = W@S; lo = W@(S-d); hi = W@(S+d); conf = W@C
print("A score %.1f range %.1f-%.1f conf %.2f"%(score,lo,hi,conf))

# ---- beta-binomial worked example ----
K=50
def post(prior, x, n):
    a0,b0=prior*K,(1-prior)*K; a,b=a0+x,b0+n-x
    return a/(a+b), beta.ppf(.05,a,b), beta.ppf(.95,a,b), beta.sf(.03,a,b)
for name,pr,x,n in [("A",.04,10,400),("B",.035,24,400),("C",.03,0,0)]:
    m,l,h,p = post(pr,x,n); print(name,"mean %.1f%% 90%% %.1f-%.1f P(>=3%%) %.1f%%"%(m*100,l*100,h*100,p*100))

# ---- Test 1: weight sensitivity (3 illustrative titles, +-20% jitter, renormalised) ----
T = np.array([[85,75,80,70,70],[80,70,60,75,70],[60,65,70,55,65.]])  # A,B,C factor scores
base = (T@W).argmax(); print("base scores", (T@W).round(1))
chg=0; N=1000
for _ in range(N):
    w = W*rng.uniform(.8,1.2,5); w/=w.sum()
    chg += (T@w).argmax()!=base
print("top title unchanged in %.1f%% of %d runs"%(100*(1-chg/N),N))

# ---- Test 2: coverage of 90% interval, 10,000 simulated pilots ----
def coverage(true_sampler, prior_mean, strength, n=400, R=10000):
    p = true_sampler(R); x = rng.binomial(n,p)
    a=prior_mean*strength+x; b=(1-prior_mean)*strength+n-x
    lo,hi=beta.ppf(.05,a,b),beta.ppf(.95,a,b)
    return ((p>=lo)&(p<=hi)).mean()
pm=.04
for K_ in (50,20,10):
    # (i) true rates drawn from the prior itself  (ii) prior mean 2x too high  (iii) 2x too low
    c1=coverage(lambda R: rng.beta(pm*K_,(1-pm)*K_,R),pm,K_)
    c2=coverage(lambda R: rng.beta(pm*.5*K_*2,(1-pm*.5)*K_*2,R),pm,K_)
    c3=coverage(lambda R: rng.beta(pm*2*K_*2,(1-pm*2)*K_*2,R),pm,K_)
    print("strength",K_,"well-specified %.1f%%  prior 2x high %.1f%%  prior 2x low %.1f%%"%(100*c1,100*c2,100*c3))

# ---- sample size ----
z=1.96; print("n for +-5%%: %.1f"%(z*z*.25/.05**2))
# ---- Fusion Drop economics ----
u=900*.98-500; print("unit contribution",u,"x150 =",u*150, "margin %.1f%%"%(100*u*150/(900*150)))
# ---- Scale illustration: payers at posterior low/mean/high, 100k monthly readers, Rs79 ----
for r in (.040,.057,.076): print(r, round(100000*r*79))
