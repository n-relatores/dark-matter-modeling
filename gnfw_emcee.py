import numpy as np
import matplotlib.pyplot as pl
import sys
import emcee
from datetime import datetime
from matplotlib.ticker import MaxNLocator

gal = np.genfromtxt('galaxy_data', names=['galaxy','fixed','band','grade'],dtype=None)
disk = 'free' 
CO_gal = ['ngc853','ngc949','ngc1012','ngc1035','ngc4451','ngc4632','ngc5303','ngc5692','ngc5949','ngc6106','ngc6207']

def lnprior(theta):
    beta, logM200, c200, lnjitter, lnML = theta
    if 0.0 < beta < 2.0 and 8 < logM200 < 12.0 and 1.0 < c200 < 40.0:
        lnp = -np.log(np.sqrt(2*np.pi)*0.7) - 0.5*lnjitter**2/0.7**2
	lnp += -np.log(np.sqrt(2*np.pi*std**2)) - 0.5*(lnML-mean)**2/std**2 
	return lnp
    return -np.inf

def gnfw(r, beta, rm2, logrho):
    lnx = np.linspace(np.log(np.min(r)/1000),np.log(np.max(r)),num=1000) 
    x=np.exp(lnx)
    r_s = rm2/(2-beta)
    integrand = r_s**3*4*np.pi*(x/r_s)**(3-beta)*10**logrho/((1+(x/r_s))**(3-beta))
    integral = np.cumsum(integrand*(lnx[1]-lnx[0]))  
    mass = np.interp(np.log(r), lnx, integral) 
    v = np.sqrt(G*mass/r)
    return v

def M200_c200_to_rho_rm2(beta, logM200, c200, rho_crit):
    M200 = 10**logM200
    r200 = (M200*3/(200*rho_crit*4*np.pi))**(1/3.) 
    rm2 = r200/c200
    r_s = rm2/(2-beta) 
    lnx = np.linspace(np.log(1e-3),np.log(r200),num=1000) 
    x=np.exp(lnx)
    integrand = r_s**3*4*np.pi*(x/r_s)**(3-beta)/((1+(x/r_s))**(3-beta))
    mass = np.sum(integrand*(lnx[1]-lnx[0])) 
    rho = M200/mass
    logrho = np.log10(rho)
    return rm2,logrho

def lnlike(theta, r, Vt, eVt):
    beta, logM200, c200, lnjitter, lnML = theta
    ML = np.exp(lnML)
    rm2,logrho = M200_c200_to_rho_rm2(beta,logM200,c200,rhocrit)
    dmmodel = gnfw(r, beta, rm2, logrho)
    starmodel = np.interp(r, stardata['r'], stardata['V'])*np.sqrt(ML)
    HImodel = np.interp(r, HIdata['r'], HIdata['V2'])
    if galaxy in CO_gal:
    	COmodel = np.interp(r, COdata['r'], COdata['V2']) 
    	model = np.sqrt(dmmodel**2 + starmodel**2 + HImodel + COmodel) 
    else:
        model = np.sqrt(dmmodel**2 + starmodel**2 + HImodel)
    inv_sigma2 = 1.0/eVt**2
    jitter = np.exp(lnjitter)
    return -Vt.size*np.log(jitter)-0.5*(np.dot(Vt-model,np.dot(iCov/jitter**2,Vt-model)))

def lnprob(theta, r, Vt, eVt):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, r, Vt, eVt)

def gnfw(r, beta, rm2, logrho):
    lnx = np.linspace(np.log(np.min(r)/1000),np.log(np.max(r)),num=1000) 
    x=np.exp(lnx)
    r_s = rm2/(2-beta)
    integrand = r_s**3*4*np.pi*(x/r_s)**(3-beta)*10**logrho/((1+(x/r_s))**(3-beta))
    integral = np.cumsum(integrand*(lnx[1]-lnx[0])) 
    mass = np.interp(np.log(r), lnx, integral) 
    v = np.sqrt(G*mass/r)
    return v

def star(r, ML):
    return np.interp(r, stardata['r'], stardata['V'])*np.sqrt(ML)

G =  4.3023e-6
rhocrit = 136 # 3H_0^2/8*pi*G (solar masses per cubic kpc)
pixscale = 1.32
date = datetime.now().strftime('%m-%d-%Y %H:%M')

results = np.genfromtxt('results.log', usecols = (0,6), dtype = None, names=['name','distance'])
MtoL = np.genfromtxt('ML_predicted.dat', usecols=(0,1,2), dtype=None, names=['name','mean','std'])

g= open('beta_star.dat','w')
g.write('#galaxy \tmean\tstd\t(beta 0.3-0.8 kpc)\n')

p = open('gal_parameters', 'a')

for galaxy in gal['galaxy']:
    fixed = gal['fixed'][gal['galaxy']==galaxy][0]
    band = gal['band'][gal['galaxy']==galaxy][0]

    m = MtoL['mean'][MtoL['name']==galaxy][0]
    mean = np.log(m)
    std = MtoL['std'][MtoL['name']==galaxy][0]/m

    print(galaxy+' emcee', disk, '%0.2f' %m)

    iCov = np.loadtxt('../diskfit/'+galaxy+'/'+galaxy+'_'+fixed+'_iCov.txt')

    distance = results['distance'][results['name']==galaxy.upper()][0]

    if fixed == 'fixnothing':
        sh= 64
    else:
        sh=63

    data = np.genfromtxt('../diskfit/'+galaxy+'/'+galaxy+'_'+fixed,usecols = (0,2,3), names=['r','Vt','eVt'],skip_header=sh) 
    stardata = np.genfromtxt('../diskfit/rotcurves/%s_%s_rot.dat' %(galaxy.upper(),band), names=['r','V'])
    data['r'] *= pixscale*distance*4.848e-3
    stardata['r']*= distance*4.848e-3
    if band == 'Ch2':
        stardata['V']*= np.sqrt(1.15)
    if data['Vt'][-1] <= 0:
        data['Vt'] *= -1

    HIdata = np.genfromtxt('../HI/HI_rotcurves/%s_HI_vels' %galaxy, usecols = (0,1,2), names=['r','V2','V']) #***
    if galaxy in CO_gal:
        COdata = np.genfromtxt('../m0-fits/H2_rotcurves/%s_H2_vels_piecewise' %galaxy, usecols = (0,1,2), names=['r','V2','V'])

    ndim, nwalkers = 5, 50
    pos = [[np.random.uniform(low=0,high=2),
                np.random.uniform(low=8,high=12),
                np.random.uniform(low=1,high=40),
                0.0+0.7*np.random.randn(),
		mean+std*np.random.randn()] for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data['r'], data['Vt'], data['eVt']))

    nsteps = 500
    print("Running first MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    print("Done with first.")

    sampler=sampler.chain
    goodsample = sampler[:,nsteps/2:,:]
    finalsample = goodsample.reshape(nwalkers*nsteps/2,ndim)
    beta_mcmc, logM200_mcmc, c200_mcmc, jit_mcmc, ML_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                             zip(*np.percentile(finalsample, [16, 50, 84], axis=0)))

    pos = [[np.clip(beta_mcmc[0]+(beta_mcmc[1]+beta_mcmc[2])/2*np.random.randn(),0,2),
                np.clip(logM200_mcmc[0]+(logM200_mcmc[1]+logM200_mcmc[2])/2*np.random.randn(),8,12),
                np.clip(c200_mcmc[0]+(c200_mcmc[1]+c200_mcmc[2])/2*np.random.randn(),10,40),
                jit_mcmc[0]+(jit_mcmc[1]+jit_mcmc[2])/2*np.random.randn(),  ML_mcmc[0]+(ML_mcmc[1]+ML_mcmc[2])/2*np.random.randn()] for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data['r'], data['Vt'], data['eVt']))

    nsteps = 10000
     # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    print("Done. Starting velocity calculations.")

    np.save(disk+'/samplers/%s_%s_sampler' %(galaxy,disk),sampler.chain)
    np.save(disk+'/samplers/%s_%s_lnprob' %(galaxy,disk),sampler.lnprobability)


    ## Calculate and save using chain:

    goodsample = sampler.chain[:,nsteps/2:,:]
    goodsample = goodsample.reshape(nwalkers*nsteps/2,ndim)

    rm2 = np.zeros(goodsample.shape[0])
    logrho = np.zeros(goodsample.shape[0])
  
    beta_star = np.zeros(nwalkers*nsteps/2)
    x = np.array([0.3, 0.8])

    r=np.linspace(np.min(data['r'])/10, np.max(data['r']),100)
    v_dm = np.zeros((nwalkers*nsteps/2,len(r)))
    v_st = np.zeros((nwalkers*nsteps/2,len(r)))
    v = np.zeros((nwalkers*nsteps/2,len(r)))
    
    for i in range(goodsample.shape[0]):
        rm2[i],logrho[i] = M200_c200_to_rho_rm2(goodsample[i,0], goodsample[i,1], goodsample[i,2], rhocrit)
	
        r_s = rm2[i]/(2-goodsample[i,0]) 
      	rho = 1/((x/r_s)**goodsample[i,0] * (1+x/r_s)**(3-goodsample[i,0]))
      	beta_star[i] = -np.log(rho[1]/rho[0])/np.log(x[1]/x[0])

        ML = np.mean(np.exp(goodsample[i,4]))

        v_dm[i] = gnfw(r, goodsample[i,0], rm2[i], logrho[i])
        v_st[i] = star(r, ML)
        v[i] = np.sqrt(v_dm[i]**2 + v_st[i]**2)

    finalsample = np.zeros((nwalkers*nsteps/2,ndim+2))
    finalsample[:,0:ndim] = goodsample
    finalsample[:,ndim] = rm2
    finalsample[:,ndim+1] = logrho
    #finalsample[:,ndim+2] = beta_star **change this and run again eventually 

    np.save(disk+'/samplers/%s_%s_finalsample' %(galaxy,disk), finalsample)

    mean_beta = np.mean(beta_star)
    std_beta = np.std(beta_star)
    g.write(galaxy+' \t%0.2f\t%0.2f\n' %(mean_beta, std_beta))

    ML = np.mean(np.exp(goodsample[:,4]))
    ML_std = np.std(np.exp(goodsample[i,4]))

    p.write('\n%s, %s (%s):\n' %(galaxy,disk,date))
    p.write('beta = %0.2f +- %0.2f\n' %(mean_beta, std_beta))
    p.write('ML = %0.2f +- %0.2f\n' %(ML, ML_std))
    

    v_dm_final = np.mean(v_dm, axis=0)
    v_st_final = np.mean(v_st, axis=0)
    v_final = np.mean(v, axis=0)

    np.save(disk+'/vels/%s_%s_v_dm' %(galaxy,disk), v_dm_final)
    np.save(disk+'/vels/%s_%s_v_st' %(galaxy,disk), v_st_final)
    np.save(disk+'/vels/%s_%s_v_tot' %(galaxy,disk), v_final)

    np.save(disk+'/vels/%s_%s_v_dm_chain' %(galaxy,disk), v_dm)
    np.save(disk+'/vels/%s_%s_v_st_chain' %(galaxy,disk), v_st)
    np.save(disk+'/vels/%s_%s_v_tot_chain' %(galaxy,disk), v)


g.close()
p.close()
