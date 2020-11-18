import numpy as np
import matplotlib.pyplot as pl
import emcee
from matplotlib.ticker import MaxNLocator
import corner
from datetime import datetime
import sys

gal = np.genfromtxt('galaxy_data', names=['galaxy','fixed','band','grade'],dtype=None)
disk = 'free'
radius = '300-800'
CO_gal = ['ngc853','ngc949','ngc1012','ngc1035','ngc4451','ngc4632','ngc5303','ngc5692','ngc5949','ngc6106','ngc6207']

G =  4.3023e-6
rhocrit = 136
pixscale = 1.32
ndim, nwalkers, nsteps = 5, 50, 10000
date = datetime.now().strftime('%m-%d-%Y %H:%M')

results = np.genfromtxt('results.log', usecols = (0,6), dtype = None, names=['name','distance'])
beta = np.genfromtxt('beta_star.dat', usecols = (0,1), names = ['galaxy','beta'], dtype=None)


def star(r, ML):
    return np.interp(r, stardata['r'], stardata['V'])*np.sqrt(ML)

def gas(r):
    HImodel = np.interp(r, HIdata['r'], HIdata['V2']) 
    if galaxy in CO_gal:
    	COmodel = np.interp(r, COdata['r'], COdata['V2']) 
    	return HImodel + COmodel 
    else:
        return HImodel

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

def gnfw(r, beta, rm2, logrho):
    lnx = np.linspace(np.log(np.min(r)/1000),np.log(np.max(r)),num=1000) 
    x=np.exp(lnx)
    r_s = rm2/(2-beta)
    integrand = r_s**3*4*np.pi*(x/r_s)**(3-beta)*10**logrho/((1+(x/r_s))**(3-beta))
    integral = np.cumsum(integrand*(lnx[1]-lnx[0])) 
    mass = np.interp(np.log(r), lnx, integral) 
    v = np.sqrt(G*mass/r)
    return v


for galaxy in gal['galaxy']:
    fixed = gal['fixed'][gal['galaxy']==galaxy][0]
    band = gal['band'][gal['galaxy']==galaxy][0]
    grade = gal['grade'][gal['galaxy']==galaxy][0]
    mean_beta = beta['beta'][beta['galaxy']==galaxy][0]

    print(galaxy+' plots', disk)


    distance = results['distance'][results['name']==galaxy.upper()][0]
    if fixed == 'fixnothing':
        sh= 64
    else:
        sh=63

    #data = np.genfromtxt('../diskfit/'+galaxy+'/'+galaxy+'_'+fixed,usecols = (0,2,3), names=['r','Vt','eVt'],skip_header=sh)
    if galaxy in ['ugc3371', 'ugc11891']:
        data = np.genfromtxt('../diskfit/'+galaxy+'/'+galaxy+'_'+fixed,usecols = (0,2,3), names=['r','Vt','eVt'],skip_header=sh)
    else:
        data = np.genfromtxt('../diskfit/'+galaxy+'/'+galaxy+'_'+fixed,usecols = (0,2,3,4,5), names=['r','Vt','eVt','Vr','eVr'],skip_header=sh)
    stardata = np.genfromtxt('../diskfit/rotcurves/%s_%s_rot.dat' %(galaxy.upper(),band), names=['r','V'])
    data['r'] *= pixscale*distance*4.848e-3
    stardata['r']*= distance*4.848e-3
    if band == 'Ch2':
        stardata['V']*= np.sqrt(1.15)
    if data['Vt'][-1] <= 0:
        data['Vt'] *= -1
        if galaxy not in ['ugc3371', 'ugc11891']:
            data['Vr'] *= -1 
    
    HIdata = np.genfromtxt('../HI/HI_rotcurves/%s_HI_vels' %galaxy, usecols = (0,1,2), names=['r','V2','V']) 
    if galaxy in CO_gal:
        COdata = np.genfromtxt('../m0-fits/H2_rotcurves/%s_H2_vels_piecewise' %galaxy, usecols = (0,1,2), names=['r','V2','V']) 

    finalsample = np.load(disk+'/samplers/%s_%s_finalsample.npy' %(galaxy,disk))

    v_dm = np.load(disk+'/vels/%s_%s_v_dm.npy' %(galaxy,disk))
    v_st = np.load(disk+'/vels/%s_%s_v_st.npy' %(galaxy,disk))
    v_tot = np.load(disk+'/vels/%s_%s_v_tot.npy' %(galaxy,disk))

    r=np.linspace(np.min(data['r'])/10, np.max(data['r']),100)

    v_HI = np.interp(r, HIdata['r'], HIdata['V'])
    if galaxy in CO_gal:
        v_CO = np.interp(r, COdata['r'], COdata['V'])

    pl.plot(r,v_tot, color='m', label='Total')
    pl.plot(r,v_dm, color='b', label='Dark Matter')
    pl.plot(r,v_st,color='g', label='Stellar')
    pl.plot(r,v_HI, color='r', label='Atomic Gas')
    if galaxy in CO_gal:
        pl.plot(r,v_CO, color='orange', label='Molecular Gas')
    pl.errorbar(data['r'], data['Vt'], data['eVt'], color ='c',label='PCWI')
    if galaxy not in ['ugc3371', 'ugc11891']:
        pl.errorbar(data['r'], data['Vr'], data['eVr'], color ='c', linestyle='--') 
    pl.xlabel('r (kpc)')
    pl.ylabel('v (km/s)')
    pl.legend(loc=2) #4
    ax = pl.gca()
    ax.yaxis.set_ticks_position('both')
    yl = ax.get_xlim()/(distance*4.848e-3)
    ax2 = ax.twiny()
    ax2.set_xlim(yl)
    ax2.set_xlabel('r (arcsec)') #no grade = 0.92
    pl.title('%s (%d): ' r'$\beta^*$' '= %0.2f' %(galaxy.upper(),grade,mean_beta), y=0.92, x=0.55)
    #pl.title('%s: ' r'$\beta^*$' '= %0.2f\nGrade = %d' %(galaxy.upper(),mean_beta,grade), y=0.85, x=0.5)
    ##pl.title('%s\n' r'$\beta^*$' '= %0.2f\nGrade = %d' %(galaxy.upper(),mean_beta,grade), y=0.8, x=0.1)
    pl.savefig(disk+'/%s_%s_plot.png' %(galaxy,disk), dpi=1000)
    pl.close()


   

