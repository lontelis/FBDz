import numpy as np 
import matplotlib.pyplot as plt
from scipy import integrate
from cosmopit import cosmology

print('#Fiducial cosmological parameters')
c     =3e5
hubble=0.6727
omegab=0.049         #0.022*pow(hubble,-2)
omegac=0.319 -omegab #0.119*pow(hubble,-2)
omegam=omegac+omegab
omegax=1.0   -omegam
w0    =-1.0
wa    =0.0

use_Mpc_over_h=True
if use_Mpc_over_h:
	H0_km_per_sec_per_Mpc_per_h = 100	
else:
	H0_km_per_sec_per_Mpc_per_h = 100*hubble

H00 = 100*hubble # Hubble parameter in km/s/Mpc
Ass = 2.1e-9     # 2.14e-9 #scalar Amplitude of the power spectrum
nss = 0.968
gamma_mg=0.545

print('#define the galaxy bias amplitude b0 in the linear deterministic bias approximation')
b0=1.0

print('#Spatially flat Universe')
print('#Define E(z) = H(z)/H0')
def Ez(zc,omegam=omegam,omegax=omegax):
	return( np.sqrt(1-omegam+omegam*pow(1+zc,3) ) )
def Hz(zc):
	return( Ez(zc)*H0_km_per_sec_per_Mpc_per_h )
print('#Define the comoving distances')
def drdz(zp,H0_km_per_sec_per_Mpc_per_h=H0_km_per_sec_per_Mpc_per_h,omegam=omegam,omegax=omegax):
	return( (c/H0_km_per_sec_per_Mpc_per_h)/Ez(zp,omegam=omegam,omegax=omegax) )
def rcom(zc):
	return(integrate.romberg(drdz,0,zc))
def DA(zc):
	return(rcom(zc)/(1+zc))
print('#Define the growth function in LCDM')
def f_tracer(zz,omegam=omegam,omegax=omegax,gamma_mg=gamma_mg):
	omz=omegam*pow(1+zz,3)/pow(Ez(zz,omegam=omegam,omegax=omegax),2)
	res = pow(omz,gamma_mg)
	return(res)
print('#Get the growth factor ')
def D_dz(zz,omegam=omegam,omegax=omegax,gamma_mg=gamma_mg):
	return(-f_tracer(zz,omegam=omegam,omegax=omegax,gamma_mg=gamma_mg)/(1+zz))
def D_z(zc,omegam=omegam,omegax=omegax,gamma_mg=gamma_mg):
	ans = integrate.romberg(D_dz, 0.0, zc, args=(omegam,omegax,gamma_mg))
	return(np.exp(ans))

def bias_z(zc,b0=1.0,which_bias_model='simple_deterministic_bias'):
	'''
	$b(z)$, deterministic bias 
	'''
	if which_bias_model=='constant':
		bias_z_constant = b0*(1.+0.0*zc)
		res = bias_z_constant
	elif which_bias_model=='simple_deterministic_bias':
		bias_z_deterministic = b0*(1.+zc)**(0.5)
		res = bias_z_deterministic
	elif which_bias_model=='Euclid_ELG':
		res = b0*(0.46+zc**(3./4.))
	elif which_bias_model=='DESI_LRG':
		D_z_temp = np.zeros((len(zc) ))
		for zc_i in range(len(zc)): D_z_temp[zc_i] = D_z(zc[zc_i])
		b_LRG = b0/D_z_temp # b0 = 1.70
		res = b_LRG
	elif which_bias_model=='DESI_ELG':
		D_z_temp = np.zeros((len(zc) ))
		for zc_i in range(len(zc)): D_z_temp[zc_i] = D_z(zc[zc_i])		
		b_ELG = b0/D_z_temp # b0 = 0.84
		res = b_ELG
	elif which_bias_model=='DESI_QSO':
		D_z_temp = np.zeros((len(zc) ))
		for zc_i in range(len(zc)): D_z_temp[zc_i] = D_z(zc[zc_i])		
		b_QSO = b0/D_z_temp # b0 = 1.20
		res = b_QSO
	elif which_bias_model=='DESI_BGS':
		D_z_temp = np.zeros((len(zc) ))
		for zc_i in range(len(zc)): D_z_temp[zc_i] = D_z(zc[zc_i])		
		b_BGS = b0/D_z_temp # b0 = 1.34
		res = b_BGS 
	elif which_bias_model=='GW_bias_opt': 
		""" higher value bias than the galaxy bias, since several BH in a galaxy """
		bias_GW =  b0*(0.46+zc**(3./4.))
		res = bias_GW
	elif which_bias_model=='GW_bias_realistic': 
		""" lower value bias than the galaxy bias, since only ~ 1 per galaxy detected """
		bias_GW = b0*np.sqrt(1+zc)
	elif which_bias_model=="GW_bias":
		""" https://arxiv.org/pdf/1603.02356.pdf, https://arxiv.org/pdf/2105.04262.pdf """
		D_z_temp = np.zeros((len(zc) ))
		for zc_i in range(len(zc)): D_z_temp[zc_i] = D_z(zc[zc_i])		
		bias_GW = b0*(1.+1./D_z_temp)
		res = bias_GW
	elif which_bias_model=='MSE': 
		raise('Implement me! ')
	else:
		raise('Implement me! Define the bias model, which is not in the list')
	return( res )

def f_ct(zc,f_ct0=1.0,which_fct_model='constant'):
	'''
		$f_{ct}(z)$, tensor of factor of contaminant of each target
	'''
	if which_fct_model=='constant':
		res = f_ct0*(zc*0.0+1.)
	elif which_fct_model=='sin':
		res = f_ct0*np.sin(zc*10.)**2.
	elif which_fct_model=='cos':
		res = f_ct0*np.cos(zc*10.)**2.		
	elif which_fct_model=='sinh':
		res = f_ct0*np.sinh(zc*10.)**2.
	elif which_fct_model=='cosh':
		res = f_ct0*np.cosh(zc*10.)**2.		
	return(res)

def gamma_para_z(zt=1.35,zc=1.0,omegam=omegam,omegax=omegax,w0=w0,wa=wa): 
	Hzt = cosmology.EE(zt,omegam=omegam,omegax=omegax,omegaRad=0.0,w0=w0,wa=wa)
	Hzc  = cosmology.EE(zc ,omegam=omegam,omegax=omegax,omegaRad=0.0,w0=w0,wa=wa)
	return( (1.+zt)/(1.+zc)/Hzt*Hzc )

def gamma_perp_z(zt=1.35,zc=1.0,omegam=omegam,omegax=omegax,w0=w0,wa=wa): 
	DAngular_zt = cosmology.get_dist(zt,type='dang',omegaRad=0.0,params=[omegam,omegax,w0,wa])
	DAngular_zc  = cosmology.get_dist(zc ,type='dang',omegaRad=0.0,params=[omegam,omegax,w0,wa])
	return( DAngular_zt/DAngular_zc )

def ratio_xi_z_N(zt,zc,
	N=1.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[1.0,'constant']
	):
	'''
	return	$\\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$
	'''
	N_temp                  = N

	b0_target               = params_bd_target[0]
	which_bias_model_target = params_bd_target[1]

	b0_contam               = params_bd_contam[0]
	which_bias_model_contam = params_bd_contam[1]

	f_ct0			= params_fct[0]
	which_fct_model = params_fct[1]

	gamma_para_z_temp = gamma_para_z(zt=zt,zc=zc)
	gamma_perp_z_temp = gamma_perp_z(zt=zt,zc=zc)

	D_z_target_temp = np.zeros((len(zt)))
	for zt_i in range(len(zt)): D_z_target_temp[zt_i] = D_z(zt[zt_i])
	bias_target_temp = bias_z(zt,b0=b0_target,which_bias_model=which_bias_model_target)

	f_ct_temp = f_ct(zt,f_ct0=f_ct0,which_fct_model=which_fct_model)

	D_z_contam_temp = np.zeros((len(zc)))
	for zc_i in range(len(zc)): D_z_contam_temp[zc_i] = D_z(zc[zc_i])
	bias_contam_temp = bias_z(zc,b0=b0_contam,which_bias_model=which_bias_model_contam)

	
	FBD_temp  = (1.-f_ct_temp)*bias_target_temp*D_z_target_temp \
	          + gamma_para_z_temp*gamma_para_z_temp**2.*f_ct_temp*bias_contam_temp*D_z_contam_temp
	
	res = FBD_temp**N_temp
	
	return(res)

z_target_array=np.linspace(1.0,2.0,100)
z_contam_array_higher_smaller=True
if z_contam_array_higher_smaller:
	z_contam_array=np.linspace(2.0,2.5,100)
	z_contam_higher_smaller=''
else: 
	z_contam_array=np.linspace(0.2,0.8,100)
	z_contam_higher_smaller='z_contam_smaller'

z_tc_array = np.concatenate(( z_target_array,z_contam_array ))

f_target_constant_array=f_ct(z_target_array,f_ct0=0.1,which_fct_model='constant')
f_target_sin_array     =f_ct(z_target_array,f_ct0=0.1,which_fct_model='sin')

f_contam_constant_array=f_ct(z_contam_array,f_ct0=0.1,which_fct_model='constant')
f_contam_sin_array     =f_ct(z_contam_array,f_ct0=0.1,which_fct_model='sin')

gamma_para_z_array = gamma_para_z(zt=z_target_array,zc=z_contam_array)
gamma_perp_z_array = gamma_perp_z(zt=z_target_array,zc=z_contam_array)

ratio_xi_z_N1_fct0_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.0,'constant'])

ratio_xi_z_N2_fct0_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.0,'constant'])

ratio_xi_z_N3_fct0_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=3.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.0,'constant'])

ratio_xi_z_N10_fct0_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=10.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.0,'constant'])

ratio_xi_z_N1_fct01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'constant'])

ratio_xi_z_N2_fct01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'constant'])

ratio_xi_z_N3_fct01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=3.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'constant'])

ratio_xi_z_N10_fct01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=10.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'constant'])

ratio_xi_z_N1_fctsin01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'sin'])

ratio_xi_z_N2_fctsin01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'sin'])

ratio_xi_z_N3_fctsin01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=3.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'sin'])

ratio_xi_z_N10_fctsin01_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=10.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct=[0.1,'sin'])

plt.ion()
plt.figure(1,figsize=(10,8)),plt.clf()

plt.plot(z_target_array,ratio_xi_z_N1_fct0_array,'b-')
plt.plot(z_target_array,ratio_xi_z_N1_fct01_array,'b.-',label='$N=1, f_{ct}(z)=10\%$')
plt.plot(z_target_array,ratio_xi_z_N1_fctsin01_array,'b--',label='$N=1, f_{ct}(z)=0.1\sin(0.1z)$')

plt.plot(z_target_array,ratio_xi_z_N2_fct0_array,'g-')
plt.plot(z_target_array,ratio_xi_z_N2_fct01_array,'g.-',label='$N=2, f_{ct}(z)=10\%$')
plt.plot(z_target_array,ratio_xi_z_N2_fctsin01_array,'g--',label='$N=2, f_{ct}(z)=0.1\sin(0.1z)$')

plt.plot(z_target_array,ratio_xi_z_N3_fct0_array,'r-')
plt.plot(z_target_array,ratio_xi_z_N3_fct01_array,'r.-',label='$N=3, f_{ct}(z)=10\%$')
plt.plot(z_target_array,ratio_xi_z_N3_fctsin01_array,'r--',label='$N=3, f_{ct}(z)=0.1\sin(0.1z)$')

#plt.plot(z_target_array,ratio_xi_z_N10_fct0_array,'k-')
#plt.plot(z_target_array,ratio_xi_z_N10_fct01_array,'k.-',label='$N=10, f_{ct}(z)=10\%$')
#plt.plot(z_target_array,ratio_xi_z_N10_fctsin01_array,'k--',label='$N=10, f_{ct}(z)=0.1\sin(z)$')

plt.title('$\mathcal{FBD}^N(z) \equiv \\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$',size=25)
plt.ylabel('$\mathcal{FBD}^N[z;f_{ct}(z)]$',size=25)
plt.xlabel('redshift, $z$',size=25)
plt.legend()
plt.grid()
plt.draw()
plt.show()

plt.ion()
plt.figure(2,figsize=(10,8)),plt.clf()

plt.subplot(211)

plt.plot(z_target_array,ratio_xi_z_N1_fct01_array/ratio_xi_z_N1_fct0_array,'b.-',label='$N=1, f_{ct}(z)=10\%$')
plt.plot(z_target_array,ratio_xi_z_N1_fctsin01_array/ratio_xi_z_N1_fct0_array,'b--',label='$N=1, f_{ct}(z)=0.1\sin^2(0.1z)$')

plt.plot(z_target_array,ratio_xi_z_N2_fct01_array/ratio_xi_z_N2_fct0_array,'g.-',label='$N=2, f_{ct}(z)=10\%$')
plt.plot(z_target_array,ratio_xi_z_N2_fctsin01_array/ratio_xi_z_N2_fct0_array,'g--',label='$N=2, f_{ct}(z)=0.1\sin^2(0.1z)$')

plt.plot(z_target_array,ratio_xi_z_N3_fct01_array/ratio_xi_z_N3_fct0_array,'r.-',label='$N=3, f_{ct}(z)=10\%$')
plt.plot(z_target_array,ratio_xi_z_N3_fctsin01_array/ratio_xi_z_N3_fct0_array,'r--',label='$N=3, f_{ct}(z)=0.1\sin^2(0.1z)$')

#plt.plot(z_target_array,ratio_xi_z_N10_fct01_array/ratio_xi_z_N10_fct0_array,'k.-',label='$N=10, f_{ct}(z)=10\%$')
#plt.plot(z_target_array,ratio_xi_z_N10_fctsin01_array/ratio_xi_z_N10_fct0_array,'k--',label='$N=10, f_{ct}(z)=0.1\sin^2(0.1z)$')

plt.xlim( np.min( z_tc_array) , np.max(z_tc_array) )

plt.title('$\mathcal{FBD}^N(z) \equiv \\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$ \n ',size=25)
plt.ylabel('$\mathcal{FBD}^N(z;f_{ct})\;/\;\mathcal{BD}^N(z)$',size=25)
#plt.xlabel('redshift, $z$',size=25)
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(z_target_array,f_target_constant_array,'b-' ,label='$f_{ct}(z)=10\%$, Targeted contaminants')
plt.plot(z_contam_array,f_contam_constant_array,'b.-',label='$f_{ct}(z)=10\%$, Contaminants origin')

plt.plot(z_target_array,f_target_sin_array,'r-' ,label='$f_{ct}(z)=0.1\sin^2(0.1z)$, Targeted contaminants')
plt.plot(z_contam_array,f_contam_sin_array,'r.-',label='$f_{ct}(z)=0.1\sin^2(0.1z)$, Contaminants origin')

plt.xlim( np.min( z_tc_array) , np.max(z_tc_array) )
plt.ylabel('$f_{ct}(z)$',size=25)
plt.xlabel('redshift, $z$',size=25)
plt.legend()
plt.grid()

plt.draw()
plt.show()


plt.ion()
plt.figure(3,figsize=(10,8)),plt.clf()

plt.subplot(211)
plt.plot(z_target_array,gamma_perp_z_array**2.,'b-' ,label='$\gamma_{\perp}^2$')
plt.plot(z_target_array,gamma_para_z_array,'g-' ,label='$\gamma_{||}$')
plt.plot(z_target_array,gamma_perp_z_array**2.*gamma_para_z_array,'r-' ,label='$\gamma_{\perp}^2\cdot{}\gamma_{||} $')

plt.xlim( np.min( z_tc_array) , np.max(z_tc_array) )
plt.ylabel('$\gamma_{ct,x}(z)$',size=25)
plt.legend()
plt.grid()

plt.draw()

plt.subplot(212)
plt.plot(z_target_array,f_target_constant_array,'b-' ,label='$f_{ct}(z)=10\%$, Targeted contaminants')
plt.plot(z_contam_array,f_contam_constant_array,'b.-',label='$f_{ct}(z)=10\%$, Contaminants origin')

plt.plot(z_target_array,f_target_sin_array,'r-' ,label='$f_{ct}(z)=0.1\sin^2(0.1z)$, Targeted contaminants')
plt.plot(z_contam_array,f_contam_sin_array,'r.-',label='$f_{ct}(z)=0.1\sin^2(0.1z)$, Contaminants origin')

plt.xlim( np.min( z_tc_array) , np.max(z_tc_array) )
plt.ylabel('$f_{ct}(z)$',size=25)
plt.xlabel('redshift, $z$',size=25)
plt.legend()
plt.grid()

plt.draw()
plt.show()


def ratio_xi_z_N_cross(zt,zc,
	N=1.0,
	params_bd_target=None,
	params_bd_contam=None,
	params_fct=None,
	):
	'''
	return	$\\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$
	'''
	N_temp                  = N

	gamma_para_z_temp = gamma_para_z(zt=zt,zc=zc)
	gamma_perp_z_temp = gamma_perp_z(zt=zt,zc=zc)

	D_z_target_temp = np.zeros((len(zt)))
	for zt_i in range(len(zt)): D_z_target_temp[zt_i] = D_z(zt[zt_i])

	D_z_contam_temp = np.zeros((len(params_bd_target),len(params_bd_contam),len(zc)))
	for Nt_i in range(len(params_bd_target)):
		for Nc_i in range(len(params_bd_contam)):
			for zc_i in range(len(zc)): 
				D_z_contam_temp[Nt_i,Nc_i,zc_i] = D_z(zc[zc_i])

	b0_target,which_bias_model_target,b0_contam,which_bias_model_contam = [],[],[],[]

	bias_target_temp,bias_contam_temp = [],[]

	for Nt_i in range(len(params_bd_target)): 	
		b0_target.append( params_bd_target[Nt_i][0] )
		which_bias_model_target.append( params_bd_target[Nt_i][1] )
		bias_target_temp.append( bias_z(zt,b0=b0_target[Nt_i],which_bias_model=which_bias_model_target[Nt_i]) )	

	bias_target_temp = np.array(bias_target_temp)

	for Nt_i in range(len(params_bd_target)): 	
		for Nc_i in range(len(params_bd_contam)): 	
			b0_contam.append( params_bd_contam[Nc_i][0]	)
			which_bias_model_contam.append(  params_bd_contam[Nc_i][1] )
			bias_contam_temp.append(  bias_z(zc,b0=b0_contam[Nc_i],which_bias_model=which_bias_model_contam[Nc_i]) )

	bias_contam_temp = np.array(bias_contam_temp).reshape(len(params_bd_target),len(params_bd_contam),len(zc))

	f_ct0,which_fct_model,f_ct_temp=[],[],[]

	for Nt_i in range(len(params_bd_target)): 	
		for Nc_i in range(len(params_bd_contam)): 	
			f_ct0.append( params_fct[Nt_i][Nc_i][0] )
			which_fct_model.append( params_fct[Nt_i][Nc_i][1] )
	f_ct0 = np.array(f_ct0).reshape(len(params_bd_target),len(params_bd_contam))

	which_fct_model=np.array(which_fct_model).reshape(len(params_bd_target),len(params_bd_contam))
	for Nt_i in range(len(params_bd_target)): 	
		for Nc_i in range(len(params_bd_target)): 	
			f_ct_temp.append( f_ct(zt,f_ct0=f_ct0[Nt_i][Nc_i],which_fct_model=which_fct_model[Nt_i][Nc_i]) )

	f_ct_temp = np.array(f_ct_temp).reshape(len(params_bd_target),len(params_bd_contam),len(zt))

	sum_c_of_f_ct_temp = np.sum(f_ct_temp,axis=1)

	Second_term_factor_temp = np.zeros((len(params_bd_target),len(params_bd_contam),len(zt)))
	for Nt_i in range(len(params_bd_target)): 	
		for Nc_i in range(len(params_bd_contam)): 	
			Second_term_factor_temp[Nt_i,Nc_i] = f_ct_temp[Nt_i,Nc_i]*bias_contam_temp[Nt_i,Nc_i]*D_z_contam_temp[Nt_i,Nc_i]

	Second_term_factor_temp = np.sum(Second_term_factor_temp,axis=1)

	FBD_temp = np.zeros((len(zt)))
	FBD_target_temp = np.zeros((len(params_bd_target),len(zt)))
	First_term = np.zeros((len(params_bd_target),len(zt)))
	Second_term = np.zeros((len(params_bd_target),len(zt)))
	for Nt_i in range(len(params_bd_target)): 	
		First_term[Nt_i]  = (1.-sum_c_of_f_ct_temp[Nt_i])*bias_target_temp[Nt_i]*D_z_target_temp 
		Second_term_factor_temp[Nt_i] = gamma_para_z_temp*gamma_para_z_temp**2.*Second_term_factor_temp[Nt_i] 
		FBD_target_temp[Nt_i] = First_term[Nt_i] + Second_term_factor_temp[Nt_i]
		FBD_temp   += FBD_target_temp[Nt_i]

	res = FBD_temp**N_temp
	
	return(res)

N_temp=1.0

ratio_xi_z_N1_fct01_cross_null0_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'constant'],[0.0,'constant'] ],
	params_bd_contam=[  [0.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N1_fct01_cross_null_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'constant'],[0.0,'constant'] ],
	params_bd_contam=[  [1.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.1,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)


ratio_xi_z_N1_fct01_cross_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'constant'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'constant'] ],
	params_fct      =[ [[0.1,'constant'],[0.1,'sin']] ,
	                   [[0.1,'constant'],[0.1,'constant']] ],
	)


ratio_xi_z_N1_fct01_cross_bias_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'constant'],[1.0,'simple_deterministic_bias'], [1.0,'Euclid_ELG'], [1.0,'DESI_ELG'], ],
	params_bd_contam=[  [0.0,'constant'],[0.0,'constant'],[0.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant'],[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant'],[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant'],[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant'],[0.0,'constant'],[0.0,'constant']] ,
	                   ],
	)

ratio_xi_z_N1_fct01_cross_ED_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'Euclid_ELG'],[0.84,'DESI_ELG'] ],
	params_bd_contam=[  [0.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N2_fct01_cross_ED_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[  [1.0,'Euclid_ELG'],[0.84,'DESI_ELG'] ],
	params_bd_contam=[  [0.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)
ratio_xi_z_N3_fct01_cross_ED_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=3.0,
	params_bd_target=[  [1.0,'Euclid_ELG'],[0.84,'DESI_ELG'] ],
	params_bd_contam=[  [0.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N5_fct01_cross_ED_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=5.0,
	params_bd_target=[  [1.0,'Euclid_ELG'],[0.84,'DESI_ELG'] ],
	params_bd_contam=[  [0.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)


print('test functional FBD')
plt.figure(500,figsize=(9,6)),plt.clf()
plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_null0_array,'k--',label='Null0 $\equiv \mathcal{D}^N(z)$')
#plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_null_array,'k-',label='Null $f_{ct=01}(z)=0.1 , b_{t=c}(z)=1.0$ ')
label_cross='$f_{ct \\neq 01}(z)=0.1, f_{ct=01}(z)=0.1\sin(z/10)$ \n $b_{t=0}(z)=1.0,b_{t=1}(z)=1.0\sqrt{1+z}$'
plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_array,'b-',label=label_cross)
plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_bias_array,'r--',label='constant $\\times$ deterministic $\\times$ Euclid $\\times$ DESI')

plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_ED_array,'r-',label='Euclid $\\times$ DESI, N=1')
plt.plot(z_target_array,ratio_xi_z_N2_fct01_cross_ED_array,'r-',label='Euclid $\\times$ DESI, N=2',alpha=0.8)
plt.plot(z_target_array,ratio_xi_z_N3_fct01_cross_ED_array,'r-',label='Euclid $\\times$ DESI, N=3',alpha=0.6)
plt.plot(z_target_array,ratio_xi_z_N5_fct01_cross_ED_array,'r-',label='Euclid $\\times$ DESI, N=5',alpha=0.4)

plt.title('$\mathcal{FBD}^N(z) \equiv \\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$',size=20)
plt.ylabel('$\mathcal{FBD}^N[z;f_{ct}(z)]$',size=20)
plt.xlabel('redshift, $z$',size=20)
plt.legend()
#plt.yscale('log')
plt.grid()
plt.draw()
plt.show()

plt.figure(501,figsize=(9,6)),plt.clf()
plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_null0_array/ratio_xi_z_N1_fct01_cross_null0_array,'k--',label='Null0 $\equiv \mathcal{D}^N(z)$')
#plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_null_array/ratio_xi_z_N1_fct01_cross_null0_array,'k-',label='Null $f_{ct}(z)=0.1 , b_{t=c}(z)=1.0$ ')
label_cross='$f_{ct \\neq 01}(z)=0.1, f_{ct=01}(z)=0.1\sin(z/10)$ \n $b_{t=0}(z)=1.0,b_{t=1}(z)=1.0\sqrt{1+z}$'
plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_array/ratio_xi_z_N1_fct01_cross_null0_array,'b-',label=label_cross)
plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_bias_array/ratio_xi_z_N1_fct01_cross_null0_array,'r--',label='constant $\\times$ deterministic $\\times$ Euclid $\\times$ DESI')

#plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_ED_array/ratio_xi_z_N1_fct01_cross_null0_array,'r-',label='Euclid $\\times$ DESI')

plt.plot(z_target_array,ratio_xi_z_N1_fct01_cross_ED_array/ratio_xi_z_N1_fct01_cross_null0_array,'r-',label='Euclid $\\times$ DESI, N=1')
plt.plot(z_target_array,ratio_xi_z_N2_fct01_cross_ED_array/ratio_xi_z_N1_fct01_cross_null0_array,'r-',label='Euclid $\\times$ DESI, N=2',alpha=0.8)
plt.plot(z_target_array,ratio_xi_z_N3_fct01_cross_ED_array/ratio_xi_z_N1_fct01_cross_null0_array,'r-',label='Euclid $\\times$ DESI, N=3',alpha=0.6)
plt.plot(z_target_array,ratio_xi_z_N5_fct01_cross_ED_array/ratio_xi_z_N1_fct01_cross_null0_array,'r-',label='Euclid $\\times$ DESI, N=5',alpha=0.4)

plt.title('$\mathcal{FBD}^N(z) \equiv \\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$',size=20)
plt.ylabel('$\mathcal{FBD}^N[z;f_{ct}(z)]/\mathcal{D}^N(z)$',size=20)
plt.xlabel('redshift, $z$',size=20)
plt.legend()
#plt.yscale('log')
plt.grid()
plt.draw()
plt.show()



print('#paper cases with simple_deterministic_bias')
N_temp=1
ratio_xi_z_N1_cross_null_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N2_cross_null_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N1_cross_bt11_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)


ratio_xi_z_N1_cross_bt1_fct01_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N1_cross_bt1_fct01sin_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'sin'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)


ratio_xi_z_N1_cross_bt1_fct0101_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'constant'],[0.1,'constant']] ,
	                   [[0.1,'constant'],[0.1,'constant']] ],
	)

ratio_xi_z_N1_cross_bt1_fct0101sin_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=N_temp,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'sin'],[0.1,'sin']] ,
	                   [[0.1,'sin'],[0.1,'sin']] ],
	)

print('functional FBD for the paper')
plt.figure(600,figsize=(9,6)),plt.clf()
plt.plot(z_target_array,ratio_xi_z_N1_cross_null_array,'k-',label='1 tracer')
plt.plot(z_target_array,ratio_xi_z_N1_cross_bt11_array,'b-',label='2 tracers')
plt.plot(z_target_array,ratio_xi_z_N1_cross_bt1_fct01_array,'g--',label='1 tracer, 1 contam, $f(z)=10\%$ ')
plt.plot(z_target_array,ratio_xi_z_N1_cross_bt1_fct01sin_array,'g-',label='1 tracer, 1 contam, $f(z)=10\% \sin(z/10)$ ')
plt.plot(z_target_array,ratio_xi_z_N1_cross_bt1_fct0101_array,'r--',label='2 tracers, 2 contam, $f(z)=10\%$ ')
plt.plot(z_target_array,ratio_xi_z_N1_cross_bt1_fct0101sin_array,'r-',label='2 tracers, 2 contam, $f(z)=10\%\sin(z/10)$ ')

plt.title('$\mathcal{FBD}(z) \equiv \\delta_{\mathrm{O}}(z; \\vec{\\tilde{r}})\; /\; \\delta_{\mathrm{m}}(z; \\vec{\\tilde{r}})$',size=20)
plt.ylabel('$\mathcal{FBD}[z;f_{ct}(z)]$',size=20)
plt.xlabel('redshift, $z$',size=20)
plt.legend()
#plt.yscale('log')
plt.grid()
plt.draw()
plt.show()
import subprocess
subprocess.call(['mkdir','images/'])
plt.savefig('images/FBD_N1correlators_example'+z_contam_higher_smaller+'.pdf')




ratio_xi_z_N1_cross_bt11_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N2_cross_bt11_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N3_cross_bt11_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=3.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N5_cross_bt11_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=5.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [0.0,'simple_deterministic_bias'],[0.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.0,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],
	)

ratio_xi_z_N1_cross_bt1_fct0101sin_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'sin'],[0.1,'sin']] ,
	                   [[0.1,'sin'],[0.1,'sin']] ],
	)

ratio_xi_z_N2_cross_bt1_fct0101sin_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=2.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'sin'],[0.1,'sin']] ,
	                   [[0.1,'sin'],[0.1,'sin']] ],
	)

ratio_xi_z_N3_cross_bt1_fct0101sin_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=3.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'sin'],[0.1,'sin']] ,
	                   [[0.1,'sin'],[0.1,'sin']] ],
	)

ratio_xi_z_N5_cross_bt1_fct0101sin_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=5.0,
	params_bd_target=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'simple_deterministic_bias'] ],
	params_fct      =[ [[0.1,'sin'],[0.1,'sin']] ,
	                   [[0.1,'sin'],[0.1,'sin']] ],
	)


print('functional FBD for the paper')
plt.figure(601,figsize=(9,6)),plt.clf()

#plt.plot(z_target_array,ratio_xi_z_N1_cross_bt11_array,'b-' ,label='2 tracers, N=1')
plt.plot(z_target_array,ratio_xi_z_N2_cross_bt11_array,'b--',label='2 tracers, N=2')
plt.plot(z_target_array,ratio_xi_z_N3_cross_bt11_array,'b.-',label='2 tracers, N=3')
plt.plot(z_target_array,ratio_xi_z_N5_cross_bt11_array,'b-',label='2 tracers, N=5')
#plt.plot(z_target_array,ratio_xi_z_N1_cross_bt1_fct0101sin_array,'r-' ,label='2 tracers, 2 contam, $f(z)=10\%\sin(z/10)$, N=1 ')
plt.plot(z_target_array,ratio_xi_z_N2_cross_bt1_fct0101sin_array,'r--',label='2 tracers, 2 contam, $f(z)=10\%\sin(z/10)$, N=2 ')
plt.plot(z_target_array,ratio_xi_z_N3_cross_bt1_fct0101sin_array,'r.-',label='2 tracers, 2 contam, $f(z)=10\%\sin(z/10)$, N=3 ')
plt.plot(z_target_array,ratio_xi_z_N5_cross_bt1_fct0101sin_array,'r-',label='2 tracers, 2 contam, $f(z)=10\%\sin(z/10)$, N=5 ')

plt.title('$\mathcal{FBD}^N(z) \equiv \\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$',size=20)
plt.ylabel('$\mathcal{FBD}^N[z;f_{ct}(z)]$',size=20)
plt.xlabel('redshift, $z$',size=20)
plt.legend()
#plt.yscale('log')
plt.grid()
plt.draw()
plt.show()
import subprocess
subprocess.call(['mkdir','images/'])
plt.savefig('images/FBD_Ncorrelators_example'+z_contam_higher_smaller+'.pdf')


print('functional FBD for the paper')
plt.figure(602,figsize=(9,6)),plt.clf()
plt.plot(z_target_array,ratio_xi_z_N1_cross_bt11_array/ratio_xi_z_N1_cross_null_array,'k-',label='N=1')
plt.plot(z_target_array,ratio_xi_z_N2_cross_bt11_array/ratio_xi_z_N2_cross_null_array,'k--',label='N=2')
plt.title('$\mathcal{FBD}^N(z) \equiv \\xi^{(N)}_{\mathrm{O}}(z; \\vec{\\tilde{r}}_N)\; /\; \\xi^{(N)}_{\mathrm{m}}(z; \\vec{\\tilde{r}}_N)$',size=20)
plt.ylabel('$\mathcal{FBD}^N[z;f_{ct}(z),N_t=2]/\mathcal{FBD}^N[z;f_{ct}(z),N_t=1]$',size=20)
plt.xlabel('redshift, $z$',size=20)
plt.legend()
#plt.yscale('log')
plt.grid()
plt.draw()
plt.show()
import subprocess
subprocess.call(['mkdir','images/'])
plt.savefig('images/FBD_Ncorrelators_example_1_vs_2_tracers'+z_contam_higher_smaller+'.pdf')




"""

ratio_xi_z_N1_fct01_cross_null_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[  [1.0,'constant'],[0.0,'constant'] ],
	params_bd_contam=[  [1.0,'constant'],[0.0,'constant'] ],
	params_fct      =[ [[0.1,'constant'],[0.0,'constant']] ,
	                   [[0.0,'constant'],[0.0,'constant']] ],

ratio_xi_z_N1_fct1_null_array = ratio_xi_z_N(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[1.0,'constant'],
	params_bd_contam=[1.0,'constant'],
	params_fct      =[0.1,'constant'])

ratio_xi_z_N1_fct01_cross_null_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[  [1.0,'constant'],[1.0,'simple_deterministic_bias'] ],
	params_bd_contam=[  [1.0,'simple_deterministic_bias'],[1.0,'constant'] ],
	params_fct      =[ [[0.1,'constant'],[0.1,'sin']] ,
	                   [[0.1,'sin'],[0.0,'constant']] ],
	)

print('test functional FBD')
plt.figure(500),plt.clf()
plt.plot(ratio_xi_z_N1_fct1_null_array*0.0+1.,'k--',linewidth=5.0)
plt.plot(ratio_xi_z_N1_fct01_cross_null_array/ratio_xi_z_N1_fct1_null_array,'r-')
plt.ylabel('test')
plt.xlabel('bins')
ratio_xi_z_N1_fct01_cross_null_array==ratio_xi_z_N1_fct1_null_array
"""

"""
ratio_xi_z_N1_fct01_cross_array = ratio_xi_z_N_cross(z_target_array,z_contam_array,
	N=1.0,
	params_bd_target=[[1.0,'constant'],[1.0,'constant'] ],
	params_bd_contam=[[1.0,'constant'],[1.0,'constant'] ],
	params_fct=[ [[0.1,'constant'],[0.1,'constant']] ,
	             [[0.1,'constant'],[0.1,'constant']] ],
	)
"""