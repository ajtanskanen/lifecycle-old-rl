'''

    dyn_prog.py
    
    implements a scheme similar to solving valuation of american options for the life cycle model
    this is a kind of dynamic programming scheme
    

'''

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import Benefits
import gym_unemployment
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm_notebook as tqdm
from lifecycle_rl import Lifecycle
from scipy.interpolate import interpn,interp1d,interp2d,RectBivariateSpline,PchipInterpolator
from scipy.integrate import quad,trapz
import time

class DynProgLifecycleRev(Lifecycle):

    def __init__(self,minimal=True,env=None,timestep=1.0,ansiopvraha_kesto300=None,min_retirementage=None,
                    ansiopvraha_kesto400=None,karenssi_kesto=None,osittainen_perustulo=None,
                    ansiopvraha_toe=None,plotdebug=False,mortality=None,
                    gamma=None,n_palkka=None,n_elake=None,n_tis=None,n_palkka_future=None,
                    max_pension=None,min_wage=None,max_wage=None,perustulo=None,perustulomalli=None,perustulo_korvaa_toimeentulotuen=None):

        super().__init__(minimal=minimal,env=env,timestep=timestep,ansiopvraha_kesto300=ansiopvraha_kesto300,
                    ansiopvraha_kesto400=ansiopvraha_kesto400,karenssi_kesto=karenssi_kesto,min_retirementage=min_retirementage,
                    ansiopvraha_toe=ansiopvraha_toe,mortality=mortality,plotdebug=plotdebug,
                    gamma=gamma,perustulo=perustulo,perustulomalli=perustulomalli,osittainen_perustulo=osittainen_perustulo,
                    perustulo_korvaa_toimeentulotuen=perustulo_korvaa_toimeentulotuen)
        
        '''
        Alusta muuttujat
        '''
        #self.min_salary=1000
        #self.hila_palkka0 = 1000 # = self.min_salary # 0
        self.spline=True
        self.spline_approx='linear' #'cubic'
        self.minspline=False # rajoita splineille alarajaa
        self.monotone_spline=False
        #self.spline_approx='quadratic'
        #self.spline_approx='linear'
        self.integmethod=0 # compare
        
        self.pw_bivariate=False
        self.pw_maxspline=1 # 1 = linear, 3 = cubic

        self.n_employment=3
        
        self.epsabs=1e-6
        
        # dynaamisen ohjelmoinnin parametrejä
        self.n_palkka = 20
        self.n_palkka_future = 21
        self.n_palkka_future_tr = 201
        self.n_elake = 40
        self.n_oapension = 200
        self.n_tis = 5 # ei vaikutusta palkkaan
        self.min_wage=1_000
        self.max_wage=85_000
        self.min_pension=0
        self.max_pension=50_000
        
        self.min_paid_pension=730*12
        
        self.min_oapension=500*12
        self.max_oapension=60_000
        
        if n_palkka is not None:
            self.n_palkka=n_palkka
        if n_palkka_future is not None:
            self.n_palkka_future=n_palkka_future
        if n_elake is not None:
            self.n_elake=n_elake
        if n_tis is not None:
            self.n_tis=n_tis
        if max_wage is not None:
            self.max_wage=max_wage
        if min_wage is not None:
            self.min_wage=min_wage
        if max_pension is not None:
            self.max_pension=max_pension
        
        self.deltapalkka = (self.max_wage-self.min_wage)/(self.n_palkka-1)
        self.deltaelake = (self.max_pension-self.min_pension)/(self.n_elake-1)
        self.delta_paid_pension = (self.max_pension-self.min_paid_pension)/(self.n_elake-1)
        self.delta_oapension = (self.max_oapension-self.min_oapension)/(self.n_oapension-1)
        self.deltatis = 1
        
        self.include_pt=False
        
        self.midfuture=int(np.floor(self.n_palkka_future/2))
        self.deltafuture=8*0.07*0.5/self.midfuture

        self.min_grid_age=self.min_age
        self.max_grid_age=self.max_age
        
        print('min',self.min_retirementage)
        
        if False:
            self.unemp_wageshock=1.0 #0.95
        else:
            self.unemp_wageshock=0.95
        
        if self.spline:
            self.get_V=self.get_V_spline
            self.get_V_vector=self.get_V_vector_spline
            self.get_actV=self.get_actV_spline
            self.get_actReward=self.get_actReward_spline
            self.get_act=self.get_act_spline
        else:
            self.get_V=self.get_V_nospline
            self.get_V_vector=self.get_V_vector_nospline
            self.get_actV=self.get_actV_nospline
            self.get_actReward=self.get_actReward_nospline
            self.get_act=self.get_act_nospline
        
    def init_grid(self):
        self.Hila = np.zeros((self.n_time+1,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_palkka))
        self.actHila = np.zeros((self.n_time+1,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_palkka,self.n_acts))        
        self.actReward = np.zeros((self.n_time+1,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_palkka,self.n_acts))        
        self.oaHila = np.zeros((self.n_time+1,self.n_oapension))
        self.oaactHila = np.zeros((self.n_time+1,self.n_oapension))        
        self.oaactReward = np.zeros((self.n_time+1,self.n_oapension))        

    def explain(self):
        print(f'n_palkka {self.n_palkka}\nn_elake {self.n_elake}\nn_palkka_future {self.n_palkka_future}\n')
        print('min_wage {} min_pension {}'.format(self.min_wage,self.min_pension))
        print('deltapalkka {} deltaelake {} delta_oapension {}'.format(self.deltapalkka,self.deltaelake,self.delta_oapension))
        print('n_tis {} deltatis {}'.format(self.n_tis,self.deltatis))
        print('gamma {} timestep {}'.format(self.gamma,self.timestep))
        self.env.explain()

    def map_elake(self,v,emp=1):
        if emp==2:
            #return self.min_paid_pension+self.delta_paid_pension*v # pitäisikö käyttää exp-hilaa?
            return self.min_oapension+self.delta_oapension*v # pitäisikö käyttää exp-hilaa?
        else:
            return self.min_pension+self.deltaelake*v # pitäisikö käyttää exp-hilaa?

    def inv_elake(self,v,emp=1):
        if emp==2:
            #vmin=max(0,min(self.n_elake-2,int(np.floor((v-self.min_paid_pension)/self.delta_paid_pension))))
            #vmax=vmin+1
            #w=(v-self.min_paid_pension)/self.delta_paid_pension-vmin # lin.approximaatio
            vmin=max(0,min(self.n_oapension-2,int(np.floor((v-self.min_oapension)/self.delta_oapension))))
            vmax=vmin+1
            w=(v-self.min_oapension)/self.delta_oapension-vmin # lin.approximaatio
        else:
            vmin=max(0,min(self.n_elake-2,int(np.floor((v-self.min_pension)/self.deltaelake))))
            vmax=vmin+1
            w=(v-self.min_pension)/self.deltaelake-vmin # lin.approximaatio
        
        if w<0:
            print(f'w<0: {w} {v} {vmin}')
            w=0
            raise ValueError('A very specific bad thing happened.')        

        return vmin,vmax,w

#     def map_exp_elake(self,v):
#         return self.min_pension+self.deltaelake*(np.exp(v*self.expelakescale)-1)
# 
#     def inv_exp_elake(self,v):
#         vmin=max(0,min(self.n_elake-2,int((np.log(v-self.min_pension)+1)/self.deltaelake)))
#         vmax=vmin+1
#         vmin_value=self.map_exp_elake(vmin)
#         vmax_value=self.map_exp_elake(vmax)
#         w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio
# 
#         return vmin,vmax,w

    def test_map_palkka(self):
        '''
        debug function
        '''
        for k in range(1000,100000,1000):
            vmin,vmax,w=self.inv_elake(k,emp=2)
            p2=(1-w)*self.map_elake(vmin,emp=2)+w*self.map_elake(vmax,emp=2)
            print(k,p2,vmin,vmax,w)
            
        for p in range(self.n_palkka): 
            palkka=self.map_palkka(p)
            print(palkka)

    def map_palkka(self,v,midpoint=False):
        if midpoint:
            return self.min_wage+max(0,self.deltapalkka*(v+0.5))
        else:
            return self.min_wage+max(0,self.deltapalkka*v) 

    def inv_palkka(self,v):
        q=int(np.floor((v-self.min_wage)/self.deltapalkka))
        vmin=int(max(0,min(self.n_palkka-2,q)))
        vmax=vmin+1
        w=(v-self.min_wage)/self.deltapalkka-vmin # lin.approximaatio
        
        if w<0:
            print(f'w<0: {w} {v} {vmin}')
            w=0
            raise ValueError('A very specific bad thing happened.')

        return vmin,vmax,w

    def map_palkka_future(self,palkka,v,med,state=1,midpoint=False):
        #if state==0:
        #    kerroin=self.unemp_wageshock
        #else:
        #    kerroin=1.0
        if midpoint:
            return med*(1+(v+0.5-self.midfuture)*self.deltafuture)
        else:
            return med*(1+(v-self.midfuture)*self.deltafuture)

    def map_palkka_future_v2(self,palkka,age,state=1,midpoint=False):
        
        if midpoint:
            x=[1.0*(v+1)/(self.n_palkka_future) for v in range(self.n_palkka_future)]
        else:
            x=[1.0*(0.5+v)/(self.n_palkka_future) for v in range(self.n_palkka_future)]

        w=self.env.wage_process_map(x,palkka,age,state=state)
        return w

    def map_palkka_future_v3(self,palkka,age,state=1,midpoint=False):
        x=0.0001+np.array([0.9998*v/self.n_palkka_future for v in range(self.n_palkka_future+1)])

        w=self.env.wage_process_map(x,palkka,age,state=state)
        return w

    def test_palkka_future(self):
        for s in range(2):
            for palkka in range(1000,50000,5000):
                for v in range(self.n_palkka_future):
                    p=self.map_palkka_future(palkka,v,s)
                    qmin,qmax,ql=self.inv_palkka_future(palkka,p,s)
                    print(f'{palkka}: {p} {qmin} {qmax} {ql} {v}')

#     def map_exp_palkka(self,v):
#         return self.hila_palkka0+self.deltapalkka*(np.exp(v*self.exppalkkascale)-1)
# 
#     def inv_exp_palkka(self,v):
#         vmin=max(0,min(self.n_palkka-2,int((np.log(v-self.hila_palkka0)+1)/self.deltapalkka)))
#         vmax=vmin+1
#         vmin_value=self.map_exp_palkka(vmin)
#         vmax_value=self.map_exp_palkka(vmax)
#         w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio
# 
#         return vmin,vmax,w

    def map_tis(self,v):
        return v

    def inv_tis(self,v):
        return int(min(self.n_tis-1,v))

    # lineaarinen approksimaatio
    def get_V_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,show=False,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t = self.map_grid_age(age)
        
        if age>self.max_age:
            return 0.0

        tismax=self.inv_tis(time_in_state)
        
        if emp==2:
            tismax=0
            #x = np.linspace(self.min_paid_pension,self.max_pension, self.n_elake)
            #y = self.Hila[t,0,:,emp,tismax,0]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaHila[t,:]
            if self.monotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x,y,fill_value="extrapolate",kind=self.spline_approx)
            if self.minspline:
                V1=max(y[0],f(max(self.min_oapension,elake))) # max tarkoittaa käytännössä takuueläkettä
            else:
                V1=max(0,f(max(self.min_oapension,elake))) # max tarkoittaa käytännössä takuueläkettä
            #print(x,y)
            #print(elake,V1)
        else:    
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            #p2min,p2max,wp2=self.inv_palkka(wage)        
            if self.pw_bivariate:
                p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                values=(1-wp)*self.Hila[t,pmin,:,emp,tismax,:]+wp*self.Hila[t,pmax,:,emp,tismax,:]
                g=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline) # default kx=3,ky=3 DOES NOT EXTRAPOLATE!
                V1 = np.squeeze(g(elake,wage))
            else:
                #p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                values=(1-we)*self.Hila[t,:,emin,emp,tismax,:]+we*self.Hila[t,:,emax,emp,tismax,:]
                g=RectBivariateSpline(w,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline) # default kx=3,ky=3
                if self.minspline:
                    V1 = max(values[0,0],np.squeeze(g(old_wage,wage)))
                else:
                    V1 = max(0.0,np.squeeze(g(old_wage,wage)))
                #print(w)
                #print(values)
                #print(old_wage,wage,V1)

        if show:      
            print(f'getV({emp},{elake},{old_wage},{wage}): p2min {p2min} p2max {p2max} wp2 {wp2})')
            print(self.Hila[t,pmin,emin,emp,tismax,p2min],self.Hila[t,pmin,emin,emp,tismax,p2max])
            
        V=np.maximum(0,V1)

        return V
        
    # lineaarinen approksimaatio
    def get_V_spline_fast(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,show=False,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t = self.map_grid_age(age)
        
        if age>self.max_age:
            return 0.0

        tismax=self.inv_tis(time_in_state)
        
        if emp==2:
            tismax=0
            #x = np.linspace(self.min_paid_pension,self.max_pension, self.n_elake)
            #y = self.Hila[t,0,:,emp,tismax,0]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaHila[t,:]
            f = PchipInterpolator(x,y,extrapolate=True)
            #f = interp1d(x,y,fill_value="extrapolate",kind=self.spline_approx)
            V1=f(np.maximum(self.min_oapension,elake))
        else:    
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            #p2min,p2max,wp2=self.inv_palkka(wage)        
            if self.pw_bivariate:
                V1 = np.squeeze((1-wp)*self.g[pmin,emp,tismax](elake,wage)+wp*self.g[pmax,emp,tismax](elake,wage))
            else:
                V1 = np.squeeze((1-wp)*self.g[emin,emp,tismax](old_wage,wage)+wp*self.g[emax,emp,tismax](old_wage,wage))

        if show:      
            print(f'getV({emp},{elake},{old_wage},{wage}): p2min {p2min} p2max {p2max} wp2 {wp2})')
            #print(self.Hila[t,pmin,emin,emp,tismax,p2min],self.Hila[t,pmin,emin,emp,tismax,p2max])
            
        V=np.maximum(0,V1)

        return V        
        
    def setup_spline_fast(self,t):
        if self.pw_bivariate:
            self.g=np.empty(shape=(self.n_palkka,2,self.n_tis),dtype=object)
            p = np.linspace(0, self.max_pension, self.n_elake)
            w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
            
            for pmin in range(0,self.n_palkka):
                for emp in range(0,2):
                    for tismax in range(0,self.n_tis):
                        self.g[pmin,emp,tismax]=RectBivariateSpline(p,w,self.Hila[t,pmin,:,emp,tismax,:],kx=self.pw_maxspline,ky=self.pw_maxspline)
        else:
            self.g=np.empty(shape=(self.n_elake,2,self.n_tis),dtype=object)
            p = np.linspace(0, self.max_pension, self.n_elake)
            w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
            for emp in range(0,2):
                for tismax in range(0,self.n_tis):
                    for emin in range(0,self.n_elake):
                        f=RectBivariateSpline(w,w,self.Hila[t,:,emin,emp,tismax,:],kx=self.pw_maxspline,ky=self.pw_maxspline)
                        self.g[emin,emp,tismax]=f
    

    # lineaarinen approksimaatio
    def get_V_vector_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wages=None,show=False,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        #if t>self.n_time:
        #    return 0
            
        Vs=np.zeros(wages.shape)
        
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t=self.map_grid_age(age)
        
        if emp==2:
            tismax=0
            #tismax=self.inv_tis(time_in_state)
            #x = np.linspace(self.min_paid_pension, self.max_pension, self.n_elake)
            #y = self.Hila[t,0,:,emp,tismax,0]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaHila[t,:]
            if self.monotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            if self.minspline:
                Vs[:]=np.maximum(y[0],f(elake))
            else:
                Vs[:]=np.maximum(0,f(elake))
        else:
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            tismax=self.inv_tis(time_in_state)
            p = np.linspace(0, self.max_pension, self.n_elake)
            w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)

            if self.pw_bivariate:
                values=(1-wp)*self.Hila[t,pmin,:,emp,tismax,:]+wp*self.Hila[t,pmax,:,emp,tismax,:]
                g=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                for ind,wage in enumerate(wages):
                    V1 = np.squeeze(g(elake,wage))
                    if self.minspline:
                        Vs[ind]=max(values[0,0],V1)
                    else:
                        Vs[ind]=max(0,V1)
            else:
                values=(1-we)*self.Hila[t,:,emin,emp,tismax,:]+we*self.Hila[t,:,emax,emp,tismax,:]
                g=RectBivariateSpline(w,w,values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                for ind,wage in enumerate(wages):
                    V1 = np.squeeze(g(old_wage,wage))
                    if self.minspline:
                        Vs[ind]=max(values[0,0],V1)
                    else:
                        Vs[ind]=max(0.0,V1)

        if show:      
            print(f'getV({emp},{elake},{old_wage},{wage}): p2min {p2min} p2max {p2max} wp2 {wp2})')
            print(self.Hila[t,pmin,emin,emp,tismax,p2min],self.Hila[t,pmin,emin,emp,tismax,p2max])

        return Vs

    # lineaarinen approksimaatio
    def get_V_vector_spline_fast(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wages=None,show=False,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        #if t>self.n_time:
        #    return 0
            
        Vs=np.zeros(wages.shape)
        
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t=self.map_grid_age(age)
        
        if emp==2:
            tismax=0
            #tismax=self.inv_tis(time_in_state)
            #x = np.linspace(self.min_paid_pension, self.max_pension, self.n_elake)
            #y = self.Hila[t,0,:,emp,tismax,0]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaHila[t,:]
            f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            Vs[:]=f(elake)
        else:
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            tismax=self.inv_tis(time_in_state)
            if self.pw_bivariate:
                for ind,wage in enumerate(wages):
                    V1 = np.squeeze((1-wp)*self.g[pmin,emp,tismax](elake,wage)+wp*self.g[pmax,emp,tismax](elake,wage))
                    Vs[ind]=max(0,V1)
            else:
                for ind,wage in enumerate(wages):
                    V1 = np.squeeze((1-we)*self.g[emin,emp,tismax](old_wage,wage)+we*self.g[emax,emp,tismax](old_wage,wage))
                    Vs[ind]=max(0,V1)

        if show:      
            print(f'getV({emp},{elake},{old_wage},{wage}): p2min {p2min} p2max {p2max} wp2 {wp2})')
            print(self.Hila[t,pmin,emin,emp,tismax,p2min],self.Hila[t,pmin,emin,emp,tismax,p2max])

        return Vs


    def map_grid_age(self,age):
        return int(np.round(age-self.min_grid_age))

    def plot_Hila(self,age,l=5,emp=1,time_in_state=1,diff=False):
        x=np.arange(0,100000,1000)
        q=np.zeros(x.shape)
        t=self.map_grid_age(age)    
        
        fig,ax=plt.subplots()
        if emp==2:
            if diff:
                for l in range(self.n_oapension):
                    k=0
                    elake=self.map_elake(l,emp=2)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=emp,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)-self.get_V(age=age,emp=0,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
            else:
                for l in range(self.n_oapension):
                    k=0
                    elake=self.map_elake(l,emp=emp)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=emp,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
        else:
            if diff:
                for l in range(self.n_elake):
                    k=0
                    elake=self.map_elake(l)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=1,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)-self.get_V(age=age,emp=0,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
            else:
                for l in range(self.n_elake):
                    k=0
                    elake=self.map_elake(l,emp=emp)
                    for palkka in x:
                        pa=max(self.min_wage,palkka)
                        q[k]=self.get_V(age=age,emp=emp,elake=elake,old_wage=pa,time_in_state=time_in_state,wage=pa)
                        k=k+1
            
                    plt.plot(x,q,label=elake)
            
        ax.set_xlabel('palkka')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()
                
    def plot_actHila(self,age,l=5,emp=1,time_in_state=1,diff=False,act=0,emp2=1):
        x=np.arange(0,100000,1000)
        q=np.zeros(x.shape)
        t=self.map_grid_age(age)    
        
        fig,ax=plt.subplots()
        if diff:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l,emp=emp)
                for palkka in x:
                    q[k]=self.get_actV(emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=emp2,age=age)-self.get_actV(emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=0,age=age)
                    k=k+1
            
                plt.plot(x,q,label=elake)
        else:
            for l in range(self.n_elake):
                k=0
                elake=self.map_elake(l,emp=emp)
                for palkka in x:
                    q[k]=self.get_actV(emp=emp,elake=elake,old_wage=palkka,time_in_state=time_in_state,wage=palkka,act=act,age=age)
                    k=k+1
            
                plt.plot(x,q,label=elake)
            
        ax.set_xlabel('palkka')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
        plt.show()

    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_actV_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,act=None,age=None,debug=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if s is not None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
        
        t=self.map_grid_age(age)
        
        #print(t,emp,elake,old_wage,age,time_in_state,wage)
            
        if emp==2:
            tismax=0
            #x = np.linspace(self.min_paid_pension, self.max_pension, self.n_elake)
            #y = self.actHila[t,0,:,emp,tismax,0,act]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaactHila[t,:]
            if self.monotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            if self.minspline:
                apx1=max(y[0],f(elake))
            else:
                apx1=max(0,f(elake))
        else:
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            p2min,p2max,wp2=self.inv_palkka(wage)        
            tismax=self.inv_tis(time_in_state)

            if self.pw_bivariate:
                p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                values=(1-wp)*self.actHila[t,pmin,:,emp,tismax,:,act]+wp*self.actHila[t,pmax,:,emp,tismax,:,act]
                f=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                if self.minspline:
                    apx1 = np.squeeze(np.maximum(values[0,0],f(elake,wage)))
                else:
                    apx1 = np.squeeze(f(elake,wage))
            else:
                #p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                values=(1-we)*self.actHila[t,:,emin,emp,tismax,:,act]+we*self.actHila[t,:,emax,emp,tismax,:,act]
                f=RectBivariateSpline(w,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                if self.minspline:
                    apx1 = np.squeeze(np.maximum(values[0,0],f(old_wage,wage)))
                else:
                    apx1 = np.squeeze(f(old_wage,wage))

        if debug:
            if wp2<0 or wp2>1:
                print('actV: emp {} elake {} old_wage {} wage {} tis {}: wp2 {}'.format(emp,elake,old_wage,wage,time_in_state,wp2))
            if wp<0 or wp>1:
                print('actV: emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,wp))
            if we<0 or we>1:
                print('actV: emp {} elake {} old_wage {} wage {} tis {}: wp {}'.format(emp,elake,old_wage,wage,time_in_state,we))
        
        V=max(0,apx1)
            
        act=int(np.argmax(V))
        maxV=np.max(V)

        return V
        
    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_actReward_spline(self,s=None,emp=None,elake=None,old_wage=None,time_in_state=None,wage=None,act=None,age=None):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        if emp is None:
            emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
            
        t = self.map_grid_age(age)
        
        if age>self.max_age:
            return 0.0

        if emp==2:
            tismax=0
            #tismax=self.inv_tis(time_in_state)
            #x = np.linspace(self.min_paid_pension, self.max_pension, self.n_elake)
            #y = self.actReward[t,0,:,emp,tismax,0,act]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaactReward[t,:]
            f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            R = np.squeeze(f(elake))
        else:
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            #p2min,p2max,wp2=self.inv_palkka(wage)        
            tismax=self.inv_tis(time_in_state)

            if self.pw_bivariate:
                p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                values=(1-wp)*self.actReward[t,pmin,:,emp,tismax,:,act]+wp*self.actReward[t,pmax,:,emp,tismax,:,act]
                f=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                R = np.squeeze(f(elake,wage))
            else:
                p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                values=(1-we)*self.actReward[t,:,emin,emp,tismax,:,act]+we*self.actReward[t,:,emax,emp,tismax,:,act]
                f=RectBivariateSpline(w,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                R = np.squeeze(f(old_wage,wage))
                    
        return R
    
    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_act_spline(self,s,full=False,debug=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''

        emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
        
        if emp==2:
            emin,emax,we=self.inv_elake(elake,emp=2)
            pmin,pmax,wp=0,1,0
            p2min,p2max,wp2=0,1,0
            tismax=0
            #tismax=self.inv_tis(time_in_state)
        else:
            emin,emax,we=self.inv_elake(elake)
            pmin,pmax,wp=self.inv_palkka(old_wage)
            p2min,p2max,wp2=self.inv_palkka(wage)
            tismax=self.inv_tis(time_in_state)
            
            if wage>self.max_wage:
                print(f'wage {wage}')
                
        t=self.map_grid_age(age)
        
        n_emp=self.n_acts
        
        V=np.zeros(n_emp)
        #emp_set=set([0,1,3])
        emp_set=set([0,1])
        if emp in emp_set:
            if age<self.min_retirementage-1:
                n_emp=3
                act_set=set([0,1])
                #n_emp=4
                #act_set=set([0,1,3])
            else:
                act_set=set([0,1,2])
                n_emp=3
                #act_set=set([0,1,2,3])
                #n_emp=4
        else:
            act_set=set([0])
            
        if emp == 2:
            #tismax=0
            #x = np.linspace(self.min_paid_pension, self.max_pension, self.n_elake)
            #y = self.actHila[t,0,:,emp,tismax,0,0]
            x = np.linspace(self.min_oapension,self.max_oapension, self.n_oapension)
            y = self.oaactHila[t,:]
            if self.monotone_spline:
                f = PchipInterpolator(x,y,extrapolate=True)
            else:
                f = interp1d(x, y,fill_value="extrapolate",kind=self.spline_approx)
            #V[0] = f(elake)
            if self.minspline:
                V[0] = max(y[0],f(elake))
            else:
                V[0] = f(elake)
        else:
            if self.pw_bivariate:
                p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                for k in act_set:
                    values=(1-wp)*self.actHila[t,pmin,:,emp,tismax,:,k]+wp*self.actHila[t,pmax,:,emp,tismax,:,k]
                    f=RectBivariateSpline(p,w, values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                    if self.minspline:
                        V[k] = np.squeeze(max(values[0,0],f(elake,wage)))
                    else:
                        V[k] = np.squeeze(f(elake,wage))
            else:
                #p = np.linspace(0, self.max_pension, self.n_elake)
                w = np.linspace(self.min_wage, self.max_wage, self.n_palkka)
                for k in act_set:
                    values=(1-we)*self.actHila[t,:,emin,emp,tismax,:,k]+we*self.actHila[t,:,emax,emp,tismax,:,k]
                    f=RectBivariateSpline(w,w,values,kx=self.pw_maxspline,ky=self.pw_maxspline)
                    if self.minspline:
                        V[k] = np.squeeze(max(values[0,0],f(old_wage,wage)))
                    else:
                        V[k] = np.squeeze(f(old_wage,wage))

            
        act=int(np.argmax(V))
        maxV=np.max(V)
        
        reward=self.get_actReward(s=s,act=act)
        
        if full:
            rs=[self.get_actReward(s=s,act=a) for a in act_set]
            return act,maxV,V,reward,rs
        else:
            return act,maxV,reward
            
    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_random_act(self,s,full=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        emp,elake,old_wage,age,time_in_state,wage=self.env.state_decode(s)
        emin,emax,we=self.inv_elake(elake)
        pmin,pmax,wp=self.inv_palkka(old_wage)
        p2min,p2max,wp2=self.inv_palkka(wage)
        tismax=self.inv_tis(time_in_state)
        
        t=self.map_grid_age(age)
        
        n_emp=self.n_acts
        
        V=np.zeros(n_emp)
        #emp_set=set([0,1,3])
        emp_set=set([0,1])
        if emp in emp_set:
            if age<self.min_retirementage-1:
                n_emp=3
                act_set=set([0,1])
                #n_emp=4
                #act_set=set([0,1,3])
            else:
                act_set=set([0,1,2])
                n_emp=3
                #act_set=set([0,1,2,3])
                #n_emp=4
        else:
            act_set=set([0])
            
        a_set=list(act_set)
        act_set=set(act_set)

        act=a_set[np.random.randint(len(act_set))]
        #maxV=V[act]
        
        reward=0 #self.get_actReward(s=s,act=act)
        maxV=0
        
        if full:
            return act,maxV,V,reward
        else:
            return act,maxV,reward            
        
    def get_actV_random(self,age):
        if age<self.min_retirementage-1:
            return np.random.randint(2)
        else:
            return np.random.randint(3)

    def test_salaries_v3(self,age=25,n=100,wage=20000,state=1,next_state=1,tis=0,n_future=21):
        w=np.zeros(n)

        self.n_palkka_future=n_future
        elake=10000
        s0=self.env.state_encode(state,elake,wage,age-1,tis,wage)
        
        if state==next_state:
            act=0
        else:
            act=1

        for k in range(n):
            self.env.state=s0
            newstate,r,done,info=self.env.step(act,dynprog=False)
            _,_,_,_,_,next_wage=self.env.state_decode(newstate)
            w[k]=next_wage
        
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins=200

        # plot the cumulative histogram
        n, bins, patches = ax.hist(w, n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')        
        
        z=np.zeros(bins.shape)
        q=np.zeros(bins.shape)
        for k,x in enumerate(bins):
            z[k]=self.env.wage_process_cumulative(x,wage,age,state=next_state)
            
        n_b=self.n_palkka_future
        q=self.map_palkka_future_v2(wage,age,state=next_state,midpoint=False)
        q_mid=self.map_palkka_future_v2(wage,age,state=next_state,midpoint=True)
        
        ax.plot(bins, z, 'k--', linewidth=1.5, label='Theoretical')

        for k in range(n_b):
            ax.axvline(q[k],color='r',ls='--')
            ax.axvline(q_mid[k],color='r',ls='dotted')

        med=self.env.wage_process_mean(wage,age,state=next_state)
        ax.axvline(med,color='b')

        # tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('Cumulative step histograms')
        ax.set_xlabel('Wage (e/y)')
        ax.set_ylabel('Likelihood of occurrence')

        plt.show()

    def test_salaries_v2(self,age=25,n=100,wage=20000,state=1,tis=0):
        w=np.zeros(n)

        elake=10000
        s0=self.env.state_encode(state,elake,wage,age-1,tis,wage)
        act=0

        for k in range(n):
            self.env.state=s0
            newstate,r,done,info=self.env.step(act,dynprog=False)
            _,_,_,_,_,next_wage=self.env.state_decode(newstate)
            w[k]=next_wage
        
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins=200

        # plot the cumulative histogram
        n, bins, patches = ax.hist(w, n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')        
        
        z=np.zeros(bins.shape)
        for k,x in enumerate(bins):
            z[k]=self.env.wage_process_cumulative(x,wage,age,state=state)
        
        ax.plot(bins, z, 'k--', linewidth=1.5, label='Theoretical')

        # tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('Cumulative step histograms')
        ax.set_xlabel('Wage (e/y)')
        ax.set_ylabel('Likelihood of occurrence')

        plt.show()
            
    def test_salaries(self,age=25,n=100,wage=20000,state=1):
        w=np.zeros(n)
        for k in range(n):
            w[k]=self.env.get_wage_raw(age,wage,state)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        n_bins=200

        # plot the cumulative histogram
        n, bins, patches = ax.hist(w, n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')        
        
        z=np.zeros(bins.shape)
        for k,x in enumerate(bins):
            z[k]=self.env.wage_process_cumulative(x,wage,age,state=state)
        
        ax.plot(bins, z, 'k--', linewidth=1.5, label='Theoretical')

        # tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('Cumulative step histograms')
        ax.set_xlabel('Wage (e/y)')
        ax.set_ylabel('Likelihood of occurrence')
        med=self.env.wage_process_mean(wage,age,state=state)
        ax.axvline(med,color='r')
        for k in range(7):
            ax.axvline(med*(1-0.07*0.5*k),color='r',ls='--')
            ax.axvline(med*(1+0.07*0.5*k),color='r',ls='--')
        plt.show()

    def get_dpreward(self,emp=1,elake=10000,ow=10000,ika=50,tis=1,w=10000,action=0):
        actions=[action]
        r,sps=self.get_rewards_continuous((emp,elake,ow,ika,tis,w),actions,debug=False)
        return r

    # this routine is needed for the dynamic programming
    def get_rewards_continuous(self,s,actions,debug=False):
        rewards=[]
        sps=[]
        
        start_state=self.env.state_encode(*s)
        #self.env.render(state=start_state)
        for a in actions:
            self.env.state=start_state
            newstate,reward,dones,info = self.env.step(a,dynprog=True)
            #if dones:
            #    self.reset()
            sps.append(np.array(newstate))
            rewards.append(reward)
            
        return rewards,sps
        
    def check_V(self):
        # check final age
        t=self.map_grid_age(70)
        diff=np.zeros((self.n_employment,self.n_elake,3))
        for emp in range(self.n_employment):
            for el in range(self.n_elake):
                for a in range(3):
                    diff[emp,el,a]=np.max(self.actReward[t,:,el,emp,:,:,a]-self.actHila[t,:,el,emp,:,:,a])
                
        print('max diff',np.max(diff))
        print('min diff',np.min(diff),' argmin',np.argmin(diff))
        
        return diff
                
        #for p in range(self.n_palkka): 
        #    for p_old in range(self.n_palkka): 

    def backtrack(self,age,debug=False):
        '''
        Dynaaminen ohjelmointi hilan avulla
        '''
        t=self.map_grid_age(age)
        
        if t==0:
            t0=0
        else:  
            t0=1
        
        #self.setup_spline_fast(t)
        
        def func0(x,emp2,elake2,palkka,tis2,ika2):
            a=self.get_V(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
            #a=self.get_V_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
            #a=palkka
            b=self.env.wage_process_pdf(x,palkka,ika2,state=emp2)
            return a*b

        def func1(x,emp2,elake2,palkka,tis2,ika2):
            a=self.get_V(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
            #a=self.get_V_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
            #a=palkka
            b=self.env.wage_process_pdf(x,palkka,ika2,state=emp2)
            c=a*b
            print(f'func1({x})={c}')
            return a*b
        
        # actions when active
        if age<self.min_retirementage-1:
            if self.include_pt:
                act_set=set([0,1,3])
            else:
                act_set=set([0,1])
        else:
            if self.include_pt:
                act_set=set([0,1,2,3])
            else:
                act_set=set([0,1,2])
        
        ret_set=set([0]) # actions in retirement
        #stay_set=set([0]) # stay put
        
        #print('backtrack')
        
        if debug: # and age<70:
            tulosta=True
        else:
            tulosta=False
            
        #tic=time.perf_counter()
        pn_weight=np.zeros((self.n_palkka,self.n_palkka_future,self.n_employment))
        wagetable=np.zeros(self.n_palkka)
        wagetable_future=np.zeros((self.n_palkka,self.n_palkka_future,self.n_employment))
        tr_wagetable_future=np.zeros((self.n_palkka,self.n_palkka_future+1,self.n_employment))
        ika2=age+1
        #print('age',age)
        for p in range(self.n_palkka): 
            palkka=self.map_palkka(p)
            wagetable[p]=palkka
            weight_old_s0=0
            weight_old_s1=0
            
            palkka_next_mid0_v=self.map_palkka_future_v2(palkka,ika2,state=0,midpoint=True)
            palkka_next_mid1_v=self.map_palkka_future_v2(palkka,ika2,state=1,midpoint=True)
            wagetable_future[p,:,0]=self.map_palkka_future_v2(palkka,ika2,state=0,midpoint=False)
            #wagetable_future[p,:,[1,3]]=self.map_palkka_future_v2(palkka,ika2,state=1,midpoint=False)
            wagetable_future[p,:,1]=self.map_palkka_future_v2(palkka,ika2,state=1,midpoint=False)

            #tr_wagetable_future[p,:,0]=self.map_palkka_future_v3(palkka,ika2,state=0,midpoint=False)
            #tr_wagetable_future[p,:,[1,3]]=self.map_palkka_future_v3(palkka,ika2,state=1,midpoint=False)
            
            # bugi kun age=69
            wns0=self.env.wage_process_cumulative(palkka_next_mid0_v,palkka,ika2,state=0) # tila ei saa vaikuttaa tässä kuin palkka_next_mid0:n kautta
            wns1=self.env.wage_process_cumulative(palkka_next_mid1_v,palkka,ika2,state=1) # tila ei saa vaikuttaa tässä kuin palkka_next_mid0:n kautta
            for pnext in range(self.n_palkka_future-1): 
                palkka_next_mid0=palkka_next_mid0_v[pnext]
                weight_new_s0=wns0[pnext] #self.env.wage_process_cumulative(palkka_next_mid0,palkka,ika2,state=0) # tila ei saa vaikuttaa tässä kuin palkka_next_mid0:n kautta
                pn_weight[p,pnext,0]=weight_new_s0-weight_old_s0
                weight_old_s0=weight_new_s0
                palkka_next_mid1=palkka_next_mid1_v[pnext]
                weight_new_s1=wns0[pnext] #self.env.wage_process_cumulative(palkka_next_mid1,palkka,ika2,state=1) # tila ei saa vaikuttaa tässä kuin palkka_next_mid1:n kautta
                #pn_weight[p,pnext,[1,3]]=weight_new_s1-weight_old_s1
                pn_weight[p,pnext,1]=weight_new_s1-weight_old_s1
                weight_old_s1=weight_new_s1
        
            pn_weight[p,self.n_palkka_future-1,0]=1.0-weight_old_s0
            #pn_weight[p,self.n_palkka_future-1,[1,3]]=1.0-weight_old_s1
            pn_weight[p,self.n_palkka_future-1,1]=1.0-weight_old_s1
            
            #print(wagetable_future[p,:,0])
            #print(wagetable_future[p,:,1])
            #print(pn_weight[p,:,0],1.0-np.sum(pn_weight[p,:,0]))
            #print(pn_weight[p,:,1],1.0-np.sum(pn_weight[p,:,1]))

        pn_weight[:,0,2]=1.0
        #toc=time.perf_counter()
        #print('time 1',toc-tic)
        
        for emp in range(self.n_employment):
            if emp==2:
                if age<self.min_retirementage-1:
                    #self.Hila[t,:,:,emp,:,:]=0
                    #self.actHila[t,:,:,emp,:,:]=0
                    #self.actReward[t,:,:,emp,:,:]=0
                    self.oaHila[t,:]=0
                    self.oaactHila[t,:]=0
                    self.oaactReward[t,:]=0
                else:
                    #tic=time.perf_counter()

                    time_in_state=self.map_tis(0)
                    for el in range(self.n_oapension):
                        elake=self.map_elake(el,emp=2)

                        # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                        rts,Sps=self.get_rewards_continuous((emp,elake,0,age,time_in_state,0),ret_set)
                        
                        for ind,a in enumerate(ret_set):
                            emp2,elake2,_,ika2,_,_=self.env.state_decode(Sps[ind])
                            #self.actHila[t,:,el,emp,:,:,a]=rts[ind]+self.gamma*self.get_V(emp=emp2,elake=elake2,old_wage=self.min_wage,wage=self.min_wage,time_in_state=0,age=ika2)
                            #self.actReward[t,:,el,emp,:,:,a]=rts[ind]
                            self.oaactHila[t,el]=rts[ind]+self.gamma*self.get_V(emp=emp2,elake=elake2,old_wage=self.min_wage,wage=self.min_wage,time_in_state=0,age=ika2)
                            #self.oaactHila[t,el]=rts[ind]+self.gamma*self.get_V_spline_fast(emp=emp2,elake=elake2,old_wage=self.min_wage,wage=self.min_wage,time_in_state=0,age=ika2)
                            self.oaactReward[t,el]=rts[ind]
                            #print('getV(emp{} e{} p{}): {}'.format(emp2,elake2,palkka,gw))
                            #print(f'rts[{ind}] {rts[ind]}')
                            
                        #self.Hila[t,:,el,emp,:,:]=self.actHila[t,0,el,emp,0,0,0]
                        self.oaHila[t,el]=self.oaactHila[t,el]
                    #toc=time.perf_counter()
                    #print('time state 2',toc-tic)

            elif emp==1:
                #tic=time.perf_counter()
                time_in_state=self.map_tis(0)
                for el in range(self.n_elake):
                    elake=self.map_elake(el)
                    for p_old in range(self.n_palkka): 
                        palkka_vanha=wagetable[p_old]
                        for p in range(self.n_palkka): 
                            palkka=wagetable[p]
                            # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                            rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                
                            for ind,a in enumerate(act_set):
                                emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                if emp2==2:
                                    q=self.get_V(emp=emp2,elake=elake2,old_wage=0,time_in_state=0,age=ika2,wage=0)
                                    #w1=self.get_V_spline_fast(emp=emp2,elake=elake2,old_wage=0,time_in_state=0,age=ika2,wage=0)
                                else:
                                    if self.integmethod==0:
                                        gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                        #gw=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                        w=pn_weight[p,:,emp2]
                                        d=np.sum(gw*w)
                                    elif self.integmethod==1:
                                        d,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                    elif self.integmethod==2:
                                        gw2=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                        #gw2=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                        b=self.env.wage_process_pdf(tr_wagetable_future[p,:,emp2],palkka,ika2,state=emp2)
                                        d=trapz(gw2*b,x=tr_wagetable_future[p,:,emp2])
                                    else: # compare
                                        d1,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                        gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                        #gw=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                        w=pn_weight[p,:,emp2]
                                        d2=np.sum(gw*w) 
                                        delta=d2-d1
                                        d=d1
                                        #if age==69 and palkka==1000.0:
                                        #    print(wagetable_future[p,:,emp])
                                            
                                        if np.abs(delta)>0.0001:
                                            print(f'em1={emp},emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{d2:.5f} vs {d1:.5f} {delta}')
                                            d1,_=quad(func1, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))

                                    q=rts[ind]+self.gamma*d
                                    
                                    #delta=w1-d
                                    #print(f'emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{w1:.5f} vs {d:.5f} {delta}')
                                    
                                    #if np.abs(delta)>0.01:
                                    #    plt.plot(tr_wagetable_future[p,:,emp2],b)
                                    #    exit()
                                
                                self.actHila[t,p_old,el,emp,:,p,a]=q
                                self.actReward[t,p_old,el,emp,:,p,a]=rts[ind]
                                
                            self.Hila[t,p_old,el,emp,:,p]=np.max(self.actHila[t,p_old,el,emp,0,p,:])
                #toc=time.perf_counter()
                #print('time state 1',toc-tic)

            elif emp==3:
                #tic=time.perf_counter()
                time_in_state=self.map_tis(0)
                for el in range(self.n_elake):
                    elake=self.map_elake(el)
                    for p_old in range(self.n_palkka): 
                        palkka_vanha=wagetable[p_old]
                        for p in range(self.n_palkka): 
                            palkka=wagetable[p]
                            # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                            rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                            #print('(emp{} e{} p_old{} p{} ika{})'.format(emp,elake,palkka_vanha,palkka,age))
                            
                            for ind,a in enumerate(act_set):
                                emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                if self.integmethod==0:
                                    gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                    #gw=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                    w=pn_weight[p,:,emp2]
                                    d=np.sum(gw*w)
                                elif self.integmethod==1:
                                    d,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                elif self.integmethod==2:
                                    gw2=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                    #gw2=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                    b=self.env.wage_process_pdf(tr_wagetable_future[p,:,emp2],palkka,ika2,state=emp2)
                                    d=trapz(gw2*b,x=tr_wagetable_future[p,:,emp2])
                                else: # compare
                                    d1,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                    gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                    #gw=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                    w=pn_weight[p,:,emp2]
                                    d2=np.sum(gw*w) 
                                    delta=d2-d1
                                    d=d1
                                    #if age==69 and palkka==1000.0:
                                    #    print(wagetable_future[p,:,emp])
                                        
                                    if np.abs(delta)>0.0001:
                                        print(f'em1={emp},emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{d2:.5f} vs {d1:.5f} {delta}')
                                        d1,_=quad(func1, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                
#                                 s=0.07
#                                 def func3(x):
#                                     a=self.get_V(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wage=x)
#                                     b=self.env.wage_process_pdf(x,palkka,ika2,state=emp2)
#                                     return a*b
#                                 
#                                 w0,_=quad(func3, 0.0, 100_000.0)
#                                 q=rts[ind]+self.gamma*w0
                                
                                
                                #if tulosta:
                                #    print('s{}: getV(emp{} oe{:.1f} e{:.1f} ow{:.1f} p{:.1f}): {} (R={})'.format(emp,emp2,elake,elake2,palkka_vanha,palkka,q,rts[ind]))
                                self.actHila[t,p_old,el,emp,:,p,a]=q
                                self.actReward[t,p_old,el,emp,:,p,a]=rts[ind]
                                
                            self.Hila[t,p_old,el,emp,:,p]=np.max(self.actHila[t,p_old,el,emp,0,p,:])
                #toc=time.perf_counter()
                #print('time state 3',toc-tic)
            elif emp==0:
                #tic=time.perf_counter()
                for p_old in range(self.n_palkka):
                    palkka_vanha=wagetable[p_old]
                    for el in range(self.n_elake):
                        elake=self.map_elake(el)
                        for tis in range(t0,self.n_tis):
                            time_in_state=self.map_tis(tis)
                            for p in range(self.n_palkka): 
                                palkka=wagetable[p]
                                # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                                rts,Sps=self.get_rewards_continuous((emp,elake,palkka_vanha,age,time_in_state,palkka),act_set)
                    
                                for ind,a in enumerate(act_set):
                                    emp2,elake2,_,ika2,tis2,_=self.env.state_decode(Sps[ind])
                                    if emp2==2:
                                        q=self.get_V(emp=emp2,elake=elake2,old_wage=0,time_in_state=0,age=ika2,wage=0)
                                    else:
                                        if self.integmethod==0:
                                            gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                            #gw=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                            #gw=palkka+wagetable_future[p,:,emp2]*0
                                            w=pn_weight[p,:,emp2]
                                            d=np.sum(gw*w) 
                                        elif self.integmethod==1:
                                            d,_=quad(func0, 1000.0, 130_000.0,epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                        elif self.integmethod==2:
                                            #gw2=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                            gw2=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=tr_wagetable_future[p,:,emp2])
                                            b=self.env.wage_process_pdf(tr_wagetable_future[p,:,emp2],palkka,ika2,state=emp2)
                                            d=trapz(gw2*b,x=tr_wagetable_future[p,:,emp2])
                                        else: # compare
                                            d1,_=quad(func0, 1000.0, min(10*palkka,130_000.0),epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))
                                            gw=self.get_V_vector(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                            #gw=self.get_V_vector_spline_fast(emp=emp2,elake=elake2,old_wage=palkka,time_in_state=tis2,age=ika2,wages=wagetable_future[p,:,emp2])
                                            #gw=palkka+wagetable_future[p,:,emp2]*0
                                            w=pn_weight[p,:,emp2]
                                            d2=np.sum(gw*w)
                                            delta=d2-d1
                                            d=d1
                                            if np.abs(delta)>0.0001:
                                                print(f'em1={emp},emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{d2:.5f} vs {d1:.5f} {delta}')
                                                d1,_=quad(func1, 1000.0,min(10*palkka,130_000.0),epsabs=self.epsabs,args=(emp2,elake2,palkka,tis2,ika2))

                                        q=rts[ind]+self.gamma*d
                                        
                                        #delta=w1-d
                                        #print(f'emp={emp2},elake={elake2:.2f},old_wage={palkka},time_in_state={tis2},age={ika2},{w1:.5f} vs {d:.5f} {delta}')

                                    self.actHila[t,p_old,el,emp,tis,p,a]=q
                                    self.actReward[t,p_old,el,emp,tis,p,a]=rts[ind]
                                    
                                self.Hila[t,p_old,el,emp,tis,p]=np.max(self.actHila[t,p_old,el,emp,tis,p,:])
                #toc=time.perf_counter()
                #print('time state 0',toc-tic)
            else:
                print('unknown state ',emp)

        if debug:
            self.print_actV(age)
            self.print_V(age)
            #print('at age {} mean V {} mean r {}'.format(age,np.mean(self.Hila[t,:,:,:,:]),qr/m),qw/m)

                                
    def train(self,debug=False,save='best/dynamic_prog_V.h5'):
        '''
        Lasketaan optimaalinen työllistyminen/työttömyys/eläköityminen valitulla valintametodilla
        '''
        self.init_grid()
        print('Optimizing behavior')
        tqdm_e = tqdm(range(int(self.n_time-1)), desc='Score', leave=True, unit=" year")

        for age in range(self.max_age,self.min_age-1,-1):
            t=self.map_grid_age(age)    
            self.backtrack(age,debug=debug)
            tqdm_e.set_description("Year " + str(t))
            tqdm_e.update(1)

        self.save_V(save)

    def simulate(self,debug=False,pop=1_000,save=None,load=None,ini_pension=None,ini_wage=None,ini_age=None,random_act=False):
        '''
        Lasketaan työllisyysasteet ikäluokittain
        '''
        if pop is not None:
            self.n_pop=pop

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year,dynprog=True)
        if load is not None:
            self.load_V(load)        
            
        print('simulate debug',debug)

        tqdm_e = tqdm(range(int(pop)), desc='Population', leave=True, unit=" p")
        
        rewards_pred=np.zeros((pop,self.n_time))
        rewards=np.zeros((pop,self.n_time))

        for n in range(pop):
            state=self.env.reset(pension=ini_pension,ini_wage=ini_wage,ini_age=ini_age)
            
            for t in range(self.n_time+1):
                
                if random_act:
                    act,maxV,rewards_pred[n,t]=self.get_random_act(state)
                else:
                    if debug:
                        act,maxV,v,rewards_pred[n,t],rs=self.get_act(state,full=True)
                    else:
                        act,maxV,rewards_pred[n,t]=self.get_act(state)

                #if debug:
                #    self.env.render(state=state,pred_r=rewards_pred[n,t])
                    #print(v,rs)
                    
                newstate,r,done,info=self.env.step(act,dynprog=False)

                if debug:
                    self.env.render(state=newstate,reward=r,pred_r=rewards_pred[n,t])
                
                rewards[n,t]=r
                 
                if done: 
                    self.episodestats.add(n,act,r,state,newstate,info,debug=debug,aveV=maxV)
                    #print(info,r)
                    #print(newstate,info[0]['terminal_observation'])
                    tqdm_e.update(1)
                    tqdm_e.set_description("Pop " + str(n))
                    break
                else:
                    self.episodestats.add(n,act,r,state,newstate,info,debug=debug,aveV=maxV)
                    
                state=newstate
        
        coef=1-np.var(rewards-rewards_pred)/np.var(rewards)
        print('Explained variance ',coef)
        print('Pred variance {} variance {} diff variance {}'.format(np.var(rewards_pred),np.var(rewards),np.var(rewards-rewards_pred)))
        absmax=np.abs(rewards-rewards_pred)
        print('Max diff in r {} in {}'.format(np.max(absmax),np.argmax(absmax)))
        
        #for n in range(pop):
        #    coef=(1-np.var(rewards[n,:-2]-rewards_pred[n,:-2]))/np.var(rewards[n,:-2])
        #    print(F'{n}: {coef}')

        
        if save is not None:
            self.episodestats.save_sim(save)
          
#     def simulate_det(self,debug=False,pop=1_000,save=None,load=None,ini_pension=None,ini_wage=None,ini_age=None,ini_old_wage=None):
#         '''
#         Lasketaan työllisyysasteet ikäluokittain
#         '''
#         if pop is not None:
#             self.n_pop=pop
# 
#         self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
#                                 self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)
#         if load is not None:
#             self.load_V(load)        
# 
#         tqdm_e = tqdm(range(int(pop)), desc='Population', leave=True, unit=" p")
#         
#         rewards_pred=np.zeros((pop,self.n_time))
#         rewards=np.zeros((pop,self.n_time))
# 
#         for n in range(pop):
#             state=self.env.reset(pension=ini_pension,ini_wage=ini_wage,ini_age=ini_age,ini_old_wage=ini_old_wage)
#             
#             for t in range(self.n_time):
#                 if debug:
#                     act,maxV,v,rewards_pred[n,t]=self.get_act(state,full=True)
#                 else:
#                     act,maxV,rewards_pred[n,t]=self.get_act(state)
#                 
#                 newstate,r,done,info=self.env.step(act,dynprog=False)
# 
#                 if debug:
#                     self.env.render(state=state,reward=r,pred_r=rewards_pred[n,t])
#                     print(v)                    
# 
#                 rewards[n,t]=r
#                  
#                 if done: 
#                     self.episodestats.add(n,act,r,state,newstate,info,debug=debug,aveV=maxV)
#                     tqdm_e.update(1)
#                     tqdm_e.set_description("Pop " + str(n))
#                     break
#                 else:
#                     self.episodestats.add(n,act,r,state,newstate,info,debug=debug,aveV=maxV)
#                     
#                 state=newstate
#         
#         coef=1-np.var(rewards-rewards_pred)/np.var(rewards)
#         print('Explained variance ',coef)
#         print('Pred variance {} variance {} diff variance {}'.format(np.var(rewards_pred),np.var(rewards),np.var(rewards-rewards_pred)))
#         
#         #for n in range(pop):
#         #    coef=(1-np.var(rewards[n,:-2]-rewards_pred[n,:-2]))/np.var(rewards[n,:-2])
#         #    print(F'{n}: {coef}')
# 
#         
#         if save is not None:
#             self.episodestats.save_sim(save)
          
    def plot_statsV(self,aveV):
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        #x=range(self.min_age,self.min_age+self.n_time)
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('Keskiarvo V')
        mV=np.mean(aveV,axis=1)
        ax.plot(x,mV)
        plt.show()
        fig,ax=plt.subplots()
        ax.set_xlabel('Ikä')
        ax.set_ylabel('palkka')
        ax.plot(x,mV[:,0:10])
        plt.show()                      
        
    def save_V(self,filename):
        f = h5py.File(filename, 'w')
        ft='f16'
        dset = f.create_dataset("Hila", data=self.Hila, dtype=ft)
        f.create_dataset("min_grid_age", data=self.min_grid_age, dtype=ft)
        f.create_dataset("max_grid_age", data=self.max_grid_age, dtype=ft)
        f.create_dataset("max_wage", data=self.max_wage, dtype=ft)
        f.create_dataset("min_wage", data=self.min_wage, dtype=ft)
        f.create_dataset("max_pension", data=self.max_pension, dtype=ft)
        f.create_dataset("min_paid_pension", data=self.min_paid_pension, dtype=ft)
        f.create_dataset("actHila", data=self.actHila, dtype=ft)
        f.create_dataset("actReward", data=self.actReward, dtype=ft)
        #f.create_dataset("hila_palkka0", data=self.hila_palkka0, dtype=ft)
        f.create_dataset("n_palkka", data=self.n_palkka, dtype='i4')
        f.create_dataset("deltapalkka", data=self.deltapalkka, dtype=ft)
        f.create_dataset("n_elake", data=self.n_elake, dtype='i4')
        f.create_dataset("n_oapension", data=self.n_oapension, dtype='i4')
        f.create_dataset("deltaelake", data=self.deltaelake, dtype=ft)
        f.create_dataset("delta_paid_pension", data=self.delta_paid_pension, dtype=ft)
        f.create_dataset("delta_oapension", data=self.delta_oapension, dtype=ft)
        f.create_dataset("n_tis", data=self.n_tis, dtype='i4')
        f.create_dataset("deltatis", data=self.deltatis, dtype=ft)
        f.create_dataset("oaHila", data=self.oaHila, dtype=ft)
        f.create_dataset("oaactHila", data=self.oaactHila, dtype=ft)
        f.create_dataset("oaactReward", data=self.oaactReward, dtype=ft)
        f.create_dataset("min_oapension", data=self.min_oapension, dtype=ft)
        f.create_dataset("max_oapension", data=self.max_oapension, dtype=ft)
        
        f.close()
        
    def load_V(self,filename):
        f = h5py.File(filename, 'r')
        self.Hila = f['Hila'][()]
        self.min_grid_age = f['min_grid_age'][()]
        self.max_grid_age = f['max_grid_age'][()]
        self.actReward = f['actReward'][()]
        self.actHila = f['actHila'][()]
        self.n_palkka = f['n_palkka'][()]
        self.deltapalkka = f['deltapalkka'][()]
        self.n_elake = f['n_elake'][()]
        self.deltaelake = f['deltaelake'][()]
        self.n_tis = f['n_tis'][()]
        self.deltatis = f['deltatis'][()]
        
        if 'max_pension' in f.keys():
            self.max_pension=f['max_pension'][()]
            self.max_wage=f['max_wage'][()]
            
        if 'min_wage' in f.keys():
            self.min_wage=f['min_wage'][()]
        else:
            if 'hila_palkka0' in f.keys():
                self.min_wage=f['hila_palkka0'][()]

        if 'min_paid_pension' in f.keys():
            self.delta_paid_pension = f['delta_paid_pension'][()]
            self.min_paid_pension = f['min_paid_pension'][()]

        if 'n_oapension' in f.keys():
            self.n_oapension = f['n_oapension'][()]
            self.oaHila = f['oaHila'][()]
            self.oaactHila = f['oaactHila'][()]
            self.oaactReward = f['oaactReward'][()]
            self.delta_oapension = f['delta_oapension'][()]
            self.min_oapension = f['min_oapension'][()]
            self.max_oapension = f['max_oapension'][()]
            
        f.close()
        
    def plot_act(self):
        act_ratio=np.sum(self.actions==1,axis=1)/2000
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,act_ratio)    
        ax.set_xlabel('Aika')
        ax.set_ylabel('Tilasta pois-siirtyneet')
        plt.show()

    def plot_higher(self,t):
        emp=1
        prev=0
        q=np.array(self.actHila[t,:,:,emp,prev,0]>self.actHila[t,:,:,emp,prev,1]).astype(int)
        self.plot_img(q,xlabel="Pension",ylabel="Wage",title='Employed, stay in state')
        emp=0
        q=np.array(self.actHila[t,:,:,emp,prev,0]>self.actHila[t,:,:,emp,prev,1]).astype(int)
        self.plot_img(q,xlabel="Pension",ylabel="Wage",title='Unemployed, stay in state')

    def print_V(self,age):
        t=self.map_grid_age(age)    
        print('t=',t,'age=',age)
        print('töissä\n',self.get_diag_V(t,1))
        print('ei töissä\n',self.get_diag_V(t,0))
        print('eläke\n',self.get_diag_V(t,2))
        #print('osatyö\n',self.get_diag_V(t,3))

    def get_diag_V(self,t,emp,tis=1):
        if emp != 2:
            sh=self.Hila.shape
            h=np.zeros((sh[1],sh[2]))
            for k in range(sh[1]):
                for l in range(sh[2]):
                    h[k,l]=self.Hila[t,k,l,emp,tis,k]
        else:
            sh=self.oaHila.shape
            h=np.zeros((sh[1]))
            for k in range(sh[1]):
                h[k]=self.oaHila[t,k]
                        
        return h
    
    def get_diag_actV(self,t,emp,act,time_in_state=1):
        if emp != 2:
            sh=self.actHila.shape
            h=np.zeros((sh[1],sh[2]))
            for k in range(sh[1]):
                for l in range(sh[2]):
                    h[k,l]=self.actHila[t,k,l,emp,time_in_state,k,act]
        else:
            sh=self.oaactHila.shape
            h=np.zeros((sh[1]))
            for k in range(sh[1]):
                h[k]=self.oaactHila[t,k]
                        
        return h
    
    def plot_V(self,age):
        t=self.map_grid_age(age)
        self.plot_img(self.get_diag_V(t,1),xlabel="Pension",ylabel="Wage",title='Töissä')
        self.plot_img(self.get_diag_V(t,0),xlabel="Pension",ylabel="Wage",title='Työttömänä')
        self.plot_img(self.get_diag_V(t,1)-self.get_diag_V(t,0),xlabel="Pension",ylabel="Wage",title='Työssä-Työtön')

    def print_actV(self,age,time_in_state=1):
        t=self.map_grid_age(age)
        print('t={} age={}'.format(t,age))
        if age>=self.min_retirementage:
            print('eläke: pysyy\n{}\n'.format(self.get_diag_actV(t,2,0,time_in_state=time_in_state)))
            print('töissä: pois töistä\n{}\ntöissä: pysyy\n{}\ntöissä: eläköityy\n{}\n'.format(self.get_diag_actV(t,1,1,time_in_state=time_in_state),self.get_diag_actV(t,1,0,time_in_state=time_in_state),self.get_diag_actV(t,1,2,time_in_state=time_in_state)))
            print('ei töissä: töihin\n{}\nei töissä: pysyy\n{}\nei töissä: eläköityy\n{}\n'.format(self.get_diag_actV(t,0,1,time_in_state=time_in_state),self.get_diag_actV(t,0,0,time_in_state=time_in_state),self.get_diag_actV(t,0,2,time_in_state=time_in_state)))
        else:
            print('töissä: pois töistä\n',self.get_diag_actV(t,1,1),'\ntöissä: pysyy\n',self.get_diag_actV(t,1,0,time_in_state=time_in_state))
            #print('töissä: osatöihin\n',self.get_diag_actV(t,1,3),'\nei töissä: osatöihin\n',self.get_diag_actV(t,1,3,time_in_state=time_in_state))
            print('ei töissä: töihin\n',self.get_diag_actV(t,0,1),'\nei töissä: pysyy\n',self.get_diag_actV(t,0,0,time_in_state=time_in_state))
            #print('osatöissä: pysyy\n',self.get_diag_actV(t,3,0,time_in_state=time_in_state))
            #print('osatöissä: töihin\n',self.get_diag_actV(t,3,3),'\nosatöissä: työttömäksi\n',self.get_diag_actV(t,3,1,time_in_state=time_in_state))

    def print_act(self,age,time_in_state=0):
        print('age=',age)
        if age>=self.min_retirementage:
            print('eläke (act)\n')
            display(self.get_act_q(age,2,time_in_state=time_in_state))

        print('töissä (act)\n')
        display(self.get_act_q(age,1,time_in_state=time_in_state))
        print('ei töissä (act)\n')
        display(self.get_act_q(age,0,time_in_state=time_in_state))

    def plot_actV_diff(self,age):
        t=self.map_grid_age(age)
        self.plot_img(self.get_diag_actV(t,1,1)-self.get_diag_actV(t,1,0),xlabel="Pension",ylabel="Wage",title='Töissä (ero switch-stay)')
        self.plot_img(self.get_diag_actV(t,0,1)-self.get_diag_actV(t,0,0),xlabel="Pension",ylabel="Wage",title='Työttömänä (ero switch-stay)')

    def plot_act(self,age,time_in_state=0):
        #q3=self.get_act_q(age,emp=3,time_in_state=time_in_state)
        q1=self.get_act_q(age,emp=1,time_in_state=time_in_state)
        q2=self.get_act_q(age,emp=0,time_in_state=time_in_state)

        self.plot_img(q1,xlabel="Pension",ylabel="Wage",title='Töissä',vmin=0,vmax=3)
        self.plot_img(q2,xlabel="Pension",ylabel="Wage",title='Työttömänä',vmin=0,vmax=3)
        ##self.plot_img(q3,xlabel="Pension",ylabel="Wage",title='Osatyössä',vmin=0,vmax=3)
    
    def get_act_q(self,age,emp=1,time_in_state=0,debug=False):
        t=self.map_grid_age(age)
        q=np.zeros((self.n_palkka,self.n_elake))
        for p in range(self.n_palkka): 
            for el in range(self.n_elake):
                palkka=self.map_palkka(p)
                elake=self.map_elake(el,emp=emp)

                q[p,el]=int(np.argmax(self.actHila[t,p,el,emp,time_in_state,p,:]))
                if debug:
                    print('p {} e {} a {} q {}'.format(palkka,elake,q[p,el],self.actHila[t,p,el,emp,time_in_state,p,:]))
        return q

    def print_q(self,age,emp=1,time_in_state=0):
        _=self.get_act_q(age=age,emp=emp,time_in_state=time_in_state,debug=True)

    def compare_act(self,age,cc,time_in_state=0,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                    deterministic=True,vmin=None,vmax=None,dire='kuvat',show_results=False):
        q1=self.get_act_q(age,emp=0,time_in_state=time_in_state)
        q2=cc.get_RL_act(age,emp=0,time_in_state=time_in_state,rlmodel=rlmodel,
            load=load,deterministic=deterministic,
            n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
            min_wage=self.min_wage,min_pension=self.min_pension)
        q3=self.get_act_q(age,emp=1,time_in_state=time_in_state)
        q4=cc.get_RL_act(age,emp=1,time_in_state=time_in_state,rlmodel=rlmodel,
            load=load,deterministic=deterministic,
            n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
            min_wage=self.min_wage,min_pension=self.min_pension)
        #q5=self.get_act_q(age,emp=3,time_in_state=time_in_state)
        #q6=cc.get_RL_act(age,emp=3,time_in_state=time_in_state,rlmodel=rlmodel,
        #    load=load,deterministic=deterministic,
        #    n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
        #    min_wage=self.min_wage,min_pension=self.min_pension)
        
        self.plot_twoimg(q3,q4,title1='Employed DP {}'.format(age),title2='Employed RL {}'.format(age),vmin=0,vmax=2,figname=dire+'/emp_'+str(age))
        self.plot_twoimg(q1,q2,title1='Unemployed DP {}'.format(age),title2='Unemployed RL {}'.format(age),vmin=0,vmax=2,figname=dire+'/unemp_'+str(age))
        #self.plot_twoimg(q5,q6,title1='Parttime DP {}'.format(age),title2='Parttime RL {}'.format(age),vmin=0,vmax=2,figname=dire+'/parttime_'+str(age))
        
    def compare_ages(self,cc,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                     deterministic=True,time_in_state=0):
        for age in set([20,25,30,35,40,45,50,55,59,60,61,62,63,64,65,66,67,68,69,70]):
            self.compare_act(age,cc,rlmodel=rlmodel,load=load,deterministic=deterministic,time_in_state=time_in_state)
            
    def compare_age_and_real(self,cc,rlmodel='small_acktr',load='saved/malli_perusmini99_nondet',
                     deterministic=True,time_in_state=0,age=50,dire='kuvat',results=None,figname=None,emp1=0,emp2=1):
        self.load_sim(results)
        q1=cc.get_RL_act(age,emp=emp1,time_in_state=time_in_state,rlmodel=rlmodel,
           load=load,deterministic=deterministic,n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
           min_wage=self.min_wage,min_pension=self.min_pension)
        q2=cc.get_RL_act(age,emp=emp2,time_in_state=time_in_state,rlmodel=rlmodel,
           load=load,deterministic=deterministic,n_palkka=self.n_palkka,deltapalkka=self.deltapalkka,n_elake=self.n_elake,deltaelake=self.deltaelake,
           min_wage=self.min_wage,min_pension=self.min_pension)
        fig,axs=self.plot_twoimg(q1,q2,title1='Unemployed RL {}'.format(age),title2='Employed RL {}'.format(age),vmin=0,vmax=2,
           show_results=False,alpha=0.5)
        print('scatter...')
           
        if emp1 != 2:
            c1='w'
        else:
            c1='w'
        if emp2 != 2:
            c2='w'
        else:
            c2='w'
        
        t=self.map_age(age)
        xa=[]
        ya=[]
        xb=[]
        yb=[]
        for k in range(self.episodestats.n_pop):
            x0,x1,dx=self.inv_elake(self.episodestats.infostats_pop_pension[t,k])
            y0,y1,dy=self.inv_palkka(self.episodestats.infostats_pop_wage[t,k])

            y2=min(self.n_palkka,y0+dy)

            if self.episodestats.popempstate[t,k]==0:
                xa.append(x0+dx)
                ya.append(y2)
                #axs[0].scatter(x0+dx,y0+dy,marker='.',s=2,c=c1)
            elif self.episodestats.popempstate[t,k]==1:
                xb.append(x0+dx)
                yb.append(y2)
                #axs[1].scatter(x0+dx,y0+dy,marker='.',s=2,c=c2)

        axs[0].scatter(xa,ya,marker='.',s=2,c=c1)
        axs[1].scatter(xb,yb,marker='.',s=2,c=c2)
        
        if figname is not None:
            plt.savefig(figname+'.eps', format='eps')

        fig.show()
                     
    def comp_pave(self):
    
        self.ave=np.zeros(self.n_time)
        self.sted=np.zeros(self.n_time)
        self.real=self.episodestats.comp_presentvalue()
        for t in range(1,self.n_time):
            self.ave[t]=np.mean(self.episodestats.aveV[t,:]-self.real[t,:])
            self.sted[t]=np.std(self.episodestats.aveV[t,:]-self.real[t,:])
            
        self.pave=np.zeros((self.n_time,3))
        self.nave=np.zeros((self.n_time,3))
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                s=int(self.episodestats.popempstate[t,k])
                self.pave[t,s]+=self.episodestats.aveV[t,k]-self.real[t,k]
                self.nave[t,s]+=1            

        self.tave=np.zeros((self.n_time,10))
        self.ntave=np.zeros((self.n_time,10))
        self.pvave=np.zeros((self.n_time,5))
        self.npvave=np.zeros((self.n_time,5))
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                wa=int(np.floor(self.episodestats.infostats_pop_wage[t,k]/10000))
                self.tave[t,wa]+=self.episodestats.aveV[t,k]-self.real[t,k]
                self.ntave[t,wa]+=1
                pa=int(np.floor(self.episodestats.infostats_pop_pension[t,k]/10000))
                self.pvave[t,pa]+=self.episodestats.aveV[t,k]-self.real[t,k]
                self.npvave[t,pa]+=1
        
        self.pave=self.pave/np.maximum(1.0,self.nave)
        self.pvave=self.pvave/np.maximum(1.0,self.npvave)
        self.tave=self.tave/np.maximum(1.0,self.ntave)

        self.oavave=np.zeros((self.n_time,5))
        self.n_oavave=np.zeros((self.n_time,5))
        for k in range(self.n_pop):
            for t in range(1,self.n_time):
                if int(self.episodestats.popempstate[t,k])==2:
                    wa=int(np.floor(self.episodestats.infostats_pop_pension[t,k]/10000))
                    self.oavave[t,wa]+=self.episodestats.aveV[t,k]-self.real[t,k]
                    self.n_oavave[t,wa]+=1
        self.oavave=self.oavave/np.maximum(1.0,self.n_oavave)

    def plot_pave(self):
        plt.plot(self.ave[1:])
        plt.show()
        
        plt.plot(self.pave[:,0],'r',label='unemp')
        plt.plot(self.pave[:,1],'k',label='emp')
        plt.legend()
        plt.title('emp/unemp aveV error')
        plt.show()
        
        plt.plot(self.pave[:,2],'k',label='ret')     
        plt.title('ret aveV error')
        plt.show()
        
        for k in range(5):
            plt.plot(self.tave[:,k],label=str(k))
        plt.legend()
        plt.title('aveV error by wage level')
        plt.show()        
        for k in range(5,10):
            plt.plot(self.tave[:,k],label=str(k))
        plt.legend()
        plt.title('aveV error by wage level')
        plt.show()        
        for k in range(10):
            plt.plot(self.ntave[:,k],label=str(k))
        plt.legend()
        plt.title('number of agents by wage level')
        plt.show()        

        for k in range(5):
            plt.plot(self.oavave[45:,k],label=str(k))
        plt.legend()
        plt.title('error by retirees by wage level')
        plt.show()        
        for k in range(5):
            plt.plot(self.n_oavave[45:,k],label=str(k))
        plt.legend()
        plt.title('number of retirees by wage level')
        plt.show()
        
        for k in range(5):
            plt.plot(self.pvave[:,k],label=str(k))
        plt.legend()
        plt.title('aveV error by pension level')
        plt.show()        
        for k in range(5):
            plt.plot(self.npvave[:,k],label=str(k))
        plt.legend()
        plt.title('number of agents by pension level')
        plt.show()        

    def plot_p_aveV(self,num=500,start=1):
        plt.plot(self.episodestats.aveV[start:,num],'r')
        plt.plot(self.real[start:,num])
        plt.show()
        plt.plot(self.episodestats.aveV[start:,num]-self.real[start:,num])
        plt.show()
        plt.plot(self.episodestats.popempstate[start:,num])                                