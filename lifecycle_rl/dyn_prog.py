'''

    dyn_prog.py
    
    implements a scheme similar to solving valuation of american options for the life cycle model
    despite the name, it is questionable 
    

'''

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import Benefits
import gym_unemployment
#import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm_notebook as tqdm
from lifecycle_rl import Lifecycle
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines import A2C, ACER, DQN, ACKTR #, TRPO



class DynProgLifecycle(Lifecycle):

    def __init__(self,minimal=False,env=None,timestep=1.0,ansiopvraha_kesto300=None,\
                    ansiopvraha_kesto400=None,karenssi_kesto=None,\
                    ansiopvraha_toe=None,perustulo=None,mortality=None):

        super().__init__(minimal=minimal,env=env,timestep=timestep,ansiopvraha_kesto300=ansiopvraha_kesto300,\
                    ansiopvraha_kesto400=ansiopvraha_kesto400,karenssi_kesto=karenssi_kesto,\
                    ansiopvraha_toe=ansiopvraha_toe,perustulo=perustulo,mortality=mortality)
        
        '''
        Alusta muuttujat
        '''
        self.hila_palkka0 = 0
        self.hila_elake0 = 0
        
        # dynaamisen ohjelmoinnin parametrejä
        self.n_palkka = 10
        self.deltapalkka = 5000
        self.n_elake = 10
        self.deltaelake = 2500
        self.n_tis = 5
        self.deltatis = 1

    def init_grid(self):
        self.Hila = np.zeros((self.n_time+2,self.n_palkka,self.n_elake,self.n_employment,self.n_tis))
        self.actHila = np.zeros((self.n_time+2,self.n_palkka,self.n_elake,self.n_employment,self.n_tis,self.n_acts))        

    def map_elake(self,v):
        return self.hila_elake0+self.deltaelake*v # pitäisikö käyttää exp-hilaa?

    def inv_elake(self,v):
        vmin=max(0,min(self.n_elake-2,int((v-self.hila_elake0)/self.deltaelake)))
        vmax=vmin+1
        w=(v-self.hila_elake0)/self.deltaelake-vmin # lin.approximaatio

        return vmin,vmax,w

    def map_exp_elake(self,v):
        return self.hila_elake0+self.deltaelake*(np.exp(v*self.expelakescale)-1)

    def inv_exp_elake(self,v):
        vmin=max(0,min(self.n_elake-2,int((np.log(v-self.hila_elake0)+1)/self.deltaelake)))
        vmax=vmin+1
        vmin_value=self.map_exp_elake(vmin)
        vmax_value=self.map_exp_elake(vmax)
        w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio

        return vmin,vmax,w

    def map_palkka(self,v):
        return self.hila_palkka0+self.deltapalkka*v # pitäisikö käyttää exp-hilaa?

    def inv_palkka(self,v):
        vmin=max(0,min(self.n_palkka-2,int((v-self.hila_palkka0)/self.deltapalkka)))
        vmax=vmin+1
        w=(v-self.hila_palkka0)/self.deltapalkka-vmin # lin.approximaatio

        return vmin,vmax,w

    def map_exp_palkka(self,v):
        return self.hila_palkka0+self.deltapalkka*(np.exp(v*self.exppalkkascale)-1)

    def inv_exp_palkka(self,v):
        vmin=max(0,min(self.n_palkka-2,int((np.log(v-self.hila_palkka0)+1)/self.deltapalkka)))
        vmax=vmin+1
        vmin_value=self.map_exp_palkka(vmin)
        vmax_value=self.map_exp_palkka(vmax)
        w=(v-vmin_value)/(self.vmax_value-vmin_value) # lin.approximaatio

        return vmin,vmax,w

    def map_tis(self,v):
        return v # pitäisikö käyttää exp-hilaa?

    def inv_tis(self,v):
        vmin=0 #max(0,min(self.n_tis-2,int((v-self.hila_tis0)/self.deltatis)))
        vmax=min(self.n_tis-1,v)
        w=0.0

        #return int(vmin),int(vmax),w    
        return int(vmax)
    
    # lineaarinen approksimaatio
    def get_V(self,t,s):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        emp,elake,palkka,ika,time_in_state=self.env.state_decode(s)
        emin,emax,we=self.inv_elake(elake)
        pmin,pmax,wp=self.inv_palkka(palkka)
        tismax=self.inv_tis(time_in_state)
        emp=int(emp)
        
        #V=(1-wp)*((1-we)*((1-wt)*self.Hila[t,pmin,emin,emp,tismin]+wt*self.Hila[t,pmin,emin,emp,tismax])\
        #             +we*((1-wt)*self.Hila[t,pmin,emax,emp,tismin]+wt*self.Hila[t,pmin,emax,emp,tismax]))+\
        #      wp*((1-we)*((1-wt)*self.Hila[t,pmax,emin,emp,tismin]+wt*self.Hila[t,pmax,emin,emp,tismax])\
        #             +we*((1-wt)*self.Hila[t,pmax,emax,emp,tismin]+wt*self.Hila[t,pmax,emax,emp,tismax]))

        V=(1-wp)*((1-we)*(self.Hila[t,pmin,emin,emp,tismax])\
                     +we*(self.Hila[t,pmin,emax,emp,tismax]))+\
              wp*((1-we)*(self.Hila[t,pmax,emin,emp,tismax])\
                     +we*(self.Hila[t,pmax,emax,emp,tismax]))

        V=max(0,V)

        return V

    # lineaarinen approksimaatio dynaamisessa ohjelmoinnissa
    def get_actV(self,t,s,full=False):
        '''
        hae hilasta tilan s arvo hetkelle t
        '''
        #print(s)
        emp,elake,palkka,ika,time_in_state=self.env.state_decode(s)
        emin,emax,we=self.inv_elake(elake)
        pmin,pmax,wp=self.inv_palkka(palkka)
        tismax=self.inv_tis(time_in_state)
        emp=int(emp)
        tismax=int(tismax)
        
        n_emp=self.n_acts
        
        V=np.zeros(n_emp)
        emp_set=set([0,1])
        if emp in emp_set:
            if ika<=self.min_retirementage:
                n_emp=2
            else:
                n_emp=3
            
        for k in range(n_emp):
            #print(t,pmin,emin,emp,tismax,k)
            V[k]=max(0,(1-wp)*((1-we)*(self.actHila[t,pmin,emin,emp,tismax,k])+we*(self.actHila[t,pmin,emax,emp,tismax,k]))+\
                           wp*((1-we)*(self.actHila[t,pmax,emin,emp,tismax,k])+we*(self.actHila[t,pmax,emax,emp,tismax,k])))
            #V[k]=max(0,(1-wp)*((1-we)*((1-wt)*self.actHila[t,pmin,emin,emp,tismin,k]+wt*self.actHila[t,pmin,emin,emp,tismax,k])\
            #                      +we*((1-wt)*self.actHila[t,pmin,emax,emp,tismin,k]+wt*self.actHila[t,pmin,emax,emp,tismax,k]))+\
            #               wp*((1-we)*((1-wt)*self.actHila[t,pmax,emin,emp,tismin,k]+wt*self.actHila[t,pmax,emin,emp,tismax,k])\
            #                      +we*((1-wt)*self.actHila[t,pmax,emax,emp,tismin,k]+wt*self.actHila[t,pmax,emax,emp,tismax,k])))
            
            
        act=int(np.argmax(V))
        maxV=np.max(V)

        if full:
            return act,maxV,V
        else:
            return act,maxV
        
    def get_actV_random(self,age):
        if age<self.min_retirementage:
            return np.random.randint(2)
        else:
            return np.random.randint(3)

    # this routine is needed for the dynamic programming
    def get_rewards_continuous(self,s,actions):
        rewards=[]
        sps=[]
        
        start_state=self.env.state_encode(*s)
        #deco=self.env.state_decode(start_state)
        #print('s',*s) #,start_state,deco)
        for a in actions:
            self.env.state=start_state
            newstate, reward, dones, info = self.env.step(a,dynprog=True)
            #if dones:
            #    self.reset()
            sps.append(np.array(newstate))
            rewards.append(reward)
            
        return rewards,sps

    def backtrack(self,t,debug=False):
        '''
        Dynaaminen ohjelmointi hilan avulla
        '''
        age=self.min_age+t
        
        if age<=self.min_retirementage:
            act_set=set([0,1])
        else:
            act_set=set([0,1,2])
            
        m=0
        qr=0
        qw=0
        
        for emp in range(self.n_employment):
            for el in range(self.n_elake):
                for p in range(self.n_palkka): 
                    for tis in range(self.n_tis):
                        palkka=self.map_palkka(p)
                        elake=self.map_elake(el)
                        time_in_state=self.map_tis(tis)

                        # hetken t tila (emp,prev,elake,palkka). Lasketaan palkkio+gamma*U, jos ei vaihda tilaa
                        rts,Sps=self.get_rewards_continuous((emp,elake,palkka,age,time_in_state),act_set)
                        
                        for ind,a in enumerate(act_set):
                            m=m+1
                            #qr=qr+rts[ind]
                            w=self.get_V(t+1,Sps[ind])
                            #qw=qw+self.gamma*w
                            self.actHila[t,p,el,emp,tis,a]=rts[ind]+self.gamma*w

                        self.Hila[t,p,el,emp,tis]=np.max(self.actHila[t,p,el,emp,tis,:])

        if debug:
            self.print_actV(t)
            self.print_V(t)
            print('at age {} mean V {} mean r {}'.format(age,np.mean(self.Hila[t,:,:,:,:]),qr/m),qw/m)

                                
    def train(self,debug=False,save='best/dynamic_prog_V.h5'):
        '''
        Lasketaan optimaalinen työllistyminen/työttömyys/eläköityminen valitulla valintametodilla
        '''
        self.init_grid()
        print('Optimizing behavior')
        tqdm_e = tqdm(range(int(self.n_time)), desc='Score', leave=True, unit=" year")

        for age in range(self.max_age,self.min_age-1,-1):
            t=age-self.min_age
            #print(t)
            self.backtrack(t)
            tqdm_e.set_description("Year " + str(t))
            tqdm_e.update(1)

        self.save_V(save)
          
        
    def simulate(self,debug=False,pop=1_000,save=None,load='dynamic_prog_V.h5'):
        '''
        Lasketaan työllisyysasteet ikäluokittain
        '''
        if pop is not None:
            self.n_pop=pop

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage)
        self.load_V(load)        

        self.env.seed(1234)   
        self.env.env_seed(4567)            
        tqdm_e = tqdm(range(int(pop)), desc='Population', leave=True, unit=" p")

        for n in range(pop):
            state=self.env.reset()
            
            for t in range(self.n_time):
                if debug:
                    act,maxV,v=self.get_actV(t,state,full=True)
                else:
                    act,maxV=self.get_actV(t,state)
                
                newstate,r,done,info=self.env.step(act)
                #print(r,info['r'])
                #r=info['r']
                self.episodestats.add(n,act,r,state,newstate,debug=debug,aveV=maxV)
                state=newstate
                
                if done:
                    tqdm_e.update(1)                
                    if debug:
                        print('done')
                    break      
        
        if save is not None:
            self.episodestats.save_sim(save)
          
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
        dset = f.create_dataset("Hila", data=self.Hila, dtype='f16')
        dset2 = f.create_dataset("actHila", data=self.actHila, dtype='f16')
        f.close()
        
    def load_V(self,filename):
        f = h5py.File(filename, 'r')
        self.Hila = f.get('Hila').value
        self.actHila = f.get('actHila').value
        f.close()
        
    def plot_act(self):
        act_ratio=np.sum(self.actions==1,axis=1)/2000
        x=np.linspace(self.min_age,self.max_age,self.n_time)
        fig,ax=plt.subplots()
        ax.plot(x,act_ratio)    
        ax.set_xlabel('Aika')
        ax.set_ylabel('Tilasta pois-siirtyneet')
        plt.show()

    def plot_img(self,img,xlabel="Eläke",ylabel="Palkka",title="Eployed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        heatmap = plt.pcolor(img) 
        plt.colorbar(heatmap)        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_higher(self,t):
        emp=1
        prev=0
        q=np.array(self.actHila[t,:,:,emp,prev,0]>self.actHila[t,:,:,emp,prev,1]).astype(int)
        self.plot_img(q,xlabel="Eläke",ylabel="Palkka",title='Employed, stay in state')
        emp=0
        q=np.array(self.actHila[t,:,:,emp,prev,0]>self.actHila[t,:,:,emp,prev,1]).astype(int)
        self.plot_img(q,xlabel="Eläke",ylabel="Palkka",title='Unemployed, stay in state')

    def print_V(self,t):
        print('t=',t)
        print('töissä\n',self.Hila[t,:,:,1,0])
        print('ei töissä\n',self.Hila[t,:,:,0,0])
        print('eläke\n',self.Hila[t,:,:,2,0])

    def plot_V(self,t):
        self.plot_img(self.Hila[t,:,:,1,0],xlabel="Eläke",ylabel="Palkka",title='Töissä')
        self.plot_img(self.Hila[t,:,:,0,0],xlabel="Eläke",ylabel="Palkka",title='Työttömänä')
        self.plot_img(self.Hila[t,:,:,1,0]-self.Hila[t,:,:,0,0],xlabel="Eläke",ylabel="Palkka",title='Työssä-Työtön')

    def print_actV(self,t):
        print('t=',t)
        if t+self.min_age>self.min_retirementage:
            print('eläke (act) pois\n{}\neläke (act) pysyy\n{}\n'.format(self.actHila[t,:,:,2,0,1],self.actHila[t,:,:,2,0,0]))
            print('töissä (act) pois\n{}\ntöissä (act) pysyy\n{}\ntöissä (act) eläköityy\n{}\n'.format(self.actHila[t,:,:,1,0,1],self.actHila[t,:,:,1,0,0],self.actHila[t,:,:,1,0,2]))
            print('ei töissä (act) pois\n{}\nei töissä (act) pysyy\n{}\nei töissä eläköityy\n{}\n'.format(self.actHila[t,:,:,0,0,1],self.actHila[t,:,:,0,0,0],self.actHila[t,:,:,0,0,2]))
        else:
            print('töissä (act) pois\n',self.actHila[t,:,:,1,0,1],'\ntöissä (act) pysyy\n',self.actHila[t,:,:,1,0,0])
            print('ei töissä (act) pois\n',self.actHila[t,:,:,0,0,1],'\nei töissä (act) pysyy\n',self.actHila[t,:,:,0,0,0])
        #print('töissä (act ero)\n',self.actHila[t,:,:,1,0,1]-self.actHila[t,:,:,1,0,0])
        #print('ei töissä (act ero)\n',self.actHila[t,:,:,0,0,1]-self.actHila[t,:,:,0,0,0])

    def plot_actV_diff(self,t):
        self.plot_img(self.actHila[t,:,:,1,0,1]-self.actHila[t,:,:,1,0,0],xlabel="Eläke",ylabel="Palkka",title='Töissä (ero switch-stay)')
        self.plot_img(self.actHila[t,:,:,0,0,1]-self.actHila[t,:,:,0,0,0],xlabel="Eläke",ylabel="Palkka",title='Työttömänä (ero switch-stay)')

    def plot_actV(self,t,emp=1,time_in_state=0):
        q=np.zeros((self.n_elake,self.n_palkka))
        for el in range(self.n_elake):
            for p in range(self.n_palkka): 
                palkka=self.map_palkka(p)
                elake=self.map_elake(el)

                q[el,p]=np.argmax(self.actHila[t,p,el,emp,time_in_state,:])

        self.plot_img(q,xlabel="Eläke",ylabel="Palkka",title='argmax')

    
    def RL_simulate_V(self,age,emp=0,time_in_state=0,rlmodel='acktr',load='perus'):
        self.episodestats_init()
        
        policy_kwargs,n_cpu,savename,loadname=self.get_multiprocess_env(rlmodel,load)

        env = SubprocVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])
        #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])

        #print('predicting...')

        if rlmodel=='a2c':
            model = A2C.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif rlmodel=='acer':
            model = ACER.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        elif rlmodel=='acktr':
            model = ACKTR.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
        else:        
            model = DQN.load(savename, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs)
            
        prev=0
        toe=0
        self.fake_act=np.zeros((self.n_elake,self.n_palkka))
        for el in range(self.n_elake):
            for p in range(self.n_palkka): 
                palkka=self.map_palkka(p)
                elake=self.map_elake(el)
                if emp==2:
                    state=self.env.state_encode(emp,elake,0,age,time_in_state)
                elif emp==1:
                    state=self.env.state_encode(emp,0,palkka,age,time_in_state)
                else:
                    state=self.env.state_encode(emp,0,palkka,age,time_in_state)

                act, predstate = model.predict(state)
                self.fake_act[el,p]=act
                        
        self.plot_img(self.fake_act,xlabel="Palkka",ylabel="Eläke",title="pred "+load+" action tilasta "+str(emp))
