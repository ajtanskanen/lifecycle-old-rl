'''

    lifecycle.py

    implements the lifecycle model that predicts how people will act in the presence of
    social security

'''

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
#from fin_benefits import Benefits
import matplotlib.pyplot as plt
import gym_unemployment
#import cygym_unemployment
import h5py
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm_notebook as tqdm
import os
from . episodestats import EpisodeStats
from . simstats import SimStats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['OMP_NUM_THREADS'] = '4'  # or any {'0', '1', '2'}
OMP_NUM_THREADS=4

import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# use stable baselines
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines import A2C, ACER, DQN, ACKTR#, TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from .vec_monitor import VecMonitor

class Lifecycle():

    def __init__(self,env=None,minimal=False,timestep=0.25,ansiopvraha_kesto300=None,
                    ansiopvraha_kesto400=None,karenssi_kesto=None,
                    ansiopvraha_toe=None,perustulo=None,mortality=None,
                    randomness=None,include_putki=None,preferencenoise=None,
                    callback_minsteps=None,pinkslip=True,plotdebug=False,
                    use_sigma_reduction=None,porrasta_putki=None,perustulomalli=None,
                    porrasta_1askel=None,porrasta_2askel=None,porrasta_3askel=None,
                    osittainen_perustulo=None,gamma=None,exploration=None,exploration_ratio=None,
                    irr_vain_tyoelake=None,additional_income_tax=None,additional_tyel_premium=None,
                    additional_kunnallisvero=None,additional_income_tax_high=None,
                    year=2018,version=2,scale_tyel_accrual=None,preferencenoise_level=None,
                    scale_additional_tyel_accrual=None,valtionverotaso=None,perustulo_asetettava=None,
                    porrasta_toe=None,include_halftoe=None):                    
        '''
        Alusta muuttujat
        '''
        self.timestep=timestep # 0.25 = 3kk askel
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.min_age = 18
        self.max_age = 70
        self.min_retirementage=63.5
        self.max_retirementage=68
        self.n_pop = 1000
        self.callback_minsteps = 1_000
        self.year=year
        self.version=version

        # apumuuttujia
        self.n_age = self.max_age-self.min_age+1
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+2
        self.gamma = 0.92

        self.karenssi_kesto=0.25

        self.plotdebug=False
        self.include_pinkslip=True
        self.mortality=False
        self.perustulo=False
        self.ansiopvraha_kesto300=None
        self.ansiopvraha_kesto400=None
        self.include_putki=None
        self.ansiopvraha_toe=None
        self.environment='unemployment-v0'
        self.include_preferencenoise=False
        self.preferencenoise_level=0
        self.porrasta_putki=True
        self.porrasta_1askel=True
        self.porrasta_2askel=True
        self.porrasta_3askel=True
        self.osittainen_perustulo=True
        self.irr_vain_tyoelake=True
        self.additional_income_tax=0.0
        self.additional_tyel_premium=0.0
        self.additional_kunnallisvero=0.0
        self.additional_income_tax_high=0.0
        self.scale_tyel_accrual=False
        self.scale_additional_tyel_accrual=0
        self.perustulo_asetettava=None
        self.valtionverotaso=None
        self.include_halftoe=None
        self.porrasta_toe=None

        if callback_minsteps is not None:
            self.callback_minsteps=callback_minsteps

        if karenssi_kesto is not None:
            self.karenssi_kesto=karenssi_kesto

        if irr_vain_tyoelake is not None:
            self.irr_vain_tyoelake=irr_vain_tyoelake

        if additional_income_tax is not None:
            self.additional_income_tax=additional_income_tax

        if additional_tyel_premium is not None:
            self.additional_tyel_premium=additional_tyel_premium

        if additional_kunnallisvero is not None:
            self.additional_kunnallisvero=additional_kunnallisvero

        if additional_income_tax_high is not None:
            self.additional_income_tax_high=additional_income_tax_high
            
        if perustulo_asetettava is not None:
            self.perustulo_asetettava=perustulo_asetettava

        if valtionverotaso is not None:
            self.valtionverotaso=valtionverotaso

        if plotdebug is not None:
            self.plotdebug=plotdebug

        if pinkslip is not None:
            self.include_pinkslip=pinkslip
            
        if mortality is not None:
            self.mortality=mortality

        self.randomness=randomness

        if ansiopvraha_kesto300 is not None:
            self.ansiopvraha_kesto300=ansiopvraha_kesto300

        if include_putki is not None:
            self.include_putki=include_putki

        self.exploration=False
        self.exploration_ratio=0.10

        if exploration is not None:
            self.exploration=exploration
            
        if exploration_ratio is not None:
            self.exploration_ratio=exploration_ratio

        if ansiopvraha_kesto400 is not None:
            self.ansiopvraha_kesto400=ansiopvraha_kesto400

        if ansiopvraha_toe is not None:
            self.ansiopvraha_toe=ansiopvraha_toe
            
        if porrasta_putki is not None:
            self.porrasta_putki=porrasta_putki
            
        if porrasta_1askel is not None:
            self.porrasta_1askel=porrasta_1askel

        if gamma is not None:
            self.gamma=gamma

        if porrasta_2askel is not None:
            self.porrasta_2askel=porrasta_2askel

        if porrasta_3askel is not None:
            self.porrasta_3askel=porrasta_3askel

        if perustulo is not None:
            self.perustulo=perustulo
            
        if preferencenoise is not None:
            self.include_preferencenoise=preferencenoise
            
        if preferencenoise_level is not None:
            self.preferencenoise_level=preferencenoise_level
            
        if osittainen_perustulo is not None:
            self.osittainen_perustulo=osittainen_perustulo
            
        if scale_tyel_accrual is not None:
            self.scale_tyel_accrual=scale_tyel_accrual
            
        if scale_additional_tyel_accrual is not None:
            self.scale_additional_tyel_accrual=scale_additional_tyel_accrual

        if env is not None:
            self.environment=env
            
        self.use_sigma_reduction=True
        if use_sigma_reduction is not None:
            self.use_sigma_reduction=use_sigma_reduction

        if include_halftoe is not None:
            self.include_halftoe=include_halftoe
        if porrasta_toe is not None:
            self.porrasta_toe=porrasta_toe
            
        # alustetaan gym-environment
        if minimal:
            #if EK:
            #    self.environment='unemploymentEK-v0'
            #else:
            #    self.environment='unemployment-v0'

            self.minimal=True
            self.version=0
            self.gym_kwargs={'step': self.timestep,'gamma':self.gamma,
                'min_age': self.min_age, 'max_age': self.max_age,
                'plotdebug': self.plotdebug, 
                'min_retirementage': self.min_retirementage, 'max_retirementage':self.max_retirementage,
                'reset_exploration_go': self.exploration,'reset_exploration_ratio': self.exploration_ratio,
                'irr_vain_tyoelake': self.irr_vain_tyoelake}
            #self.n_employment = 3
            #self.n_acts = 3
        else:
            #if EK:
            #    self.environment='unemploymentEK-v1'
            #else:
            #    self.environment='unemployment-v1'

            self.minimal=False
            self.gym_kwargs={'step': self.timestep,'gamma':self.gamma,
                'min_age': self.min_age, 'max_age': self.max_age,
                'min_retirementage': self.min_retirementage, 'max_retirementage':self.max_retirementage,
                'ansiopvraha_kesto300': self.ansiopvraha_kesto300,'ansiopvraha_kesto400': self.ansiopvraha_kesto400,
                'ansiopvraha_toe': self.ansiopvraha_toe,'include_pinkslip':self.include_pinkslip,
                'perustulo': self.perustulo, 'karenssi_kesto': self.karenssi_kesto,
                'mortality': self.mortality, 'randomness': self.randomness,
                'porrasta_putki': self.porrasta_putki, 'porrasta_1askel': self.porrasta_1askel,
                'porrasta_2askel': self.porrasta_2askel, 'porrasta_3askel': self.porrasta_3askel,
                'include_putki': self.include_putki, 'use_sigma_reduction': self.use_sigma_reduction,
                'plotdebug': self.plotdebug, 'include_preferencenoise': self.include_preferencenoise,
                'preferencenoise_level': self.preferencenoise_level,
                'perustulomalli': perustulomalli,'osittainen_perustulo':self.osittainen_perustulo,
                'reset_exploration_go': self.exploration,'reset_exploration_ratio': self.exploration_ratio,
                'irr_vain_tyoelake': self.irr_vain_tyoelake, 
                'additional_income_tax': self.additional_income_tax,
                'additional_tyel_premium': self.additional_tyel_premium, 
                'additional_income_tax_high': self.additional_income_tax_high,
                'additional_kunnallisvero': self.additional_kunnallisvero,
                'scale_tyel_accrual': self.scale_tyel_accrual,
                'scale_additional_tyel_accrual': self.scale_additional_tyel_accrual,
                'perustulo': self.perustulo, 'valtionverotaso': self.valtionverotaso,
                'perustulo_asetettava': self.perustulo_asetettava, 
                'include_halftoe': self.include_halftoe,
                'porrasta_toe': self.porrasta_toe}
            #self.n_acts = 4
            #if self.mortality:
            #    self.n_employment = 16
            #else:
            #    self.n_employment = 15
                
        # Create log dir & results dirs
        self.log_dir = "tmp/" # +str(env_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tenb_dir = "tmp/tenb/" # +str(env_id)
        os.makedirs(self.tenb_dir, exist_ok=True)
        os.makedirs("saved/", exist_ok=True)
        os.makedirs("results/", exist_ok=True)
        os.makedirs("best/", exist_ok=True)
        
        self.env = gym.make(self.environment,kwargs=self.gym_kwargs)
        self.n_employment,self.n_acts=self.env.get_n_states()

        self.episodestats=SimStats(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                   self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,
                                   version=self.version,params=self.gym_kwargs,year=self.year)

    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        self.env.explain()
        #print('Parameters of lifecycle:\ntimestep {}\ngamma {} per anno\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.min_age,self.max_age,self.min_retirementage))
        #print('max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_toe {}'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_toe))
        #print('perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}'.format(self.perustulo,self.karenssi_kesto,self.mortality,self.randomness))
        #print('include_putki {}\ninclude_pinkslip {}\n'.format(self.include_putki,self.include_pinkslip))

    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)

    def get_multiprocess_env(self,rlmodel,debug=False,arch=None,predict=False):

        if arch is not None:
            print('arch',arch)

        # multiprocess environment
        if rlmodel=='a2c':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='acer':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='deep_acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 512, 256, 128, 64]) 
            n_cpu = 8 # 12 # 20
        elif rlmodel=='acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 512, 256]) 
            n_cpu = 8 # 12 # 20
        elif rlmodel=='leaky_acktr': # tf.nn.leakyrelu
            if arch is not None:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=arch) 
            else:
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[256, 256, 16]) 
            if predict:
                n_cpu = 20
            else:
                n_cpu = 8 # 12 # 20
        elif rlmodel=='small_acktr' or rlmodel=='small_lnacktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256, 128]) 
            n_cpu = 4 #8
        elif rlmodel=='large_acktr':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256, 64, 16])
            n_cpu = 4 # 12
        elif rlmodel=='lstm' or rlmodel=='lnacktr':
            policy_kwargs = dict()
            n_cpu = 4
        elif rlmodel=='trpo':
            policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 64, 16])
            n_cpu = 4
        elif rlmodel=='dqn': # DQN
            policy_kwargs = dict(act_fun=tf.nn.relu, layers=[64, 64])
            n_cpu = 1
            rlmodel='dqn'
        else:
            error('Unknown rlmodel')

        if debug:
            n_cpu=1
            
        return policy_kwargs,n_cpu

    def setup_rlmodel(self,rlmodel,loadname,env,batch,policy_kwargs,learning_rate,
                      cont,max_grad_norm=None,tensorboard=False,verbose=1,n_cpu=1,learning_schedule='linear',
                      vf=None,gae_lambda=None):
        '''
        Alustaa RL-mallin ajoa varten
        '''
        n_cpu_tf_sess=4 #n_cpu #4
        #batch=max(1,int(np.ceil(batch/n_cpu)))
        batch=max(1,int(np.ceil(batch/n_cpu)))
        
        full_tensorboard_log=True
        if vf is not None:
            vf_coef=vf
        else:
            vf_coef=0.10 # baseline 0.25, best 0.10

        if max_grad_norm is None:
            max_grad_norm=0.05
            
        max_grad_norm=0.001
        kfac_clip=0.001
        
        if cont:
            learning_rate=0.25*learning_rate
            
        #scaled_learning_rate=learning_rate*np.sqrt(batch)
        scaled_learning_rate=learning_rate*batch
        print('batch {} learning rate {} scaled {} n_cpu {}'.format(batch,learning_rate,
            scaled_learning_rate,n_cpu))

        if cont:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = A2C.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                     tensorboard_log=self.tenb_dir, policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,
                                     n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = A2C.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                     policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel in set(['small_acktr','acktr','large_acktr','deep_acktr','leaky_acktr']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                       learning_rate=np.sqrt(batch)*learning_rate,vf_coef=vf_coef,gae_lambda=gae_lambda,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel=='small_lnacktr' or rlmodel=='lnacktr':
                from stable_baselines.common.policies import MlpLnLstmPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,kfac_clip=kfac_clip,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=np.sqrt(batch)*learning_rate,kfac_clip=kfac_clip,
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy 
                if tensorboard:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       tensorboard_log=self.tenb_dir,learning_rate=scaled_learning_rate, 
                                       policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,
                                       full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR.load(loadname, env=env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                       learning_rate=scaled_learning_rate, policy_kwargs=policy_kwargs,
                                       max_grad_norm=max_grad_norm,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            else:
                if tensorboard:
                    from stable_baselines.deepq.policies import MlpPolicy # for DQN
                    model = DQN.load(loadname, env=env, verbose=verbose,gamma=self.gamma,
                                     batch_size=batch,tensorboard_log=self.tenb_dir,
                                     policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,
                                     full_tensorboard_log=full_tensorboard_log,learning_rate=learning_rate)
                else:
                    from stable_baselines.deepq.policies import MlpPolicy # for DQN
                    model = DQN.load(loadname, env=env, verbose=verbose,gamma=self.gamma,
                                     batch_size=batch,tensorboard_log=self.tenb_dir,
                                     policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,
                                     learning_rate=learning_rate)
        else:
            if rlmodel=='a2c':
                from stable_baselines.common.policies import MlpPolicy 
                model = A2C(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time, 
                            tensorboard_log=self.tenb_dir, policy_kwargs=policy_kwargs,lr_schedule=learning_schedule)
            elif rlmodel in set(['small_acktr','acktr','large_acktr','deep_acktr','leaky_acktr']):
                from stable_baselines.common.policies import MlpPolicy 
                if tensorboard:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate,kfac_clip=kfac_clip,
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                full_tensorboard_log=full_tensorboard_log,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR(MlpPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,kfac_clip=kfac_clip,
                                learning_rate=scaled_learning_rate, max_grad_norm=max_grad_norm,gae_lambda=gae_lambda,vf_coef=vf_coef,
                                policy_kwargs=policy_kwargs,lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
            elif rlmodel=='small_lnacktr' or rlmodel=='lnacktr':
                from stable_baselines.common.policies import MlpLnLstmPolicy 
                if tensorboard:
                    model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                tensorboard_log=self.tenb_dir, learning_rate=scaled_learning_rate, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,full_tensorboard_log=full_tensorboard_log,
                                lr_schedule=learning_schedule,n_cpu_tf_sess=n_cpu_tf_sess)
                else:
                    model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                                learning_rate=scaled_learning_rate,n_cpu_tf_sess=n_cpu_tf_sess, 
                                policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule)
            elif rlmodel=='lstm':
                from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy 
                model = ACKTR(MlpLstmPolicy, env, verbose=verbose,gamma=self.gamma,n_steps=batch*self.n_time,
                            tensorboard_log=self.tenb_dir, learning_rate=learning_rate,n_cpu_tf_sess=n_cpu_tf_sess, 
                            policy_kwargs=policy_kwargs,max_grad_norm=max_grad_norm,lr_schedule=learning_schedule)
            else:
                from stable_baselines.deepq.policies import MlpPolicy # for DQN
                if tensorboard:
                    model = DQN(MlpPolicy, env, verbose=verbose,gamma=self.gamma,batch_size=batch, 
                                tensorboard_log=self.tenb_dir,learning_rate=learning_rate,
                                policy_kwargs=policy_kwargs,full_tensorboard_log=full_tensorboard_log) 
                else:
                    model = DQN(MlpPolicy, env, verbose=verbose,gamma=self.gamma,batch_size=batch,
                                learning_rate=learning_rate,policy_kwargs=policy_kwargs) 
                            
        return model

    def make_env(self,env_id, rank, kwargs, seed=None, use_monitor=True):
        """
        Utility function for multiprocessed env.#

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id,kwargs=kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.env_seed(seed + rank + 100)

            # monitor enables various things, not used by default
            #print('monitor=',use_monitor)
            #if use_monitor:
            #    env = Monitor(env, self.log_dir, allow_early_resets=True)

            return env

        if seed is not None:
            set_global_seeds(seed)

        return _init()

    def callback(self,_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        #global n_steps, best_mean_reward
        # Print stats every 1000 calls, if needed

        min_steps=0
        mod_steps=1
        if self.callback_minsteps is not None:
            hist_eps=self.callback_minsteps
        else:
            hist_eps=5000
        if (self.n_steps + 1) % mod_steps == 0 and self.n_steps > min_steps:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) >= hist_eps:
                mean_reward = np.mean(y[-hist_eps:])
                print(x[-1], 'timesteps', len(y), 'episodes') #, 'mean', mean_reward, 'out of', y[-hist_eps:])

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    print("NEW Best mean reward: {:.2f} - Last best reward per episode: {:.2f}".format(mean_reward,self.best_mean_reward))
                    self.best_mean_reward = mean_reward
                    # Example for saving best model                    print("Saving new best model")
                    print('saved as ',self.bestname)
                    _locals['self'].save(self.bestname)
                else:
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
    
        self.n_steps += 1

        return True

    def train(self,train=False,debug=False,steps=20000,cont=False,rlmodel='dqn',
                save='saved/malli',pop=None,batch=1,max_grad_norm=None,learning_rate=0.25,
                start_from=None,max_n_cpu=1000,plot=True,use_vecmonitor=False,
                bestname='tmp/best2',use_callback=False,log_interval=100,verbose=1,plotdebug=False,
                learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
        '''
        Opetusrutiini
        '''
        self.best_mean_reward, self.n_steps = -np.inf, 0

        if pop is not None:
            self.n_pop=pop

        self.rlmodel=rlmodel
        self.bestname=bestname

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        # multiprocess environment
        policy_kwargs,n_cpu=self.get_multiprocess_env(self.rlmodel,debug=debug,arch=arch)  

        self.savename=save
        n_cpu=min(max_n_cpu,n_cpu)

        if debug:
            print('use_vecmonitor',use_vecmonitor)
            print('use_callback',use_callback)

        gkwargs=self.gym_kwargs.copy()
        gkwargs.update({'train':True})
    
        nonvec=False
        if nonvec:
            env=self.env
        else:
            if use_vecmonitor:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=False) for i in range(n_cpu)], start_method='spawn')
                env = VecMonitor(env,filename=self.log_dir+'monitor.csv')
            else:
                env = SubprocVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='spawn')
                #env = ShmemVecEnv([lambda: self.make_env(self.environment, i, gkwargs, use_monitor=use_callback) for i in range(n_cpu)], start_method='fork')

            #if False:
                #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=gkwargs) for i in range(n_cpu)])

        normalize=False
        if normalize:
            normalize_kwargs={}
            env = VecNormalize(env, **normalize_kwargs)

        model=self.setup_rlmodel(self.rlmodel,start_from,env,batch,policy_kwargs,learning_rate,
                                cont,max_grad_norm=max_grad_norm,verbose=verbose,n_cpu=n_cpu,
                                learning_schedule=learning_schedule,vf=vf,gae_lambda=gae_lambda)
        print('training...')

        if use_callback: # tässä ongelma, vecmonitor toimii => kuitenkin monta callbackia
            model.learn(total_timesteps=steps, callback=self.callback,log_interval=log_interval)
        else:
            model.learn(total_timesteps=steps, log_interval=log_interval)

        model.save(save)
        print('done')

        del model,env

    def save_to_hdf(self,filename,nimi,arr,dtype):
        f = h5py.File(filename, 'w')
        dset = f.create_dataset(nimi, data=arr, dtype=dtype)
        f.close()

    def load_hdf(self,filename,nimi):
        f = h5py.File(filename, 'r')
        val=f.get(nimi).value
        f.close()
        return val
        
    def setup_model(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,
                    deterministic=False,arch=None,predict=False):

        if pop is not None:
            self.n_pop=pop

        if load is not None:
            self.loadname=load

        if rlmodel is not None:
            self.rlmodel=rlmodel
            
        print('simulate')
            
        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)

        print('simulating ',self.loadname)

        # multiprocess environment
        policy_kwargs,n_cpu=self.get_multiprocess_env(rlmodel,debug=debug,arch=arch,predict=predict)
        n_cpu_tf_sess=4

        nonvec=False
        if nonvec:
            env=self.env
            #env.seed(4567)
            #env.env_seed(4567)
        else:
            env = SubprocVecEnv([lambda: self.make_env(self.environment, i, self.gym_kwargs) for i in range(n_cpu)])
            #env = SubprocVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])
            #env = DummyVecEnv([lambda: gym.make(self.environment,kwargs=self.gym_kwargs) for i in range(n_cpu)])

        normalize=False
        if normalize:
            normalize_kwargs={}
            env = VecNormalize(env, **normalize_kwargs)
            
        print('predicting...')

        if self.rlmodel=='a2c':
            model = A2C.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel in set(['acktr','small_acktr','lnacktr','small_lnacktr','deep_acktr','leaky_acktr']):
            model = ACKTR.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel=='trpo':
            model = TRPO.load(load, env=env, verbose=1,gamma=self.gamma, policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        elif self.rlmodel=='dqn':
            model = DQN.load(load, env=env, verbose=1,gamma=self.gamma,prioritized_replay=True,policy_kwargs=policy_kwargs,n_cpu_tf_sess=n_cpu_tf_sess)
        else:
            error('unknown model')

        return model,env,n_cpu

    def simulate(self,debug=False,rlmodel='acktr',plot=True,load=None,pop=None,
                 deterministic=False,save='results/testsimulate',arch=None):

        model,env,n_cpu=self.setup_model(debug=debug,rlmodel=rlmodel,plot=plot,load=load,pop=pop,
                 deterministic=deterministic,arch=arch,predict=True)

        states = env.reset()
        n=n_cpu-1
        p=np.zeros(self.n_pop)
        pop_num=np.array([k for k in range(n_cpu)])
        tqdm_e = tqdm(range(int(self.n_pop)), desc='Population', leave=True, unit=" p")

        while n<self.n_pop:
            act, predstate = model.predict(states,deterministic=deterministic)
            newstate, rewards, dones, infos = env.step(act)

            #done=False
            for k in range(n_cpu):
                if dones[k]:
                    self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],infos[k]['terminal_observation'] ,infos[k],debug=debug)
                    tqdm_e.update(1)
                    n+=1
                    tqdm_e.set_description("Pop " + str(n))
                    #done=True
                    pop_num[k]=n
                else:
                    self.episodestats.add(pop_num[k],act[k],rewards[k],states[k],newstate[k],infos[k],debug=debug)
    
            states = newstate

        self.episodestats.save_sim(save)

        if plot:
            self.render()

        if False:
            return self.emp
   
    def get_reward(self):
        return self.episodestats.get_reward()
   
    def render(self,load=None,figname=None):
        if load is not None:
            self.episodestats.render(load=load,figname=figname)
        else:
            self.episodestats.render(figname=figname)
   
    def render_reward(self,load=None,figname=None):
        if load is not None:
            self.episodestats.load_sim(load)
            return self.episodestats.comp_total_reward()
        else:
            return self.episodestats.comp_total_reward()
   
    def render_laffer(self,load=None,figname=None):
        if load is not None:
            self.episodestats.load_sim(load)
            
        rew=self.episodestats.comp_total_reward()
            
        q=self.episodestats.comp_budget()
        q2=self.episodestats.comp_participants(scale=True)
        htv=q2['htv']
        palkansaajia=q2['palkansaajia']
        muut_tulot=q['muut tulot']
        tC=0.2*max(0,q['tyotulosumma']-q['verot+maksut'])
        kiila,qc=self.episodestats.comp_verokiila()
        #tyollaste,_,tyotaste,_,_=self.episodestats.comp_employed()
        tyollaste,tyotaste=0,0
        #
        #qq={}
        #qq['tI']=q['verot+maksut']/q['tyotulosumma']
        #qq['tC']=tC/q['tyotulosumma']
        #qq['tP']=q['ta_maksut']/q['tyotulosumma']
        #
        #print(qq,qc)
            
        return rew,q['tyotulosumma'],q['verot+maksut'],htv,muut_tulot,kiila,tyollaste,tyotaste,palkansaajia
   
    def load_sim(self,load=None):
        self.episodestats.load_sim(load)
   
    def run_dummy(self,strategy='emp',debug=False,pop=None):
        '''
        Lasketaan työllisyysasteet ikäluokittain satunnaisella politiikalla
        hyödyllinen unemp-envin profilointiin
        '''
        
        self.n_pop=pop

        self.episodestats.reset(self.timestep,self.n_time,self.n_employment,self.n_pop,
                                self.env,self.minimal,self.min_age,self.max_age,self.min_retirementage,self.year)
        if pop is not None:
            self.n_pop=pop

        print('simulating...')
        tqdm_e = tqdm(range(int(self.n_pop)), desc='Population', leave=True, unit=" p")
        
        self.env.seed(1234) 
        state = self.env.reset()
        age=self.min_age
        p=np.zeros(self.n_pop)
        n_cpu=1
        n=0
        pop_num=np.array([k for k in range(n_cpu)])
        v=2 # versio
        while n<self.n_pop:
            #print('agent {}'.format(n))
            tqdm_e.update(1)
            tqdm_e.set_description("Pop " + str(n))
            
            done=False
            pop_num[0]=n
            
            while not done:
                if v==1:
                    emp,_,_,_,age=self.env.state_decode(state) # current employment state
                elif v==3:
                    emp,_,_,_,age,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state
                else:
                    emp,_,_,_,age,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=self.env.state_decode(state) # current employment state            

                if strategy=='random':
                    act=np.random.randint(3)
                elif strategy=='emp':
                    if age>=self.min_retirementage:
                        if emp!=2:
                            act=2
                        else:
                            act=0
                    else:
                        if emp==0:
                            act=1
                        else:
                            act=0
                elif strategy=='unemp':
                    if age>=self.retirement_age:
                        if emp!=2:
                            act=2
                        else:
                            act=0
                    else:
                        if emp==1:
                            act=1
                        else:
                            act=0
                elif strategy=='alt':
                    if age>=self.min_retirementage:
                        if emp!=2:
                            act=2
                        else:
                            act=0
                    else:
                        act=1

                newstate,r,done,_=self.env.step(act)
                
                if done:
                    n=n+1
                    if debug:
                        print('done')
                    newstate=self.env.reset()
                    break
                else:
                    self.episodestats.add(pop_num[0],act,r,state,newstate,debug=debug)
                    
                state=newstate

        #self.plot_stats(5)
        #self.plot_stats()
        #self.plot_reward()

    def compare_with(self,cc2,label1='vaihtoehto',label2='perus',figname=None,grayscale=False,dash=True):
        '''
        compare_with

        compare results obtained another model
        '''
        self.episodestats.compare_with(cc2.episodestats,label1=label1,label2=label2,grayscale=grayscale,figname=figname,dash=dash)

    def run_results(self,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',twostage=False,
               save='saved/perusmalli',debug=False,simut='simut',results='results/simut_res',
               stats='results/simut_stats',deterministic=True,train=True,predict=True,
               batch1=1,batch2=100,cont=False,start_from=None,plot=False,callback_minsteps=None,
               verbose=1,plotdebug=None,max_grad_norm=None,learning_rate=0.25,log_interval=10,
               learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
   
        '''
        run_results

        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
        
        if self.plotdebug or plotdebug:
            debug=True
   
        self.n_pop=pop
        if callback_minsteps is not None:
            self.callback_minsteps=callback_minsteps

        if train: 
            print('train...')
            if cont:
                self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,verbose=verbose,
                                  debug=debug,save=save,batch1=batch1,batch2=batch2,
                                  cont=cont,start_from=start_from,twostage=twostage,plotdebug=plotdebug,
                                  max_grad_norm=max_grad_norm,learning_rate=learning_rate,log_interval=log_interval,
                                  learning_schedule=learning_schedule,vf=vf,arch=arch,gae_lambda=gae_lambda)
            else:
                self.run_protocol(rlmodel=rlmodel,steps1=steps1,steps2=steps2,verbose=verbose,
                                 debug=debug,batch1=batch1,batch2=batch2,cont=cont,
                                 save=save,twostage=twostage,plotdebug=plotdebug,
                                 max_grad_norm=max_grad_norm,learning_rate=learning_rate,log_interval=log_interval,
                                 learning_schedule=learning_schedule,vf=vf,arch=arch,gae_lambda=gae_lambda)
        if predict:
            #print('predict...')
            self.predict_protocol(pop=pop,rlmodel=rlmodel,load=save,plotdebug=plotdebug,
                          debug=debug,deterministic=deterministic,results=results,arch=arch)
        if plot:
            self.render(load=results)
          
    def run_protocol(self,steps1=2_000_000,steps2=1_000_000,rlmodel='acktr',
               debug=False,batch1=1,batch2=1000,cont=False,twostage=False,log_interval=10,
               start_from=None,save='best3',verbose=1,plotdebug=None,max_grad_norm=None,
               learning_rate=0.25,learning_schedule='linear',vf=None,arch=None,gae_lambda=None):
        '''
        run_protocol

        train RL model in two steps:
        1. train with a short batch, not saving the best model
        2. train with a long batch, save the best model during the training
        '''
        
        if twostage:
            tmpname='tmp/simut_100'
        else:
            tmpname=save
            
        if steps1>0:
            print('phase 1')
            if cont:
                self.train(steps=steps1,cont=cont,rlmodel=rlmodel,save=tmpname,batch=batch1,debug=debug,
                           start_from=start_from,use_callback=False,use_vecmonitor=False,
                           log_interval=log_interval,verbose=1,plotdebug=plotdebug,vf=vf,arch=arch,gae_lambda=gae_lambda,
                           max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule)
            else:
                self.train(steps=steps1,cont=False,rlmodel=rlmodel,save=tmpname,batch=batch1,debug=debug,vf=vf,arch=arch,
                           use_callback=False,use_vecmonitor=False,log_interval=log_interval,verbose=1,plotdebug=plotdebug,gae_lambda=gae_lambda,
                           max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule)

        if twostage and steps2>0:
            print('phase 2')
            self.train(steps=steps2,cont=True,rlmodel=rlmodel,save=tmpname,
                       debug=debug,start_from=tmpname,batch=batch2,verbose=verbose,
                       use_callback=False,use_vecmonitor=False,log_interval=log_interval,bestname=save,plotdebug=plotdebug,
                       max_grad_norm=max_grad_norm,learning_rate=learning_rate,learning_schedule=learning_schedule)

    def predict_protocol(self,pop=1_00,rlmodel='acktr',results='results/simut_res',arch=None,
                         load='saved/malli',debug=False,deterministic=False,plotdebug=None):
        '''
        predict_protocol

        simulate the three models obtained from run_protocol
        '''
 
        # simulate the saved best
        self.simulate(pop=pop,rlmodel=rlmodel,plot=False,debug=debug,arch=arch,
                      load=load,save=results,deterministic=deterministic)

    def run_distrib(self,n=5,steps1=100,steps2=100,pop=1_000,rlmodel='acktr',
               save='saved/distrib_base_',debug=False,simut='simut',results='results/distrib_',
               deterministic=True,train=True,predict=True,batch1=1,batch2=100,cont=False,
               start_from=None,plot=False,twostage=False,callback_minsteps=None,
               stats_results='results/distrib_stats',startn=None,verbose=1,
               learning_rate=0.25,learning_schedule='linear'):
   
        '''
        run_verify

        train a model based on a protocol, and then simulate it
        plot results if needed
        '''
   
        if startn is None:
            startn=0
    
        self.n_pop=pop

        # repeat simulation n times
        for num in range(startn,n):
            bestname2=save+'_'+str(100+num)
            results2=results+'_'+str(100+num)
            print('computing {}'.format(num))
        
            self.run_results(steps1=steps1,steps2=steps2,pop=pop,rlmodel=rlmodel,
               twostage=twostage,save=bestname2,debug=debug,simut=simut,results=results2,
               deterministic=deterministic,train=train,predict=predict,
               batch1=batch1,batch2=batch2,cont=cont,start_from=start_from,plot=False,
               callback_minsteps=callback_minsteps,verbose=verbose,learning_rate=learning_rate,learning_schedule=learning_schedule)

        #self.render_distrib(load=results,n=n,stats_results=stats_results)
            
    def comp_distribs(self,load=None,n=1,startn=0,stats_results='results/distrib_stats',singlefile=False):
        if load is None:
            return
            
        self.episodestats.run_simstats(load,stats_results,n,startn=startn,singlefile=singlefile)

    def render_distrib(self,stats_results='results/distrib_stats',plot=False,figname=None):
        self.episodestats.plot_simstats(stats_results,figname=figname)

        # gather results ...
        #if plot:
        #    print('plot')
            
    def compare_distrib(self,filename1,filename2,n=1,label1='perus',label2='vaihtoehto',figname=None):
        self.episodestats.compare_simstats(filename1,filename2,label1=label1,label2=label2,figname=figname)

    def plot_rewdist(self,age=20,sum=False,all=False):
        t=self.map_age(age)
        self.episodestats.plot_rewdist(t=t,sum=sum,all=all)

    def plot_saldist(self,age=20,sum=False,all=False,n=10):
        t=self.map_age(age)
        self.episodestats.plot_saldist(t=t,sum=sum,all=all,n=n)

    def plot_RL_act(self,t,rlmodel='acktr',load='perus',debug=True,deterministic=True,
                        n_palkka=80,deltapalkka=1000,n_elake=40,deltaelake=1500,
                        hila_palkka0=0,hila_elake0=0):
        model,env,n_cpu=self.setup_model(rlmodel=rlmodel,load=load,debug=debug)
        RL_unemp=self.RL_simulate_V(model,env,t,emp=0,deterministic=deterministic,n_palkka=n_palkka,deltapalkka=deltapalkka,n_elake=n_elake,deltaelake=deltaelake,
                        hila_palkka0=hila_palkka0,hila_elake0=hila_elake0)
        RL_emp=self.RL_simulate_V(model,env,t,emp=1,deterministic=deterministic,n_palkka=n_palkka,deltapalkka=deltapalkka,n_elake=n_elake,deltaelake=deltaelake,
                        hila_palkka0=hila_palkka0,hila_elake0=hila_elake0)
        self.plot_img(RL_emp,xlabel="Pension",ylabel="Wage",title='Töissä')
        self.plot_img(RL_unemp,xlabel="Pension",ylabel="Wage",title='Työttömänä')
        self.plot_img(RL_emp-RL_unemp,xlabel="Pension",ylabel="Wage",title='Työssä-Työtön')

    def get_RL_act(self,t,emp=0,time_in_state=0,rlmodel='acktr',load='perus',debug=True,deterministic=True,
                        n_palkka=80,deltapalkka=1000,n_elake=40,deltaelake=1500,
                        hila_palkka0=0,hila_elake0=0):
        model,env,n_cpu=self.setup_model(rlmodel=rlmodel,load=load,debug=debug)
        return self.RL_simulate_V(model,env,t,emp=emp,deterministic=deterministic,time_in_state=time_in_state,
                        n_palkka=n_palkka,deltapalkka=deltapalkka,n_elake=n_elake,deltaelake=deltaelake,
                        hila_palkka0=hila_palkka0,hila_elake0=hila_elake0)

    def plot_img(self,img,xlabel="Eläke",ylabel="Palkka",title="Employed"):
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        heatmap = plt.pcolor(img) 
        plt.colorbar(heatmap)        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.show()
        
    def plot_twoimg(self,img1,img2,xlabel="Pension",ylabel="Wage",title1="Employed",title2="Employed",
                    vmin=None,vmax=None,figname=None):
        fig, axs = plt.subplots(ncols=2)
        if vmin is not None:
            im0 = axs[0].imshow(img1,vmin=vmin,vmax=vmax,cmap='gray')
        else:
            im0 = axs[0].imshow(img1)
        #heatmap = plt.pcolor(img1) 
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="20%", pad=0.05)
        cbar0 = plt.colorbar(im0, cax=cax0)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(title1)
        if vmin is not None:
            im1 = axs[1].imshow(img2,vmin=vmin,vmax=vmax,cmap='gray')
        else:
            im1 = axs[1].imshow(img2)
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="20%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)
        #heatmap = plt.pcolor(img2) 
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        axs[1].set_title(title2)
        #plt.colorbar(heatmap)        
        plt.subplots_adjust(wspace=0.3)
        if figname is not None:
            plt.savefig(figname+'.eps', format='eps')

        plt.show()
        
    def filter_act(self,act,state):
        employment_status,pension,old_wage,age,time_in_state,next_wage=self.env.state_decode(state)
        if age<self.min_retirementage:
            if act==2:
                act=0
        
        return act

    def RL_simulate_V(self,model,env,age,emp=0,time_in_state=0,deterministic=True,
                        n_palkka=80,deltapalkka=1000,n_elake=40,deltaelake=1500,
                        hila_palkka0=0,hila_elake0=0):
        # dynaamisen ohjelmoinnin parametrejä
        def map_elake(v):
            return hila_elake0+deltaelake*v # pitäisikö käyttää exp-hilaa?

        def map_palkka(v):
            return hila_palkka0+deltapalkka*v # pitäisikö käyttää exp-hilaa?
    
        prev=0
        toe=0
        fake_act=np.zeros((n_palkka,n_elake))
        for el in range(n_elake):
            for p in range(n_palkka): 
                palkka=map_palkka(p)
                elake=map_elake(el)
                if emp==2:
                    state=self.env.state_encode(emp,elake,0,age,time_in_state,0)
                elif emp==1:
                    state=self.env.state_encode(emp,elake,palkka,age,time_in_state,palkka)
                else:
                    state=self.env.state_encode(emp,elake,palkka,age,time_in_state,palkka)

                act, predstate = model.predict(state,deterministic=deterministic)
                act=self.filter_act(act,state)
                fake_act[p,el]=act
                 
        return fake_act
        
    def L2error(self):
        '''
        Laskee L2-virheen havaittuun työllisyysasteen L2-virhe/vuosi tasossa
        Käytetään optimoinnissa
        '''
        L2=self.episodestats.comp_L1error()
        
        return L2
        
    def comp_aggkannusteet(self,n=None,savefile=None):
        self.episodestats.comp_aggkannusteet(self.env.ben,n=n,savefile=savefile)
        
    def plot_aggkannusteet(self,loadfile,baseloadfile=None,figname=None,label=None,baselabel=None):
        self.episodestats.plot_aggkannusteet(self.env.ben,loadfile,baseloadfile=baseloadfile,figname=figname,
                                             label=label,baselabel=baselabel)
        
    def comp_taxratios(self,grouped=True):
        return self.episodestats.comp_taxratios(grouped=grouped)
        
    def comp_verokiila(self,grouped=True):
        return self.episodestats.comp_verokiila(grouped=grouped)
    
        