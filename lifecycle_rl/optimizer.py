'''

Bayesian optimization for lifecycle models

The aim is to reproduce employment rate at each age

'''

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from pathlib import Path
from os import path
import numpy as np

from .lifecycle import Lifecycle

class OptimizeLifecycle():
    def __init__(self,initargs=None,runargs=None):
        '''
        Alusta muuttujat
        '''
        self.runargs=runargs
        self.lf=Lifecycle(**initargs)
        
    def black_box_function(self,**x):
        """Function with unknown internals we wish to maximize.

        
        """
        print(x)
        self.lf.env.set_utility_params(**x)
        self.lf.run_results(**self.runargs)
        return -self.lf.L2error()

    def optimize(self,reset=False):
        # Bounded region of parameter space
        pbounds = {'men_kappa_fulltime': (0.3, 0.6), 'men_kappa_parttime': (0.3, 0.7), #'men_mu_scale': (0.01,0.3), 'men_mu_age': (57,62),
                   'women_kappa_fulltime': (0.3, 0.6), 'women_kappa_parttime': (0.3, 0.7)} #, 'women_mu_scale': (0.01,0.3), 'women_mu_age': (57,62)}

        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        
        LOG_DIR = Path().absolute() / 'bayes_opt_logs'
        LOG_DIR.mkdir(exist_ok=True)
        filename = 'log_0.json'

        # talletus
        logfile=str(LOG_DIR / filename)
        logger = JSONLogger(path=logfile)
        if Path(logfile).exists() and not reset:
            load_logs(optimizer, logs=[logfile]);
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=2,
            n_iter=20,
        )
        
        print('The best parameters found {}'.format(optimizer.max))
        
class BalanceLifeCycle():
    def __init__(self,initargs=None,runargs=None,ref_muut=9.5e9,additional_income_tax=0):
        '''
        Alusta muuttujat
        '''
        self.runargs=runargs
        self.initargs=initargs
        self.ref_muut=ref_muut
        self.additional_income_tax=additional_income_tax
        
    def black_box_function(self,**x):
        """
        Function with unknown internals we wish to maximize.
        """
        print(x)
        initargs2=self.initargs
        initargs2['extra_ppr']=x['extra_ppr']
        initargs2['additional_income_tax']=self.additional_income_tax
        repeats=1
        err=np.empty(repeats)
        for r in range(repeats):
            cc=Lifecycle(**initargs2)
            cc.run_results(**self.runargs)
            err[r]=cc.L2BudgetError(self.ref_muut)
            
        ave=np.nanmean(err)
        print(f'ave {ave}')
            
        return ave

    def test_black_box_function(self,**x):
        """
        Function with unknown internals we wish to maximize.
        """
        print(x)
        initargs2=self.initargs
        initargs2['extra_ppr']=x['extra_ppr']
        initargs2['additional_income_tax']=self.additional_income_tax
        repeats=1
        err=np.empty(repeats)
        for r in range(repeats):
            e=np.random.normal(0,0.05)
            err[r]=-(x['extra_ppr']+e-0.1)**2
            
        ave=np.nanmean(err)
        print(f'ave {ave}')
            
        return ave

    def optimize(self,reset=False,min_ppr=-0.3,max_ppr=0.3,debug=False):
        # Bounded region of parameter space
        pbounds = {'extra_ppr': (min_ppr, max_ppr)} #, 'women_mu_scale': (0.01,0.3), 'women_mu_age': (57,62)}

        if debug:
            optimizer = BayesianOptimization(
                f=self.test_black_box_function,
                pbounds=pbounds,
                verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                #random_state=1,
            )
        else:
            optimizer = BayesianOptimization(
                f=self.black_box_function,
                pbounds=pbounds,
                verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                #random_state=1,
            )
        
        LOG_DIR = Path().absolute() / 'bayes_opt_logs'
        LOG_DIR.mkdir(exist_ok=True)
        num=int(self.additional_income_tax*100)
        filename = 'log_'+str(num)+'.json'

        # talletus
        logfile=str(LOG_DIR / filename)
        logger = JSONLogger(path=logfile)
        if Path(logfile).exists() and not reset:
            load_logs(optimizer, logs=[logfile]);
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=2,
            n_iter=60,
        )
        
        print('The best parameters found {}'.format(optimizer.max))
        
        return optimizer.max

