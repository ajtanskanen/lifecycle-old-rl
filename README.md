# Life cycle model

The life cycle model describes agents making optimal decisions in the framework of 
Finnish social security. The main interest in this setting is how changes in the social security
impacts employment. 

The library depends on separate econogym and on benefits modules that implement
various states of agents and the benefits of social security scheme. The optimal 
behavior of the agents is solved using Reinforcement Learning library stable baselines.

The library reproduces the observed employment rates in Finland quite well, at all ages
from 20-70 separately for women and men. 

The model is written in Python.

Description of the lifecycle model can be found (in Finnish!) from articles (Tanskanen, 2019; Tanskanen, 2020).


## References

	@misc{lifecycle_rl_,
	  author = {Antti J. Tanskanen},
	  title = {Elinkaarimalli},
	  year = {2019},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/ajtanskanen/lifecycle_rl}},
	}

The library is described in articles

    @article{tanskanen2020deep,
      title={Deep reinforced learning enables solving discrete-choice life cycle models to analyze social security reforms},
      author={Tanskanen, Antti J},
      journal={arXiv preprint arXiv:2010.13471},
      year={2020}
    }
	
	@misc{lifecycle_rl_kak,
	  author = {Antti J. Tanskanen},
	  title = {Unelmoivatko robotit ansiosidonnaisesta sosiaaliturvasta?},
	  year = {2019},
	  publisher = {},
	  journal = {KAK},
	  howpublished = {TBD},
	}	