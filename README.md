# DEXTRON: DEXTerity enviRONment

DEXTRON is a stochastic environment with sparse reward function which simulates a prosthetic hand with real human transport motions. The objective of the environment is to successfully grasp the only object in the environment -- a cylinder. DEXTRON uses real data collected from 9 human subjects:

<p align="center">
  <img src="./doc/1_data_collection.gif" width="320">
</p>

## Installation

Install [Digideep](https://github.com/sharif1093/digideep) before usage.

## How are environment settings sampled in DEXTRON?

<p align="center">
  <img src="./doc/slide.jpg" width="640">
</p>

Every time DEXTRON is instantiated, the real trajectory, its offset, and its duration are sampled. However, for the sampled environment settings there may not be any policies which can achieve the reward. In order to make sure that at least one policy exists for every sampled environment setting, we run a Monte Carlo simulation on a family of parameterized policies. The following figure shows sampled settings for which at least one successful policy exists:

<p align="center">
  <img src="./doc/2_trajectories_after_mc.gif" width="480">
</p>

Now, running DEXTRON with random actions looks like:

<p align="center">
  <img src="./doc/3_dextron.gif" width="320">
</p>


## Results

The current best success rate on DEXTRON is 75% based on Soft Actor-Critic method and learning from demonstrations.

| Success Cases  | Failure Cases |
:-------------------------:|:-------------------------:
<img src="./doc/4_success_cases.gif" width="320"> | <img src="./doc/5_failure_cases.gif" width="320"> |


## Citation

```bibtex
@INPROCEEDINGS{dextron21,
  title      = "End-to-end grasping policies for human-in-the-loop robots via
                deep reinforcement learning",
  booktitle  = "2021 {IEEE} International Conference on Robotics and
                Automation ({ICRA})",
  author     = "Sharif, Mohammadreza and Erdogmus, Deniz and Amato, Christopher and Padir, Taskin",
  publisher  = "IEEE",
  year       =  {2021},
```

## License

BSD 2-clause.
