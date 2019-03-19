# Digideep: Dextron

This project is based on [Digideep](https://github.com/sharif1093/digideep) project, and implements the environment
for solving the dexterous prosthetic hand manipulation.

## Command-line execution

```bash
# A typical training
python -m digideep.main --save-modules "dextron" --params dextron.params.default --cpanel '{"time_limit":6}'

# Loading
python -m digideep.main --play --load-checkpoint "<path-to-checkpoint>"

# Loading a saved checkpoint using its saved modules
PYTHONPATH="<path-to-session>/modules" python -m digideep.main --play --load-checkpoint "<path-to-checkpoint>"

# Visualizing model
python -m digideep.environment.play --module "dextron.zoo" --model "DMCHandGrasp-v0"

```

## Parameters important in exploration/exploitation trade-off

* `entropy_coef`: The more this coefficient, the more the exploration is encouraged. It should be balanced with the rest of the losses. If it is too high,
  random actions are encouraged and the policy won't learn. If it is too low, only little explorarion is encouraged and possibly we'll get stuck in a local
  minimum. *Look at the loss values to find a good coefficient.*
    * See "https://github.com/dennybritz/reinforcement-learning/issues/34".

> Entropy loss is a clever and simple mechanism to encourage the agent to explore by providing a loss parameter that teaches the network to avoid very 
> confident predictions. As the distribution of the predictions becomes more spread out, the network will sample those moves more often and learn that 
> they can lead to greater reward.

* `num_workers`, `n_steps`, ``: Exploration, i.e. the amount of data that is explored under current policy.
* `n_update`, `num_mini_batches`: Exploitation, i.e. the amount of updates on the policy given current explorarion status. Also how to use
  them (mini-batches) and how many times to update in batches of what size?

---

* When exploration is too much, and exploitation is little, overfitting will happen. The `value_loss` will decrease but no policy is learned.
* When exploitation is too much, and exploration is little, ...


## What we can know from inspection of an RL agent getting trained

* From `/update/value_loss` we can see if the policy is become stable or not. If it becomes stable (loss ~ 0) and still the learned
  policy is not good, we should retrain with new parameter set. If the problem is weak exploration, we should increase `entropy_coef`
  so exploration is increased.
* When there is little randomness in the environments (and agent), decrease the number of parallel workers. It may cause to overfit to data.
    * Randomize openness of the hand. Start from arbitrary positions.
    * Randomize approach direction.
    * Randomize start position.

* If the capacity of the policy network is too big, then it will easily overfit to data. With simpler (smaller) networks there is less
  chance of overfitting.

* Use a regularizer to increase generalization of the model and lower the possibility of overfitting.

