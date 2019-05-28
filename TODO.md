# TODO list of project digideep & dextron

## Research:

- [ ] Connect the demonstration part to the memory.
- [ ] Make the agent not generate action when in learning from demo mode (to save computational resources.) (Es: 6)
- [ ] Train using demonstrations + SAC.
- [ ] SAC:
    - [ ] Multi-step (Monte Carlo like) updates.
    - [ ] Double Q networks with targets.
- [ ] Eliminating the initial noise at the beginning of each environment. (Kinda fast transition to the randomized initial states.)

## Documentations

- [ ] Write the documentation for Normal vs. Vector wrappers:
    - Vector wrappers can be serializable: So observation/reward normalization cannot be done as a Normal wrapper but only a Vector wrapper. However action normalizer can be done in both.
    - Normal wrappers cannot be serializable.
    - Normal wrappers are isolated (they operate in other threads) and cannot communicate any further information rather than `(obs,rew,done,info)`.

## New methods

- [ ] Implement HER (now that observation as a dict capability exists it would be easy.)

## Enhancements

**Wrappers**

- [ ] Implement `WrapperAddTimeStepDict` for MuJoCo environments.
- [ ] Implement `WrapperMaskObsDict` which will be used to mask a path in the observation dictionary.

**Visualization**

- [ ] Online plotting of loss functions (through visdom/tensorboard): use log files. (Es: 5)
- [ ] Ready functions to plot useful plots easily and automatically place them in `./plots` folder.

## Refactoring and restructuring

- [ ] Refactor parameter files to share codes between param files.
    - [ ] Have `cpanel` for each dictionary. Have an input argument which `cpanel` items should be exposed. Have an option for `remapping` the `cpanel` items.
- [ ] Wrap `utility/plotting.py` with `visdom_engine`.
- [ ] Refactor `utility/stats.py` to produce a log file instead of using visdom directly. The log file should be the same style as `monitoring.py`.

