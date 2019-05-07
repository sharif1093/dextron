# Useful Snippets

```python
def initialize_episode(self, physics):
    # Printing `ctrl` data in physics.named
    from pprint import pprint
    pprint(physics.named.data.ctrl)
    pprint(physics.named.model.actuator_ctrlrange)

    # Printing `body_mass` and `body_inertia` of a certain object.
    print("--- Inertia of long_cylinder:", physics.named.model.body_inertia["long_cylinder"])
    print("--- Mass of long_cylinder:", physics.named.model.body_mass["long_cylinder"])

    # Randomize joints to start with:
    from dm_control.suite.utils import randomizers
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # Printing all elements in the physics model:
    from pprint import pprint
    pprint(dir(physics.model))

    # Printing Mocap body informations
    print("number of mocap bodies:", physics.model.nmocap)
    print("mocap_pos  for mocap:", physics.named.data.mocap_pos["mocap"])
    print("mocap_quat for mocap:", physics.named.data.mocap_quat["mocap"])
    
    # Initialize the mocap location:
    physics.named.data.mocap_pos["mocap"] = np.array([.1,.1,.1], dtype=np.float32)
```


```python
def get_observation(self, physics):
    obs = collections.OrderedDict()
    
    # Ignores horizontal position to maintain translational invariance:
    obs['position'] = physics.data.qpos[1:].copy()
    obs['position'] = physics.data.qpos[:].copy()
    obs['velocity'] = physics.data.qvel[:].copy()

    # Data is something that changes in the physics.data
    from pprint import pprint
    pprint(dir(physics.data))

    obs['mocap_pos'] = physics.data.mocap_pos[:].copy()
    obs['mocap_quat'] = physics.data.mocap_quat[:].copy()

    # What about the object??
    obs['mocap_pos'] = physics.data.mocap_pos[:].copy()
    obs['mocap_quat'] = physics.data.mocap_quat[:].copy()

    obs['xpos_object'] = physics.named.data.xpos['long_cylinder'].copy()
    obs['xquat_object'] = physics.named.data.xquat['long_cylinder'].copy()

    obs['rel_obj_hand'] = obs['mocap_pos'] - obs['xpos_object']
```


```python
def get_reward(self, physics):

    # Get data of an object
    height = physics.named.data.xipos['long_cylinder', 'z']

    # Get sensor data
    touch_data = np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])

    # Penalize early terminations
    if self.get_termination(physics) == 0.0:
        reward = -(6-physics.time())*5

    # Getting initial position of an object
    physics.named.model.qpos0['long_cylinder']
    # To see inside this object:
    print(physics.named.model.qpos0.item())

    # Using `rewards.tolerance` in calculating rewards
    standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
    if self._hopping:
        hopping = rewards.tolerance(physics.speed(),
                                    bounds=(_HOP_SPEED, float('inf')),
                                    margin=_HOP_SPEED/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
        return standing * hopping
    else:
        small_control = rewards.tolerance(physics.control(),
                                          margin=1, value_at_margin=0,
                                          sigmoid='quadratic').mean()
        small_control = (small_control + 4) / 5
        return standing * small_control
    if reward < 0:
        physics._reset_next_step = True
    
    
    if reward < -.1 or reward > 1:
        print("HERE WE ARE RESETTING ...")
        physics.reset()
        reward = 0
```
