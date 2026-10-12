import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import random
import gymnasium as gym
import numpy as np
import gymnasium_stag_hunt
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import time

class TFStagHuntWrapper(py_environment.PyEnvironment):
    """
    Native TF-Agents Environment for Multi-Agent Parameter Sharing.
    Optionally includes handcrafted geometric features for stags and plants.
    """
    def __init__(self, env_name='StagHunt-Hunt-v0', **kwargs):
        """Creates a TF-Agents adapter around a two-agent Gymnasium game."""
        super().__init__()
        self.env_name = env_name

        # Separate wrapper-only options from arguments passed to Gymnasium.
        self._handcrafted_features = kwargs.pop('handcrafted_features', False)
        
        configs = {
            'obs_type': 'coords',
            'enable_multiagent': True,
        }
        
        self.render_mode = kwargs.pop('render_mode', None)
        self._initial_seed = kwargs.pop('seed', None)
        configs.update(kwargs)

        # Create the raw game and declare per-agent TF-Agents tensor shapes.
        self.env = gym.make(env_name, **configs)
        self.last_info = {}
        
        raw_feature_size = self.env.observation_space.shape[-1]
        self._feature_size = len(
            self._transform_observation(np.zeros(raw_feature_size, dtype=np.float32))
        )
        self._observation_spec = array_spec.ArraySpec(
            shape=(self._feature_size,),
            dtype=np.float32,
            name='observation'
        )
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=4,
            name='action'
        )

    @property
    def batched(self):
        """Indicates that this wrapper batches observations by agent."""
        return True
        
    @property
    def batch_size(self):
        """Returns the number of agents represented in each observation batch."""
        return 2

    def action_spec(self):
        """Returns the scalar discrete action specification for one agent."""
        return self._action_spec

    def observation_spec(self):
        """Returns the feature specification for one agent's observation."""
        return self._observation_spec

    def _inject_geometric_radar(self, flat_observation):
        """
        Computes complete Manhattan distances and directional deltas.
        """
        # Normalize coordinates first so signed direction deltas remain valid.
        flat_observation = flat_observation.astype(np.float32)

        MioX, MioY = flat_observation[0], flat_observation[1]
        SocioX, SocioY = flat_observation[2], flat_observation[3]
        StagX, StagY = flat_observation[4], flat_observation[5]

        # Measure each agent's distance and direction relative to the stag.
        dist_Mio_Stag = abs(MioX - StagX) + abs(MioY - StagY)
        dist_Socio_Stag = abs(SocioX - StagX) + abs(SocioY - StagY)
        
        dx_Mio_Stag = StagX - MioX
        dy_Mio_Stag = StagY - MioY
        dx_Socio_Stag = StagX - SocioX
        dy_Socio_Stag = StagY - SocioY

        # Add partner-relative geometry shared by both agent observations.
        dist_Mio_Socio = abs(MioX - SocioX) + abs(MioY - SocioY)
        dx_Mio_Socio = SocioX - MioX
        dy_Mio_Socio = SocioY - MioY

        radar_base = [
            dist_Mio_Stag, dist_Socio_Stag, dx_Mio_Stag, dy_Mio_Stag, dx_Socio_Stag, dy_Socio_Stag,
            dist_Mio_Socio, dx_Mio_Socio, dy_Mio_Socio
        ]

        # Encode each available plant, using zero slots for absent plants.
        plant_radar = []
        
        for idx in range(6, min(10, len(flat_observation)), 2):
            plant_x = flat_observation[idx]
            plant_y = flat_observation[idx + 1]
            
            # Fill the six slots with zeros when a plant is absent.
            if plant_x > 10 or plant_y > 10:
                plant_radar.extend([0.0] * 6)
            else:
                dist_Mio_Pianta = abs(MioX - plant_x) + abs(MioY - plant_y)
                dist_Socio_Pianta = abs(SocioX - plant_x) + abs(SocioY - plant_y)
                
                dx_Mio_Pianta = plant_x - MioX
                dy_Mio_Pianta = plant_y - MioY
                
                dx_Socio_Pianta = plant_x - SocioX
                dy_Socio_Pianta = plant_y - SocioY
                
                plant_radar.extend([
                    dist_Mio_Pianta, dist_Socio_Pianta,
                    dx_Mio_Pianta, dy_Mio_Pianta,
                    dx_Socio_Pianta, dy_Socio_Pianta
                ])

        # Pad to a fixed feature count so network input shapes stay constant.
        while len(plant_radar) < 12:
            plant_radar.extend([0.0] * 6)

        radar_features = np.array(radar_base + plant_radar, dtype=np.float32)

        # Preserve the raw observation and append all derived radar values.
        return np.concatenate([flat_observation, radar_features], axis=0)

    @staticmethod
    def _relative(me, target):
        """Manhattan distance and directional deltas from `me` to `target`."""
        dx, dy = target[0] - me[0], target[1] - me[1]
        return [abs(dx) + abs(dy), dx, dy]

    def _inject_harvest_radar(self, flat_observation):
        """Partner radar, one fixed slot per plant, nearest mature/young plant."""
        # Parse the agent pair and initialize nearest-plant tracking by maturity.
        obs = flat_observation.astype(np.float32)
        me, partner = obs[0:2], obs[2:4]
        radar = self._relative(me, partner)
        nearest = {0: None, 1: None}
        for start in range(4, len(obs), 3):
            plant = obs[start:start + 2]
            mature = int(obs[start + 2])
            mine = self._relative(me, plant)
            radar += mine + self._relative(partner, plant)
            if nearest[mature] is None or mine[0] < nearest[mature][0]:
                nearest[mature] = mine
        # Append nearest-plant features, using a sentinel when a type is absent.
        for mature in (1, 0):
            radar += nearest[mature] if nearest[mature] is not None else [-1.0, 0.0, 0.0]
        return np.concatenate([obs, np.asarray(radar, dtype=np.float32)], axis=0)

    def _inject_escalation_radar(self, flat_observation):
        """Partner radar plus mark radar for both agents."""
        # Keep the raw coordinates and append pairwise distance/direction values.
        obs = flat_observation.astype(np.float32)
        me, partner, mark = obs[0:2], obs[2:4], obs[4:6]
        radar = self._relative(me, partner) + self._relative(me, mark) + self._relative(partner, mark)
        return np.concatenate([obs, np.asarray(radar, dtype=np.float32)], axis=0)

    def _transform_observation(self, observation):
        """Adds configured handcrafted features to an observation."""
        # Select the feature encoder for the active game when extras are enabled.
        if self._handcrafted_features:
            if self.env_name == 'StagHunt-Hunt-v0':
                return self._inject_geometric_radar(observation)
            if self.env_name == 'StagHunt-Harvest-v0':
                return self._inject_harvest_radar(np.asarray(observation))
            if self.env_name == 'StagHunt-Escalation-v0':
                return self._inject_escalation_radar(np.asarray(observation))
        # Always expose numeric observations in the dtype declared by the spec.
        return np.asarray(observation, dtype=np.float32)
    
    def _reset(self):
        """Resets the game and returns both agents' transformed observations."""
        # Apply the configured seed only on the first reset of this wrapper.
        if self._initial_seed is not None:
            random.seed(self._initial_seed)
            np.random.seed(self._initial_seed)
            obs, info = self.env.reset(seed=self._initial_seed)
            self._initial_seed = None
        else:
            obs, info = self.env.reset()
        self.last_info = info
        #time.sleep(1) 
        
        # Transform each agent's view and package the initial batched timestep.
        obs_A_enhanced = self._transform_observation(obs[0])
        obs_B_enhanced = self._transform_observation(obs[1])
        
        obs_stacked = np.array([obs_A_enhanced, obs_B_enhanced], dtype=np.float32)
        return ts.restart(obs_stacked, batch_size=2)

    def _step(self, action):
        """Applies both agents' actions and packages the resulting time step."""
        # Run one joint game step, then transform the observations independently.
        next_obs, rewards, term, trunc, info = self.env.step(action.tolist())
        self.last_info = info
        
        obs_A_enhanced = self._transform_observation(next_obs[0])
        obs_B_enhanced = self._transform_observation(next_obs[1])

        obs_stacked = np.array([obs_A_enhanced, obs_B_enhanced], dtype=np.float32)
        rewards_stacked = np.array([float(rewards[0]), float(rewards[1])], dtype=np.float32)
        
        # Map Gymnasium termination/truncation flags to TF-Agents step fields.
        done = term or trunc
        
        if done:
            step_type = np.array([ts.StepType.LAST, ts.StepType.LAST], dtype=np.int32)
            discount = np.array([0.0, 0.0], dtype=np.float32)
        else:
            step_type = np.array([ts.StepType.MID, ts.StepType.MID], dtype=np.int32)
            discount = np.array([1.0, 1.0], dtype=np.float32)
            
        return ts.TimeStep(
            step_type=step_type, 
            reward=rewards_stacked, 
            discount=discount, 
            observation=obs_stacked
        )
    
    def set_stag_run_away_after_maul(self, value: bool):
        """Sets whether the Hunt stag flees after a mauling."""
        # Ignore this Hunt-specific option for the other registered games.
        if self.env_name != 'StagHunt-Hunt-v0':
            return

        # Update the outer wrapper when possible, otherwise mutate the base game.
        if hasattr(self.env, 'run_away_after_maul'):
            self.env.run_away_after_maul = value
            print("run_away_after_maul setted")
        elif hasattr(self.env.unwrapped, 'run_away_after_maul'):
            print("run_away_after_maul setted")
            self.env.unwrapped.run_away_after_maul = value

    def set_stag_follow_probability(self, probability):
        """Sets the Hunt stag's probability of following an agent."""
        # Forward the schedule value to the underlying Hunt game only.
        if self.env_name != 'StagHunt-Hunt-v0':
            return
        self.env.unwrapped.set_stag_follow_probability(probability)


class ParallelStagHuntEnv(TFStagHuntWrapper):
    """One two-agent game suitable for a ParallelPyEnvironment worker."""

    @property
    def batched(self):
        """Indicates that each worker exposes one complete two-agent game."""
        return False

    def observation_spec(self):
        """Returns the observation specification for both agents in one game."""
        return array_spec.ArraySpec(
            shape=(2, self._feature_size), dtype=np.float32, name='observation'
        )

    def action_spec(self):
        """Returns the vector action specification for both agents."""
        return array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum=4, name='action'
        )

    def reward_spec(self):
        """Returns the vector reward specification for both agents."""
        return array_spec.ArraySpec(shape=(2,), dtype=np.float32, name='reward')

    def _reset(self):
        """Resets one parallel game and returns both agents' observations."""
        # Seed each worker's first episode independently, then leave later resets unseeded.
        if self._initial_seed is not None:
            random.seed(self._initial_seed)
            np.random.seed(self._initial_seed)
            obs, _ = self.env.reset(seed=self._initial_seed)
            self._initial_seed = None
        else:
            obs, _ = self.env.reset()

        # Stack both agent views into the vector layout expected by a worker.
        observations = np.array([
            self._transform_observation(obs[0]),
            self._transform_observation(obs[1]),
        ], dtype=np.float32)
        return ts.TimeStep(
            step_type=np.int32(ts.StepType.FIRST),
            reward=np.zeros(2, dtype=np.float32),
            discount=np.float32(1.0),
            observation=observations,
        )

    def _step(self, action):
        """Applies a vector of agent actions and returns the game time step."""
        # Apply both worker actions together and convert results to declared dtypes.
        next_obs, rewards, term, trunc, _ = self.env.step(
            np.asarray(action, dtype=np.int32).tolist()
        )
        observations = np.array([
            self._transform_observation(next_obs[0]),
            self._transform_observation(next_obs[1]),
        ], dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)

        # Emit a terminal or continuing scalar step type for this whole game.
        if term or trunc:
            return ts.TimeStep(
                step_type=np.int32(ts.StepType.LAST),
                reward=rewards,
                discount=np.float32(0.0),
                observation=observations,
            )
        return ts.TimeStep(
            step_type=np.int32(ts.StepType.MID),
            reward=rewards,
            discount=np.float32(1.0),
            observation=observations,
        )