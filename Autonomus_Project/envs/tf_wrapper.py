import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
    Includes handcrafted features (complete geometric radar for stags and plants).
    """
    def __init__(self, env_name='StagHunt-Hunt-v0', **kwargs):
        super().__init__()
        
        configs = {
            'obs_type': 'coords',
            'enable_multiagent': True,
        }
        
        self.render_mode = kwargs.pop('render_mode', None)
        configs.update(kwargs)

        self.env = gym.make(env_name, **configs)
        
        # The array expands from 10 to 31 entries for the complete vector mapping.
        # 10 (base) + 6 (stag) + 3 (partner) + 6 (plant 1) + 6 (plant 2) = 31
        self._observation_spec = array_spec.ArraySpec(
            shape=(31,),
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
        return True
        
    @property
    def batch_size(self):
        return 2

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _inject_geometric_radar(self, flat_observation):
        """
        Computes complete Manhattan distances and directional deltas.
        """
        # Convert to float so negative directional values are preserved.
        flat_observation = flat_observation.astype(np.float32)

        MioX, MioY = flat_observation[0], flat_observation[1]
        SocioX, SocioY = flat_observation[2], flat_observation[3]
        StagX, StagY = flat_observation[4], flat_observation[5]

        # --- A. STAG RADAR ---
        dist_Mio_Stag = abs(MioX - StagX) + abs(MioY - StagY)
        dist_Socio_Stag = abs(SocioX - StagX) + abs(SocioY - StagY)
        
        dx_Mio_Stag = StagX - MioX
        dy_Mio_Stag = StagY - MioY
        dx_Socio_Stag = StagX - SocioX
        dy_Socio_Stag = StagY - SocioY

        # --- B. PARTNER RADAR ---
        dist_Mio_Socio = abs(MioX - SocioX) + abs(MioY - SocioY)
        dx_Mio_Socio = SocioX - MioX
        dy_Mio_Socio = SocioY - MioY

        radar_base = [
            dist_Mio_Stag, dist_Socio_Stag, dx_Mio_Stag, dy_Mio_Stag, dx_Socio_Stag, dy_Socio_Stag,
            dist_Mio_Socio, dx_Mio_Socio, dy_Mio_Socio
        ]

        # --- C. PLANT RADAR ---
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

        # Fill missing plants to reach exactly 12 slots (6 x 2).
        while len(plant_radar) < 12:
            plant_radar.extend([0.0] * 6)

        radar_features = np.array(radar_base + plant_radar, dtype=np.float32)

        # Concatenate the original 10 features with the 21 radar features.
        return np.concatenate([flat_observation, radar_features], axis=0)
    
    def _reset(self):
        obs, info = self.env.reset()
        #time.sleep(1) 
        
        obs_A_enhanced = self._inject_geometric_radar(obs[0])
        obs_B_enhanced = self._inject_geometric_radar(obs[1])
        
        obs_stacked = np.array([obs_A_enhanced, obs_B_enhanced], dtype=np.float32)
        return ts.restart(obs_stacked, batch_size=2)

    def _step(self, action):
        next_obs, rewards, term, trunc, info = self.env.step(action.tolist())
        
        obs_A_enhanced = self._inject_geometric_radar(next_obs[0])
        obs_B_enhanced = self._inject_geometric_radar(next_obs[1])

        obs_stacked = np.array([obs_A_enhanced, obs_B_enhanced], dtype=np.float32)
        rewards_stacked = np.array([float(rewards[0]), float(rewards[1])], dtype=np.float32)
        
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
        if hasattr(self.env, 'run_away_after_maul'):
            self.env.run_away_after_maul = value
            print("run_away_after_maul setted")
        elif hasattr(self.env.unwrapped, 'run_away_after_maul'):
            print("run_away_after_maul setted")
            self.env.unwrapped.run_away_after_maul = value