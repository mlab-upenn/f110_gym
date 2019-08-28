from __future__ import print_function
import os, sys, cv2, math, time
import numpy as np
from collections import deque

import airsim

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'


class Env(object):
    """
    Stripped down version from OpenaiGym
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None
    ser_msg_length = 0

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self, **kwargs):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def serialize_obs(self):
        """Returns a function that allows you to serialize each observation as a multipart"""
        raise NotImplementedError
    
class SIM_f110Env(Env):
    """
    Implements a Gym Environment & neccessary funcs for the F110 Autonomous Car on Microsfot Airsim
    """
    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True) 
        self.history = deque(maxlen=500) #for reversing during reset

        #GYM Properties (set in subclasses)
        self.observation_space = ['lidar', 'steer', 'img']
        self.action_space = ['angle', 'speed']

    def _get_imgs(self, labels=["0"]):
        label_to_func = lambda lbl: airsim.ImageRequest(lbl, airsim.ImageType.Scene, False, False)
        bytestr_to_np = lambda rep: np.fromstring(rep.image_data_uint8, dtype=np.uint8).reshape(rep.height, rep.width, 4)[0:2]

        responses = self.client.simGetImages(map(label_to_func, labels))
        images = map(bytestr_to_np, responses)
        return images
    
    def add_to_history(self, data):
        if abs(data.drive.steering_angle) > 0.05 and data.drive.steering_angle < -0.05 and data.drive.steering_angle is not None:
            steer_dict = {"angle":data.drive.steering_angle, "speed":data.drive.speed}
            for i in range(40):
                self.history.append(steer_dict)

    def _get_obs(self):
        #Get Camera imgs
        imgs = self._get_imgs()

        #Get LiDAR reading & transform it to look planar
        lidarData = self.client.getLidarData()

        #Get steer data (dummy for now)
        steer = {"angle": 0.0, "steering_angle_velocity": 0.0, "speed": 0.0}
        latest_dict = {'lidar': lidarData, 'steer': steer, 'img':imgs}
        self.add_to_history(steer)
        return latest_dict

    def reset(self, **kwargs):
        """
        Reset to initial position
        """
        self.client.reset()

    def get_reward(self):
        """
        TODO:Implement reward functionality
        """
        return 0

    def step(self, action):
        """
        Action should be a steer_dict = {"angle":float, "speed":float}
        """
        car_controls = airsim.CarControls()
        car_controls.throttle = action.get("speed")
        car_controls.steering = action.get("angle")

        #execute action
        self.client.setCarControls(car_controls)
        time.sleep(0.01)

        #get reward & check if done & return
        obs = self._get_obs()
        reward = self.get_reward()
        done = self.tooclose()
        info = {}
        return obs, reward, done,info
    
    def tooclose(self):
        """
        Uses latest_obs to determine if we are too_close (currently uses LIDAR)
        """
        return False