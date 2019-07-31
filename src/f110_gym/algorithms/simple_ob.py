#!/usr/bin/env python
from __future__ import print_function
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
from f110_gym.distributed.exp_sender import ExperienceSender
import rospy, cv2, random, threading
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

def update_nn(reply):
    print(reply)

def main():
    env = f110Env()
    obs = env.reset()
    obs_array = []
    sender = ExperienceSender()
    cnt = 0
    while True:
        random_action = {"angle":0.2, "speed":1.0}
        obs, reward, done, info = env.step(random_action)
        if info.get("record"):
            if cnt % 100 == 0:
                obs_array.append(obs)
            cnt+=1
        if len(obs_array) % 8 == 0 and len(obs_array) > 0:
            sender.send_obs(obs_array, env.serialize_obs(), update_nn, header_dict={'env':'f110Env'})
            obs_array = []
        if done:
            obs = env.reset()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.signal_shutdown('Done')
        pass