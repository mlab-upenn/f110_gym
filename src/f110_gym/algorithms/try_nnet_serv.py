from __future__ import print_function
from f110_gym.distributed.exp_server import ExperienceServer
import cv2, random, threading, msgpack, os
from nnet.Online import Online
from nnet.Metric_Visualizer import Metric_Visualizer
from nnet.Trainer import Trainer
from functools import partial
import msgpack_numpy as m
import numpy as np
from steps import session

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

def deserialize_obs():
    def _deser(multipart_msg):
        lidar = msgpack.loads(multipart_msg[0], encoding="utf-8")
        steer = msgpack.unpackb(multipart_msg[1], encoding="utf-8")
        md = msgpack.unpackb(multipart_msg[2])
        cv_img = multipart_msg[3]
        cv_img = np.frombuffer(cv_img, dtype=md[b'dtype'])
        cv_img = cv_img.reshape(md[b'shape'])
        obs_dict = {"img":cv_img, "lidar":lidar, "steer":steer}
        return obs_dict
    return _deser


def batch_callback(exp_path, obs_array):
    """Takes the deserialized obs_array and performs simple training operations on it
    REMEMBER: You need to send a list back.
    """
    online_learner = Online()
    vis = Metric_Visualizer()
    pkl_name = online_learner.save_obsarray_to_pickle(obs_array, os.path.join(exp_path, 'raw'))
    vis.vid_from_pklpath(os.path.join(exp_path, 'raw', pkl_name), 0, 0, show_steer=True, units='rad', live=True)
    trainer = Trainer(online=True, pklpath=os.path.join(exp_path, 'raw', pkl_name), train_id=0)
    #Send model back
    modelpath = trainer.get_model_path()
    with open(modelpath, 'rb') as binary_file:
        model_dump = bytes(binary_file.read())
    return [model_dump]

def get_exp_path():
    exp_path = os.path.join(session["params"]["abs_path"], session["params"]["sess_root"], str(session["online"]["sess_id"]), "exp")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    funclist = session["online"]["funclist"]
    print("EXPERIENCE PATH:", exp_path)
    return exp_path

exp_path = get_exp_path()
cb = partial(batch_callback, exp_path)
serv = ExperienceServer(cb, deserialize_obs(), 4)
serv.start()
serv.join()