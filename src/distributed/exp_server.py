import cv2, msgpack, os, time, threading, zmq
import numpy as np
import msgpack_numpy as m

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, recv_callback, deser_func, deser_length, open_port='tcp://*:5555'):
        """Opens a zmq.ROUTER to recv batches of 'experiences' from F110 & process them""" 
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_socket.bind(open_port)
        self.recv_callback = recv_callback
        self.deser_func = deser_func
        self.deser_length = deser_length
        m.patch()
        threading.Thread.__init__(self)

    def dump_to_obs(self, msg_dump_arr):
        """Convert a multipart msg dump into an array of obs_dicts
        """
        obs_array = []
        for i in range(len(msg_dump_arr)):
            if i % self.deser_length == 0:
                obs_dict = self.deser_func(msg_dump_arr[i:i + self.deser_length])
                obs_array.append(obs_dict)
        return obs_array

    def run(self):
        print("RUN")
        while True:
            msg_dump_arr = self.zmq_socket.recv_multipart()
            header_dict = msgpack.loads(msg_dump_arr[1], encoding="utf-8")
            print('RECV BATCH:', header_dict.get("batchnum"))
            obs_array = self.dump_to_obs(msg_dump_arr[2:])
            reply_dump_array = self.recv_callback(obs_array)
            dump_array = [msg_dump_arr[0], msg_dump_arr[1]] + reply_dump_array
            print('SENT BATCH:', header_dict.get("batchnum"))
            self.zmq_socket.send_multipart(dump_array)