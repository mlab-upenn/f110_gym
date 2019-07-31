#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import zmq, msgpack
from threading import Thread
import msgpack_numpy as m

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceSender():
    """ Opens zmq DEALER socket & sends 'experiences' over from the environment
    """
    def __init__(self, connect_to="tcp://195.0.0.3:5555"):

        #important zmq initialization stuff to connect to server
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_socket.connect(connect_to)
        myid = b'0'
        self.zmq_socket.identity = myid.encode('ascii')
        self.batchnum = 0
        self.recv_loop_running = False
        m.patch()
        
    def obs_to_dump(self, obs_array, serial_func):
        dump_array = []
        for obs_dict in obs_array:
            dump_array += serial_func(obs_dict)
        return dump_array
    
    def recv(self, recv_callback, wait_for_recv):
        """Polls ROUTER repeatedly for a message
        """
        poll = zmq.Poller()
        poll.register(self.zmq_socket, zmq.POLLIN)
        while True:
            sockets = dict(poll.poll(1000))
            if self.zmq_socket in sockets:
                msg = self.zmq_socket.recv_multipart()
                header_dict = msgpack.loads(msg[0], encoding="utf-8")
                print("\n RECVD NN FOR BATCH %s" % header_dict.get("batchnum"))
                recv_callback(msg[1:])
                if wait_for_recv:
                    self.recv_loop_running = False
                    break

    def send_obs(self, obs_array, serial_func, recv_callback, header_dict = {}, wait_for_recv=False):
        """Sends an observation to server
        recv_callback is a func to be executed on reply, obs_array is an array of obs_dicts, serial_func is the function used to serialize each dict in the obs_array
        """
        dump_array = self.obs_to_dump(obs_array, serial_func)
        header_dict['batchnum'] = self.batchnum
        header_dump = [msgpack.dumps(header_dict)]
        dump_array = header_dump + dump_array
        print("\n SENT BATCH: %s" % self.batchnum)
        self.zmq_socket.send_multipart(dump_array, copy=False)
        if not self.recv_loop_running:    
            p = Thread(target=self.recv, args=(recv_callback, wait_for_recv))
            p.daemon = True
            p.start()
            self.recv_loop_running = True
        self.batchnum += 1
