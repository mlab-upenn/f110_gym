from __future__ import print_function
import os, sys, cv2, math, time
import numpy as np
from f110_core import Env
from collections import deque

import pdb

import airsim

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

dt = 0.01
class CarParams(object):
    def __init__(self):
        self.wheelbase = .3302
        self.friction_coeff = .523
        self.h_cg = .074
        self.l_f = .15875
        self.l_r = .17145
        self.cs_f = 4.718
        self.cs_r = 5.4562
        self.mass = 3.46
        self.I_z = .04712

        #Limits
        self.max_accel = 7.51
        self.max_decel = 8.26
        self.max_speed = 7.
        self.max_steering_vel = 3.2
        self.max_steering_angle = 0.4189

class Car(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.pitch = 0
        self.roll = 0
        self.theta = 0
        self.velocity = 0
        self.steer_angle = 0
        self.angular_velocity = 0
        self.slip_angle = 0
        self.st_dyn = False
        self.params = CarParams()
    

    def init_state_from_pose(self, pose):
        curr_position = pose.position #Vector 3r
        curr_orientation = pose.orientation

        #Set State Variables
        self.x = curr_position.x_val
        self.y = -1.0 * curr_position.y_val
        self.z = curr_position.z_val

        curr_pitch, curr_roll, curr_yaw = airsim.utils.to_eularian_angles(curr_orientation)
        self.pitch = curr_pitch
        self.roll = curr_roll
        self.theta = curr_yaw

    def convert_state_to_pose(self):
        new_position = airsim.Vector3r(self.x, self.y, self.z)
        new_orientation = airsim.utils.to_quaternion(self.pitch, self.roll, self.theta)
        pose = airsim.Pose(new_position, new_orientation)
        return pose

    def compute_steer_vel(self, desired_angle):
        diff = desired_angle - self.steer_angle
        
        #Calculate Velocity
        steer_vel = 0
        if abs(diff) > 0.0001:
            steer_vel = diff / abs(diff) * self.max_steering_vel()
        else:
            steer_vel = 0
        return steer_vel

    def compute_accel(self, desired_velocity):
        diff = desired_velocity - self.velocity
        accel = 0
        if self.velocity > 0: #fwd
            if diff > 0: #accelerate
                kp = 2.0 * self.max_accel() / self.max_speed()
                accel = kp * diff
    
            else: #brake
                accel = -1.0 * self.max_decel()
        else: #reverse
            if diff > 0: #brake
                accel = self.max_decel()
            else:
                kp = 2.0 * self.max_accel() / self.max_speed()
                accel = kp * diff
        return accel

    def update_k(self, accel, steer_angle_vel):
        x_dot = self.velocity * math.cos(self.theta)
        y_dot = self.velocity * math.sin(self.theta)
        v_dot = accel
        steer_angle_dot = steer_angle_vel
        theta_dot = self.velocity / self.params.wheelbase * math.tan(self.steer_angle)
        theta_double_dot = accel / self.params.wheelbase * math.tan(self.steer_angle) + self.velocity * steer_angle_vel / (self.params.wheelbase * math.cos(self.steer_angle)**2)
        slip_angle_dot = 0

        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += theta_dot * dt
        self.velocity += v_dot * dt
        self.steer_angle += steer_angle_dot * dt
        self.angular_velocity = 0
        self.slip_angle = 0
        self.st_dyn = False

    def update_pose(self, accel, steer_angle_vel):
        thresh = .5
        err = .03
        if not self.st_dyn:
            thresh += err
        
        #if velocity is low or negative, use normal kinematic single track dynamics
        if self.velocity < thresh:
            self.update_k(accel, steer_angle_vel)
            return

        g = 9.81 #m/s^2

        # Compute first derivatives of state
        x_dot = self.velocity * math.cos(self.theta + self.slip_angle)
        y_dot = self.velocity * math.sin(self.theta + self.slip_angle)
        v_dot = accel
        steer_angle_dot = steer_angle_vel
        theta_dot = self.angular_velocity

        # For ease of next two calculations
        rear_val = g * self.params.l_r - accel * self.params.h_cg
        front_val = g * self.params.l_f + accel * self.params.h_cg

        vel_ratio = 0
        first_term = 0
        if self.velocity == 0:
            vel_ratio = 0
            first_term = 0
        else:
            vel_ratio = self.angular_velocity / self.velocity
            first_term = self.params.friction_coeff / (self.velocity * (self.params.l_r + self.params.l_f))
        

        cs_f = self.params.cs_f
        cs_r = self.params.cs_r
        l_f = self.params.l_f
        l_r = self.params.l_r
        friction_coeff = self.params.friction_coeff
        mass = self.params.mass
        I_z = self.params.I_z
        wheelbase = self.params.wheelbase

        theta_double_dot = (friction_coeff * mass / (I_z * wheelbase)) * \
                (l_f * cs_f * self.steer_angle * (rear_val) + \
                self.slip_angle * (l_r * cs_r * (front_val) - l_f * cs_f * (rear_val)) -
                vel_ratio * (math.pow(l_f, 2) * cs_f * (rear_val) + math.pow(l_r, 2) * cs_r * (front_val)))

        slip_angle_dot = (first_term) * \
            (cs_f * self.steer_angle * (rear_val) -
             self.slip_angle * (cs_r * (front_val) + cs_f * (rear_val)) + \
             vel_ratio * (cs_r * l_r * (front_val) - cs_f * l_f * (rear_val))) - \
            self.angular_velocity

        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += theta_dot * dt
        self.velocity += v_dot * dt
        self.steer_angle += steer_angle_dot * dt
        self.angular_velocity += theta_double_dot * dt
        self.slip_angle += slip_angle_dot * dt
        self.st_dyn = True

    def update_state(self, action):
        """
        action is dict{"speed":float, "angle":float in radians}
        """
        #1: Convert action into accel, steer_vel
        accel = -1.0 * self.compute_accel(action.get("speed"))
        steer_vel = self.compute_steer_vel(action.get("angle"))

        #2:Update Car Pose
        self.update_pose(accel, steer_vel)
        self.velocity = min(max(self.velocity, -self.max_speed()), self.max_speed())
        self.steer_angle = min(max(self.steer_angle, -self.params.max_steering_angle), self.params.max_steering_angle)

        return self.convert_state_to_pose()

    def max_steering_vel(self):
        return self.params.max_steering_vel

    def max_speed(self):
        return self.params.max_speed
    
    def max_decel(self):
        return self.params.max_decel

    def max_accel(self):
        return self.params.max_accel

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
        self.sensor_info = {"angle_min":-135.0 * (math.pi/180.0), "angle_incr":.004363323}

        self.car_state = Car()
        self.curr_pose = None

    ###########GYM METHODS##################################################

    def data_to_xyz(self, data):
        """ Transform Lidar pointcloud into [x, y, z]
        """
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        return points

    def _get_obs(self):
        #Get Camera imgs
        imgs = self._get_imgs()

        #Get LiDAR reading & sort returned pointclouds according to their angle (from left)
        while True:
            lidarData = self.client.getLidarData()
            if len(lidarData.point_cloud) >= 3:
                break
        lidarData = self.data_to_xyz(lidarData)


        #Get steer data
        steer = {"angle": 0.0, "steering_angle_velocity": 0.0, "speed": 0.0}
        latest_dict = {'lidar': lidarData, 'steer': steer, 'img':imgs}
        self.add_to_history(steer)
        return latest_dict

    def reset(self, **kwargs):
        """
        Reset to initial position
        """
        self.client.reset()
        time.sleep(3) #SLEEP UNTIL CAR TOUCHES THE GROUND -> Or else shaking initial_pos
        self.curr_pose = self.client.simGetVehiclePose()
        self.car_state.init_state_from_pose(self.curr_pose)
        return self._get_obs()

    def get_reward(self):
        """
        TODO:Implement reward functionality
        """
        return 0
    

    def update_kinematics(self, action):
        curr_position = self.curr_pose.position #Vector 3r 
        curr_orientation = self.curr_pose.orientation #Quaternion

        dt = 0.01
        #1: Convert action: {speed, steering_angle} to accel, steer_angle_vel
        
        # (explain new_position calculation)
        new_postion = curr_position + airsim.Vector3r(0, -0.1, 0)

        # (explain new_orientation calculation)
        # new_orientation = airsim.utils.to_quaternion()
        curr_pitch, curr_roll, curr_yaw = airsim.utils.to_eularian_angles(curr_orientation)
        new_pitch, new_roll, new_yaw = curr_pitch, curr_roll, curr_yaw + 0.01
        new_orientation = airsim.utils.to_quaternion(new_pitch, new_roll, new_yaw)
        pose = airsim.Pose(new_postion, new_orientation)
        self.curr_pose = pose
        return pose

    def step(self, action):
        """
        Action should be a steer_dict = {"angle":float, "speed":float}
        """
        car_controls = airsim.CarControls()
        car_controls.throttle = action.get("speed")
        car_controls.steering = action.get("angle")

        self.client.setCarControls(car_controls)

        # negspeed = action["speed"] * -1
        # negangle = action["angle"] * -1
        # action["speed"] = negspeed
        # action["angle"] = negangle
        
        # # Update Kinematics using Bicycle Model
        # new_pose = self.car_state.update_state(action)


        # self.client.simSetVehiclePose(new_pose, True)
        
        time.sleep(0.001)

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
        collision_info = self.client.simGetCollisionInfo()
        z_norm = collision_info.normal.z_val
        y_pos = collision_info.position.y_val
        if(collision_info.has_collided and (z_norm != -1 or y_pos < -1)):
            return True
        return False

    ###########EXTRA METHODS##################################################

    def _get_imgs(self, labels=["front_center"]):
        label_to_func = lambda lbl: airsim.ImageRequest(lbl, airsim.ImageType.Scene, False, False)
        bytestr_to_np = lambda rep: np.fromstring(rep.image_data_uint8, dtype=np.uint8).reshape(rep.height, rep.width, -1)
        responses = self.client.simGetImages(list(map(label_to_func, labels)))
        images = list(map(bytestr_to_np, responses))
        return images
    
    def add_to_history(self, steer):
        if abs(steer["angle"]) > 0.05 and steer["angle"] < -0.05:
            for i in range(40):
                self.history.append(steer)

    def render_lidar2D(self, lidar):
        """ Visualize a lidarPointcloud
        # Expects lidar data in 2d array [x, y]
        # """
        lidar_frame = np.zeros((500, 500, 3))
        cx = 250
        cy = 250
        rangecheck = lambda x, y: abs(x) < 1000. and abs(y) < 1000.
        for i in range(lidar.shape[0]):
            x = lidar[i, 0]
            y = lidar[i, 1]
            if (rangecheck(x, y)):
                scaled_x = int(cx + x*50)
                scaled_y = int(cy - y*50)
                cv2.circle(lidar_frame, (scaled_x, scaled_y), 1, (255, 255, 255), -1)
        cv2.imshow("lidarframe", lidar_frame)
        cv2.waitKey(1)