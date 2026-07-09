from simulation.sim import run_simulation
from quadruped_pympc.config import cfg
from rclpy.node import Node
import rclpy
from threading import Thread
from geometry_msgs.msg import Twist
from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
import math
import argparse
from time import sleep

class Simulation(Node):
    def __init__(self, robot_name):
        super().__init__("simulation_node")        
        #if robot_name != "go1":
        #    raise ValueError("unsupported robot model")
        self.play = False

        self.cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self.update_vel, 10)
        self.pause_play_sub = self.create_subscription(Empty, "/pause_play", self.update_play, 10)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        
        self.msg = JointState()
        self.msg.name = [''] * 12

        cfg.robot_name = robot_name
        conf = cfg.get_config()
        self.env = QuadrupedEnv(
            robot=robot_name,
            scene=conf["simulation_params"]["scene"],
            sim_dt=conf["simulation_params"]["dt"],
            ref_base_lin_vel=np.asarray((0.0, 4.0)) * conf["hip_height"],
            ref_base_ang_vel=(-0.4, 0.4),
            ground_friction_coeff=(0.5, 1.0),
            base_vel_command_type="human",
            state_obs_names=tuple([])
        )
        self.thread = Thread(target=self.simulation_loop, kwargs={'qpympc_cfg': conf, 'render': False, 'env': self.env})
        print("ready")
        self.thread.start()

    def update_vel(self, msg: Twist):
        self.env._ref_base_lin_vel_H[0] = msg.linear.x
        self.env._ref_base_lin_vel_H[1] = msg.linear.y
        self.env._ref_base_ang_yaw_dot = msg.angular.z

    def update_play(self, _):
        if not self.play:
            sleep(1)
        self.play = not self.play

    def simulation_loop(self, qpympc_cfg=None, render=False, env=None):
        simulation = run_simulation(qpympc_cfg=qpympc_cfg, render=render, env=env)
        next(simulation)
        joints = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
                  'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
                  'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
       
        while True:
            if self.play:
                next(simulation)
                self.msg.position, self.msg.velocity = env.get_joint_attributes(joints)
                self.joint_state_pub.publish(self.msg)
            else:
                sleep(0.1)

parser = argparse.ArgumentParser()
parser.add_argument("robot_name")
robot_name = parser.parse_args().robot_name
rclpy.init()
node  = Simulation(robot_name)
rclpy.spin(node)
rclpy.shutdown()

