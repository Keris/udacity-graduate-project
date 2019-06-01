"""Landing task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):
    """Simple task where the goal is to land from some target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([-cube_size / 2, -cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([cube_size / 2, cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.target_z = 0.0  # Make the agent hover at 10 units above ground
        self.count = 0

        self.reset()

    def reset(self):
        self.last_pose = None
        self.count = 0

        # Return initial condition
        return Pose(
                position=Point(0.0, 0.0, 10.0),  # drop off from a slight height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )


    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        # state = np.array([
        #         pose.position.x, pose.position.y, pose.position.z,
        #         pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_pose is None:
            dist_last_pose = np.array([0., 0., 0.])
        else:
            dist_last_pose = np.array([
                abs(pose.position.x - self.last_pose.position.x),
                abs(pose.position.y - self.last_pose.position.y),
                abs(pose.position.z - self.last_pose.position.z)
            ])
        dist_target_z = abs(pose.position.z - self.target_z)

        state = np.concatenate([
            np.array([pose.position.x, pose.position.y, pose.position.z]),
            dist_last_pose,
            np.array([dist_target_z])
        ])
        
        self.last_pose = pose

        # Compute reward / penalty and check if this episode is complete
        linear_acc = np.linalg.norm(np.array([
            linear_acceleration.x, linear_acceleration.y, linear_acceleration.z
            ])
        )

        done = False
        reward = (10.0 - dist_target_z) * 0.8
        if pose.position.z <= 0.5:
            reward += 5.0  # give a small bonus
            if linear_acc:
                reward -= 0.1 * linear_acc
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 5.0
            done = True
        
        self.count += 1

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
