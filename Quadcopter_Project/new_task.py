import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900  #was 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        
        # first provide reward for closer distance and good drone "posture"
        #distance = np.sqrt((self.sim.pose[0] - self.target_pos[0])**2 + (self.sim.pose[1] - self.target_pos[1])**2 
        #                   +(self.sim.pose[2] - self.target_pos[2])**2)
        
        reward = 0
        penalty = 0
        
        x_distance = abs(self.sim.pose[0] - self.target_pos[0])
        y_distance = abs(self.sim.pose[1] - self.target_pos[1])
        z_distance = abs(self.sim.pose[2] - self.target_pos[2])
        
        # provide incentive to be close to target; the shortest the distance, the highest the reward.
        #print(distance)
        reward += 10 / (x_distance + 0.0001)
        reward += 10 / (y_distance + 0.0001)
        reward += 1000 / (z_distance + 0.0001) #add small value to avoid division by zero in case we're on target
        
        # provide small reward for simply flying (not crashing)
        #reward += 1
        
        # provide penalty when euler angles are not zeros; this aims to make the drone more stable
        penalty = abs(self.sim.pose[3:6]).sum()
        #print("penalty:",penalty)
        
        # could add a penalty to control drone speed/velocity so we manage drone battery.
        
        # Uses current pose of sim to return reward.
        # pass result through tanh to keep it within [-1,1] range and avoid gradient problems
        return np.tanh(reward - penalty)
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state