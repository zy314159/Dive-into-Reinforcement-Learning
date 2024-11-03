import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        self.action_space=[0,1,2,3]

        self.action_meanings={0:'up',1:'down',2:'left',3:'right'}

        self.reward_map = np.array([[0,0,0,1.0],[0,None,0,-1.0],[0,0,0,0]])

        self.goal_state=(0,3)

        self.wall_state=(1,1)

        self.start_state=(2,0)
        
        self.agent_state=self.start_state
    
    @property   
    def height(self):
        return len(self.reward_map)
    
    @property
    def width(self):
        return len(self.reward_map[0])
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h,w)

    def next_state(self,state,action):
        action_move_map = [(-1,0),(1,0),(0,-1),(0,1)]
        move = action_move_map[action]
        next_state = (state[0]+move[0],state[1]+move[1])
        ny,nx = next_state

        if nx<0 or nx>=self.width or ny<0 or ny>=self.height:
            next_state = state
        
        elif next_state==self.wall_state:
            next_state = state
        return next_state
    
    def reward(self,state,action,next_state):
        return self.reward_map[next_state[0]][next_state[1]]

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)

if __name__ == '__main__':
    env=GridWorld()
    V={}
    for state in env.states():
        V[state] = np.random.randn()
    env.render_v(V)
