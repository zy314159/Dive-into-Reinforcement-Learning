def value_iter_one_step(V,env,gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state,action)
            r = env.reward(state,action,next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V

def value_iter(env,gamma,threshold=0.001,is_render=True):
    from collections import defaultdict
    V = defaultdict(lambda :0)

    while True:
        old_V = V.copy()
        V = value_iter_one_step(V,env,gamma)

        delta = 0
        for state in V.keys():
            delta = max(delta,abs(V[state] - old_V[state]))
        
        if is_render:
            env.render_v(V)

        if delta < threshold:
            break
    return V

import sys
import os
sys.path.append(os.getcwd())
from common.gridworld import GridWorld
from ch4.policy_iter import greedy_policy
from collections import defaultdict

V = defaultdict(lambda :0)
env = GridWorld()
gamma = 0.9

V=value_iter(env,gamma,threshold=0.001,is_render=True)
pi=greedy_policy(V,env,gamma)
env.render_v(V,pi)