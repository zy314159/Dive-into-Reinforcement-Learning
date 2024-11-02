from policy_eval import policy_eval

def argmax(d):
    return max(d, key=d.get)

def greedy_policy(V,env,gamma=0.9):
    pi = {}
    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state,action)
            r = env.reward(state,action,next_state)
            action_values[action] = r + gamma * V[next_state]
        max_action = argmax(action_values)
        action_probs = {a:0 for a in env.actions()}
        action_probs[max_action] = 1.0 #greedy policy
        pi[state] = action_probs
    return pi

def policy_iter(env,gamma,threshold=0.001,is_render=False):
    from collections import defaultdict
    pi = defaultdict(lambda :{0:0.25,1:0.25,2:0.25,3:0.25})
    V = defaultdict(lambda :0)

    while True:
        V = policy_eval(pi,V,env,gamma,threshold)
        new_pi = greedy_policy(V,env,gamma)

        if is_render:
            env.render_v(V,pi)

        if new_pi == pi:
            break
        pi = new_pi

    return pi

if __name__ == '__main__':
    from common.gridworld import GridWorld
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env,gamma,threshold=0.001,is_render=True) 