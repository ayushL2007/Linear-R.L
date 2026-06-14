def reward_fn(state :np.ndarray):
    reward = 0.9*state[:,1]**2-0.5*state[:,3]**2
    return float(-reward)