import numpy as np

'''
Problem: When done==True because the agent has fallen over, the environment automatically resets before we have a chance
to grab the final state (not the final observation, which is available in the info dictionary returned by step()). To get around this,
we remove the terminal transition and treat the penultimate transition as terminal instead. Yes, this is not a perfect solution
but as far as I can tell in a penultimate transition the agent is never able to actually recover from falling. 
'''

expert_levels = [ "a2c_","ppo_","sac_"]
env_name = "Walker2d-v3"

for expert_level in expert_levels:
    timestep_obs_updated = []
    timestep_states_updated = []
    timstep_rewards_updated = []
    timestep_actions_updated = []
    timestep_next_obs_updated = []
    timestep_dones_updated = []

    timestep_obs = np.load("collect_dataset/" + env_name + "/" + expert_level + "obs.npy",allow_pickle=True)
    timestep_dones = np.load("collect_dataset/" + env_name + "/" + expert_level + "dones.npy",allow_pickle=True)
    timestep_states = np.load("collect_dataset/" + env_name + "/" + expert_level + "states.npy",allow_pickle=True)
    timstep_rewards = np.load("collect_dataset/" + env_name + "/" + expert_level + "rewards.npy",allow_pickle=True)
    timestep_actions = np.load("collect_dataset/" + env_name + "/" + expert_level + "acts.npy",allow_pickle=True)
    timestep_next_obs = np.load("collect_dataset/" + env_name + "/" + expert_level + "next_obs.npy",allow_pickle=True)

    for i in range(len(timestep_obs)):
        if timestep_dones[i]:
            continue
        if i < len(timestep_obs) -1 and timestep_dones[i+1]:
            timestep_dones_updated.append(True)
        else:
            assert timestep_dones[i] == False
            timestep_dones_updated.append(timestep_dones[i])

        timestep_obs_updated.append(timestep_obs[i])
        timestep_states_updated.append(timestep_states[i])
        timstep_rewards_updated.append(timstep_rewards[i])
        timestep_actions_updated.append(timestep_actions[i])
        timestep_next_obs_updated.append(timestep_next_obs[i])

    np.save("collect_dataset/" + env_name + "/" + expert_level + "obs.npy",timestep_obs_updated)
    np.save("collect_dataset/" + env_name + "/" + expert_level + "dones.npy",timestep_dones_updated)
    np.save("collect_dataset/" + env_name + "/" + expert_level + "states.npy",timestep_states_updated)
    np.save("collect_dataset/" + env_name + "/" + expert_level + "rewards.npy",timstep_rewards_updated)
    np.save("collect_dataset/" + env_name + "/" + expert_level + "acts.npy",timestep_actions_updated)
    np.save("collect_dataset/" + env_name + "/" + expert_level + "next_obs.npy",timestep_next_obs_updated)
