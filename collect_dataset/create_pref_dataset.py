import numpy as np

env_name = "Walker2d-v3"
expert_levels = ["sac_","a2c_","ppo_"]
GAMMA = 0.999
eps = 1
segment_length = 100
n_segments_per_expert = 500
obs = []
acts = []
regrets = []

for expert_level in expert_levels:
    timestep_states = np.load("collect_dataset/" + env_name + "/" + expert_level + "states.npy",allow_pickle=True)
    timstep_rewards = np.load("collect_dataset/" + env_name + "/" + expert_level + "rewards.npy",allow_pickle=True)
    timestep_actions = np.load("collect_dataset/" + env_name + "/" + expert_level + "acts.npy",allow_pickle=True)
    timestep_dones = np.load("collect_dataset/" + env_name + "/" + expert_level + "dones.npy",allow_pickle=True)
    timestep_obs = np.load("collect_dataset/" + env_name + "/" + expert_level + "obs.npy",allow_pickle=True)

    timestep_Vs = np.load("collect_dataset/" + env_name + "/" + expert_level + "Vs.npy",allow_pickle=True)

    done_indices = np.where(timestep_dones == True)[0]

    truncated_indices = [done_indices[i] for i in range(1,len(done_indices)) if done_indices[i] - done_indices[i-1] == 9999 ]
    if done_indices[0] == 9999:
        truncated_indices.append(done_indices[0])

    selected_segment_is = []

    while len(selected_segment_is) < n_segments_per_expert:
        #Ensures that we only pick a start state that is at least segment_length away from the episodes last state IF that last state is not a terminal state
        redo=True
        while redo:
            segment_i = np.random.choice(range(len(timestep_Vs)))
            redo=False
            if segment_i > len(timestep_Vs) - 100:
                redo=True
            for truncated_i in truncated_indices:
                if truncated_i - segment_i < segment_length and truncated_i - segment_i >= 0:
                    redo=True


        segment_obs = []
        segment_acts = []
        segment_regret = 0

        for timestep in range(segment_i, segment_i+segment_length):
            if timestep_dones[timestep]:
                if timestep in truncated_indices:
                    #this means that done=True because the episode was truncated, not becauae the terminal state was reached
                    assert False

                segment_obs.append(timestep_obs[timestep])
                segment_acts.append(timestep_actions[timestep])
                segment_regret += (timestep_Vs[timestep] - (timstep_rewards[timestep] + GAMMA*0))

                #adds transitions from absorbing state

                for _ in range(timestep+1, segment_i+segment_length):
                    segment_obs.append(np.zeros(timestep_obs[timestep].shape))
                    segment_acts.append(np.zeros(timestep_actions[timestep].shape))
                break

            else:
                segment_obs.append(timestep_obs[timestep])
                segment_acts.append(timestep_actions[timestep])
                segment_regret += (timestep_Vs[timestep] - (timstep_rewards[timestep] + GAMMA*timestep_Vs[timestep+1]))


        # if len(segment_obs) != 100:
        #     print (segment_i, segment_i+segment_length)
        #     print (len(segment_obs))
        assert len(segment_obs) == len(segment_acts) == segment_length

        obs.append(segment_obs)
        acts.append(segment_acts)
        regrets.append(segment_regret)
        selected_segment_is.append(segment_i)


obs = np.array(obs)
acts = np.array(acts)
regrets = np.array(regrets)


indices = np.arange(obs.shape[0])
np.random.shuffle(indices)
assert (len(indices)/2).is_integer()
indices_1 = indices[:int(len(indices)/2)]
indices_2 = indices[int(len(indices)/2):]

obs_1 = obs[indices_1]
obs_2 = obs[indices_2]

acts_1 = acts[indices_1]
acts_2 = acts[indices_2]

regrets_1 = regrets[indices_1]
regrets_2 = regrets[indices_2]
labels = []

for r1,r2 in zip(regrets_1, regrets_2):
    if abs(r1-r2) < eps:
        labels.append(0.5)
    elif r1 < r2:
        labels.append(0)
    elif r2 < r1:
        labels.append(1)

np.savez("datasets/preference_transformer/walker2d-medium-replay-v2/num750_regret_train.npz", obs_1=obs_1, obs_2=obs_2, action_1=acts_1, action_2=acts_2, label=labels)
