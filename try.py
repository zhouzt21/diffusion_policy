import numpy as np
import pickle
import os
from tqdm import tqdm
from utils.math_utils import wrap_to_pi, euler2quat, quat2euler, get_pose_from_rot_pos
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat

# from diffusion_policy.datasets.dataset import Sim2SimEpisodeDataset
#
# dataset = Sim2SimEpisodeDataset(
#     data_root="/root/data/cano_policy_1",
#     chunk_size=20,
#     norm_stats_path="/root/data/cano_policy_1/norm_stats.pkl"
# )
#
# random_idx = np.random.randint(len(dataset), size=(1,))
#
# for idx in random_idx:
#     data = dataset.get_unnormalized_item(idx)

data_root = "/root/data/cano_policy_pd_rl_1"
num_seeds = 20
episode_list = []
num_steps_list = []
for s in range(num_seeds):
    seed_path = os.path.join(data_root, f"seed_{s}")
    ep_info = pickle.load(open(os.path.join(seed_path, "info.pkl"), "rb"))
    for ep_id, suc, num_steps in ep_info:
        if suc == "s" and num_steps > 1:
            item_index = (s, ep_id)
            episode_list.append(item_index)
            num_steps_list.append(num_steps)

cum_steps_list = np.cumsum(num_steps_list)


actions = []
obs = []

for idx, num_steps in enumerate(tqdm(num_steps_list)):
    s, ep_id = episode_list[idx]
    for i_step in range(num_steps):
        data_path = os.path.join(data_root, f"seed_{s}", f"ep_{ep_id}", f"step_{i_step}.pkl")
        data = pickle.load(open(data_path, "rb"))

        desired_action = data["action"]
        desired_dp = desired_action[:3]

        desired_dq = euler2quat(desired_action[3:6])
        desired_mat = quat2mat(desired_dq)
        desired_mat_6 = desired_mat[:, :2].reshape(-1)

        desired_gripper_width = desired_action[-1]

        desired_action = np.concatenate([
            desired_dp,
            desired_mat_6,
            [desired_gripper_width]
        ])
        actions.append(desired_action)

        obs.append(data["wrapper_obs"])

actions = np.array(actions)
obs = np.array(obs)

action_max = np.max(actions, axis=0)
action_min = np.min(actions, axis=0)
print(action_max, action_min)

obs_max = np.max(obs, axis=0)
obs_min = np.min(obs, axis=0)
print(obs_max, obs_min)





