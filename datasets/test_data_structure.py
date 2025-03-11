import numpy as np
import os

def print_pkl_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(data["action"])
    except Exception as e:
        print(f"An error occurred: {e}")

def print_npz_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(data["action"][10])
    except Exception as e:
        print(f"An error occurred: {e}")

def check_seed_folders(parent_folder):
    missing_seeds = []
    for i in range(5000):
        folder_name = f"seed_{i}"
        folder_path = os.path.join(parent_folder, folder_name)
        if not os.path.isdir(folder_path):
            missing_seeds.append(i)
    return missing_seeds

def compare_actions(pkl_file_dir, npz_file_path, num_steps):
    try:
        for i in range(num_steps):
            pkl_file_path = os.path.join(pkl_file_dir, f"step_{i}.pkl")
            pkl_data = np.load(pkl_file_path, allow_pickle=True)
            npz_data = np.load(npz_file_path, allow_pickle=True)

            pkl_action = pkl_data["action"]
            npz_action = npz_data["action"][i]
            
            with open("pkl_actions.txt", "a") as pkl_file, open("npz_actions.txt", "a") as npz_file:
                pkl_file.write(f"Step {i}: {pkl_action}\n")
                npz_file.write(f"Step {i}: {npz_action}\n")
                
            if not np.array_equal(pkl_action, npz_action):
                print(f"Difference found at step {i}:")
                print(f"pkl action: {pkl_action}")
                print(f"npz action: {npz_action}")
    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    pkl_path = "/home/zhouzhiting/Data/panda_data/cano_policy_pd_2/seed_0/ep_0"
    # print_pkl_data(pkl_path)

    npz_path = "/home/zhouzhiting/Data/panda_data/cano_policy_pd_2/seed_0/ep_0/total_steps_new.npz"
    # print_npz_data(npz_path)

    compare_actions(pkl_path, npz_path, 100)

    # parent_folder = "/home/zhouzhiting/Data/panda_data/cano_policy_pd_2"

    # missing_seeds = check_seed_folders(parent_folder)
    # if missing_seeds:
    #     print(f"Missing seed folders: {missing_seeds}")
    # else:
    #     print("All seed folders are present.")
