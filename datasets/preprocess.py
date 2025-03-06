# 预处理脚本：将每个episode的.pkl合并为单个文件
import pickle
import numpy as np
from tqdm import tqdm
import os
import imageio.v2 as imageio

def check_pkl_format(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Data is not a dictionary")

        for key in data.keys():
            print("key: ", key)
            print("value: ", type(data[key]))


def validate_dali_format(npz_path):
    """验证DALI格式数据"""
    data = np.load(npz_path)
    metadata = data['metadata'].item()
    print(f"数据集信息:")
    print(f"总步数: {metadata['num_steps']}")
    print(f"数据形状:")
    for k, shape in metadata['data_shapes'].items():
        actual_shape = data[k].shape
        assert actual_shape == shape, f"{k}形状不匹配: 期望{shape}, 实际{actual_shape}"
        print(f"  {k}: {shape}")


def convert_pose_to_array(pose):
    """将 sapien Pose 转换为 numpy 数组"""
    return np.concatenate([pose.p, pose.q])  # 位置(3) + 四元数(4)

def convert_array_to_pose(arr):
    """将 numpy 数组转换回 sapien Pose"""
    from sapien.core import Pose
    return Pose(p=arr[:3], q=arr[3:])

def convert_steps_with_image(root_path):
    camera_names = ["third"]  # 相机名称列表, "wrist"
    
    all_data_roots = [os.path.join(root_path, d) 
                    for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    for data_root in all_data_roots:
        episode_dir = os.path.join(data_root, "ep_0")
        if not os.path.exists(episode_dir):
            continue

        if os.path.exists(os.path.join(episode_dir, "total_steps.npz")):
            print(f"skip: {episode_dir} , total_steps.npz already exists")
            continue
        
        step_files = sorted([f for f in os.listdir(episode_dir) if f.startswith("step_") and f.endswith(".pkl")])
        
        # 创建空的数据字典来存储所有步骤的数据
        episode_data = {
            'tcp_pose': [],
            'gripper_width': [],
            'robot_joints': [],
            'privileged_obs': [],
            'action': [],
            'desired_grasp_pose': [],
            'desired_gripper_width': [],
        }
        
        # 添加图像数据存储
        for cam in camera_names:
            episode_data[f'images_{cam}'] = []
        
        # 收集所有步骤的数据
        for step_file in tqdm(step_files):
            step_idx = int(step_file.split('_')[1].split('.')[0])  # 获取步骤索引
            
            # 加载pkl数据
            with open(os.path.join(episode_dir, step_file), "rb") as f:
                data = pickle.load(f)
                pose_array = convert_pose_to_array(data['desired_grasp_pose'])
                
                # 将每个数据添加到对应的列表中
                episode_data['tcp_pose'].append(data['tcp_pose'])
                episode_data['gripper_width'].append(data['gripper_width'])
                episode_data['robot_joints'].append(data['robot_joints'])
                episode_data['privileged_obs'].append(data['privileged_obs'])
                episode_data['action'].append(data['action'])
                episode_data['desired_grasp_pose'].append(pose_array)
                episode_data['desired_gripper_width'].append(data['desired_gripper_width'])
                
                # 加载并存储图像数据
                for cam in camera_names:
                    image_path = os.path.join(episode_dir, f"step_{step_idx}_cam_{cam}.jpg")
                    try:
                        image = imageio.imread(image_path)
                        episode_data[f'images_{cam}'].append(image)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        raise
        
        # 将列表转换为numpy数组
        dali_format_data = {
            'tcp_pose': np.stack(episode_data['tcp_pose']),              # (N, 7)
            'gripper_width': np.array(episode_data['gripper_width']),    # (N,)
            'robot_joints': np.stack(episode_data['robot_joints']),      # (N, num_joints)
            'privileged_obs': np.stack(episode_data['privileged_obs']),  # (N, obs_dim)
            'action': np.stack(episode_data['action']),                  # (N, action_dim)
            'desired_grasp_pose': np.stack(episode_data['desired_grasp_pose']),  # (N, 7)
            'desired_gripper_width': np.array(episode_data['desired_gripper_width'])  # (N,)
        }
        
        # 添加图像数组
        for cam in camera_names:
            dali_format_data[f'images_{cam}'] = np.stack(episode_data[f'images_{cam}'])  # (N, H, W, C)
        
        # 保存为.npz格式
        save_path = os.path.join(episode_dir, "total_steps.npz")
        np.savez(
            save_path,
            **dali_format_data,
            metadata={
                'num_steps': len(step_files),
                'data_shapes': {k: v.shape for k, v in dali_format_data.items()},
                'camera_names': camera_names
            }
        )
        
        print(f"已保存到: {save_path}")
        print(f"数据形状:")
        for k, v in dali_format_data.items():
            print(f"  {k}: {v.shape}")



def convert_steps(root_path):

    all_data_roots = [os.path.join(root_path, d) 
                    for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    for data_root in all_data_roots:
        episode_dir = os.path.join(data_root, "ep_0")
        if not os.path.exists(episode_dir):
            continue
        
        if os.path.exists(os.path.join(episode_dir, "total_steps.npz")):
            print(f"skip: {episode_dir} , total_steps.npz already exists")
            continue

        step_files = sorted([f for f in os.listdir(episode_dir) if f.startswith("step_") and f.endswith(".pkl")])
        
        episode_data = {
            'tcp_pose': [],
            'gripper_width': [],
            'robot_joints': [],
            'privileged_obs': [],
            'action': [],
            'desired_grasp_pose': [],
            'desired_gripper_width': [],
        }

        # 收集所有步骤的数据
        for step_file in tqdm(step_files):
            step_idx = int(step_file.split('_')[1].split('.')[0])  # 获取步骤索引
            
            # 加载pkl数据
            with open(os.path.join(episode_dir, step_file), "rb") as f:
                data = pickle.load(f)
                pose_array = convert_pose_to_array(data['desired_grasp_pose'])
                
                # 将每个数据添加到对应的列表中
                episode_data['tcp_pose'].append(data['tcp_pose'])
                episode_data['gripper_width'].append(data['gripper_width'])
                episode_data['robot_joints'].append(data['robot_joints'])
                episode_data['privileged_obs'].append(data['privileged_obs'])
                episode_data['action'].append(data['action'])
                episode_data['desired_grasp_pose'].append(pose_array)
                episode_data['desired_gripper_width'].append(data['desired_gripper_width'])
                
        # 将列表转换为numpy数组
        dali_format_data = {
            'tcp_pose': np.stack(episode_data['tcp_pose']),              # (N, 7)
            'gripper_width': np.array(episode_data['gripper_width']),    # (N,)
            'robot_joints': np.stack(episode_data['robot_joints']),      # (N, num_joints)
            'privileged_obs': np.stack(episode_data['privileged_obs']),  # (N, obs_dim)
            'action': np.stack(episode_data['action']),                  # (N, action_dim)
            'desired_grasp_pose': np.stack(episode_data['desired_grasp_pose']),  # (N, 7)
            'desired_gripper_width': np.array(episode_data['desired_gripper_width'])  # (N,)
        }

        # 保存为.npz格式
        save_path = os.path.join(episode_dir, "total_steps.npz")
        np.savez(
            save_path,
            **dali_format_data,
            metadata={
                'num_steps': len(step_files),
                'data_shapes': {k: v.shape for k, v in dali_format_data.items()},
            }
        )
        
        print(f"已保存到: {save_path}")
        print(f"数据形状:")
        for k, v in dali_format_data.items():
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    
    root_path = "/home/zhouzhiting/Data/panda_data/cano_policy_pd_2"
    image_path = "/home/zhouzhiting/Data/panda_data/cano_policy_efficient/seed_0/ep_0/step_0_cam_third.jpg"
    # file_path = "/home/zhouzhiting/Data/panda_data/cano_policy_efficient/seed_0/ep_0/step_0.pkl"

    # check_pkl_format(file_path)

    convert_steps(root_path)

    # import imageio
    # image = imageio.imread(image_path)
    # print("image dtype: ", image.dtype)

