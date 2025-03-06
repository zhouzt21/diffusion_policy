import numpy as np
import torch
import os
import pickle
from time import time
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import colorsys
from tqdm import tqdm
import random
import time
import imageio  # import imageio.v2 as imageio
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
import sys
sys.path.append('/home/zhouzhiting/Projects')
from collections import defaultdict
import numpy as np
import threading

from diffusion_policy.utils.math_utils import wrap_to_pi, euler2quat, quat2euler, get_pose_from_rot_pos
import copy

# 
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

import cupy as cp

class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.files = []

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def feed(self, inputs):
        self.files.extend(inputs)

    def __next__(self):
        batch = []
        if len(self.files) < self.batch_size:
            raise StopIteration()

        for _ in range(self.batch_size):
            jpeg_filename = self.files.pop()
            f = open(jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        return batch

class DALIImagePipeline(Pipeline):
    def __init__(self, batch_size=1, num_threads=4, device_id=0):
        super().__init__(batch_size, num_threads, device_id, exec_pipelined=False, exec_async=False)
        self.eii = ExternalInputIterator(batch_size)
        
    def define_graph(self):
    
        jpegs = fn.external_source(source=self.eii, device="cpu")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        
        ratio = 0.95
        images = fn.random_resized_crop(
            images,
            size=(224, 224),
            random_area=(ratio*ratio, 1.0),
            random_aspect_ratio=(1.0, 1.0),
            device="gpu"
        )
        
        images = fn.cast(images, dtype=types.FLOAT)
        images = fn.normalize(images, device="gpu", mean=0.0, stddev=255.0)
        
        # 调整维度顺序 (HWC -> CHW)
        images = fn.transpose(images, perm=[2, 0, 1])
        
        return images


# 创建全局时间统计字典
timing_stats = defaultdict(list)
timing_lock = threading.Lock()  # 添加锁以确保线程安全
batch_counter = 0
print_interval = 10  # 每10个批次打印一次统计结果

def timing_decorator(func_name):
    """函数计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            with timing_lock:
                timing_stats[func_name].append(elapsed)
            return result
        return wrapper
    return decorator


class TimingContext:
    """上下文管理器用于代码块计时"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        with timing_lock:
            timing_stats[self.name].append(elapsed)


def print_timing_stats():
    """打印时间统计信息"""
    print("\n===== 数据加载时间统计 =====")
    for operation, times in sorted(timing_stats.items()):
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"{operation:<30} - 平均: {avg_time*1000:.2f}ms, 最大: {max_time*1000:.2f}ms, 最小: {min_time*1000:.2f}ms, 调用次数: {len(times)}")
    print("=========================\n")



class Sim2SimEpisodeDatasetEff(Dataset):
    def __init__(
            self,
            data_roots,
            num_seeds,
            chunk_size,
            split="train",
            norm_stats_path=None,
            augment_images=True,
            use_desired_action=True,
            **kwargs
    ):
        super().__init__()

        self.device_id = 0  
        self.image_pipeline = None

        with TimingContext("dataset_initialization"):  #time
            self.data_roots = data_roots
            self.camera_names = kwargs.get("camera_names", ["third", "wrist"])
            self.chunk_size = chunk_size
            self.augment_images = augment_images
            self.use_desired_action = use_desired_action
            self.transformations = None
            self.split = split
            
            # 加载预处理数据
            self.episode_data = {}
            episode_list = []
            num_steps_list = []

            with TimingContext("total_steps loading"): #time
                for data_idx, data_root in enumerate(self.data_roots):
                    for s in range(num_seeds):
                        seed_path = os.path.join(data_root, f"seed_{s}")
                        total_steps_path = os.path.join(seed_path, "ep_0", "total_steps.npz")
                        if not os.path.exists(total_steps_path):
                            continue
                            
                        data = np.load(total_steps_path, allow_pickle=True)
                        metadata = data['metadata'].item()
                        num_steps = metadata['num_steps']
                        
                        if num_steps > 1:
                            item_index = (data_idx, s, 0)  # ep_id 始终为0
                            episode_list.append(item_index)
                            num_steps_list.append(num_steps)
                       
                            self.episode_data[item_index] = {
                                'tcp_pose': data['tcp_pose'],
                                'gripper_width': data['gripper_width'],
                                'robot_joints': data['robot_joints'],
                                'privileged_obs': data['privileged_obs'],
                                'action': data['action'],
                                'desired_grasp_pose': data['desired_grasp_pose'],
                                'desired_gripper_width': data['desired_gripper_width']
                            }
                            
                            with TimingContext("camera loading"): #time
                                for cam in self.camera_names:
                                    self.episode_data[item_index][f'images_{cam}'] = data[f'images_{cam}']

            if split == "train":
                self.episode_list = episode_list[:int(0.99 * len(episode_list))]
                self.num_steps_list = num_steps_list[:int(0.99 * len(episode_list))]
            else:
                self.episode_list = episode_list[int(0.99 * len(episode_list)):]
                self.num_steps_list = num_steps_list[int(0.99 * len(episode_list)):]

            self.cum_steps_list = np.cumsum(self.num_steps_list)
            print(split, len(self.episode_list), self.cum_steps_list[-1])

            if norm_stats_path is None:
                stats = self.compute_normalize_stats()
            else:
                stats = pickle.load(open(norm_stats_path, "rb"))
            print(stats)
            self.update_obs_normalize_params(stats)

            self.__getitem__(0)  # initialize self.transformations

    def get_unnormalized_item(self, index):
        with TimingContext("get_unnormalized_item_total"):
            result_dict = {}
            result_dict["lang"] = " "
            
            with TimingContext("locate_trajectory"):
                traj_idx, start_ts = self._locate(index)
                end_ts = min(self.num_steps_list[traj_idx], start_ts + self.chunk_size + 1)
                is_pad = np.zeros((self.chunk_size,), dtype=bool)
                if end_ts < start_ts + self.chunk_size + 1:
                    is_pad[-(start_ts + self.chunk_size + 1 - end_ts):] = True
                result_dict["is_pad"] = is_pad
    
                data_idx, s, ep_id = self.episode_list[traj_idx]
                data_root = self.data_roots[data_idx]
    

            # 修改图像加载部分，使用DALI
            with TimingContext("image_loading"):
                image_paths = []
                for cam in self.camera_names:
                    image_path = os.path.join(data_root, 
                                            f"seed_{s}", 
                                            f"ep_{ep_id}", 
                                            f"step_{start_ts}_cam_{cam}.jpg")

                    # image_path = os.path.join(data_root, 
                    #                         f"seed_0", 
                    #                         f"ep_0", 
                    #                         f"step_0_cam_third.jpeg")
                    image_paths.append(image_path)
                
                # 创建或更新DALI pipeline
                if self.image_pipeline is None:
                    self.image_pipeline = DALIImagePipeline(
                        batch_size=1,
                        num_threads=4,
                        device_id=self.device_id
                    )
                    self.image_pipeline.build()
                
                self.image_pipeline.eii.feed(image_paths)
                # 执行pipeline获取处理后的图像
                outputs = self.image_pipeline.run()
                images = outputs[0].as_tensor()  # 直接获取GPU上的张量 (shape: [batch_size, 3, 224, 224])

                # 转换为PyTorch Tensor（保持在GPU）
                # images_tensor = torch.from_dlpack(images.to_dlpack())  # 或使用.as_cpu()转CPU
                images_tensor = torch.as_tensor(cp.array(images), device='cuda')
                result_dict["images"] = images_tensor

            # 从缓存数据中获取相应片段
            with TimingContext("data_fetching"): 
                episode_data = self.episode_data[(data_idx, s, ep_id)]
                tcp_poses = episode_data['tcp_pose'][start_ts:end_ts]
                gripper_widths = episode_data['gripper_width'][start_ts:end_ts]
                actions = episode_data['action'][start_ts:end_ts]
    
            # 处理位姿数据
            with TimingContext("pose_processing"):
                action_chunk = np.zeros((self.chunk_size, 10), dtype=np.float32)
                pose_at_obs = None
                pose_chunk = []
                gripper_width_chunk = []
                proprio_state = np.zeros((10,), dtype=np.float32)
                robot_state = np.zeros((10,), dtype=np.float32)
    
                # 处理初始位姿
                with TimingContext("initial_pose_processing"):
                    tcp_pose = tcp_poses[0]
                    pose_p, pose_q = tcp_pose[:3], tcp_pose[3:]
                    pose_mat = quat2mat(pose_q)
                    pose = get_pose_from_rot_pos(pose_mat, pose_p)
                    pose_at_obs = pose
                    pose_mat_6 = pose_mat[:, :2].reshape(-1)
                    proprio_state[:] = np.concatenate([
                        pose_p,
                        pose_mat_6,
                        np.array([gripper_widths[0]]),
                    ])
                    robot_state[-1] = gripper_widths[0]
    
                # 处理后续位姿
                with TimingContext("subsequent_poses_processing"):
                    for i in range(1, len(tcp_poses)):
                        if self.use_desired_action:
                            action = actions[i]
                            pose_chunk.append(get_pose_from_rot_pos(
                                quat2mat(euler2quat(action[3:6])),
                                action[:3]
                            ))
                            gripper_width_chunk.append(np.array([action[-1]]))
    
                # 计算相对位姿
                with TimingContext("relative_pose_calculation"):
                    _pose_relative = np.eye(4)
                    robot_state[:9] = np.concatenate(
                        [_pose_relative[:3, 3], _pose_relative[:3, :2].reshape(-1)]
                    )
                    for i in range(len(pose_chunk)):
                        _pose_relative = np.linalg.inv(pose_at_obs) @ pose_chunk[i]
                        action_chunk[i] = np.concatenate([
                            _pose_relative[:3, 3],
                            _pose_relative[:3, :2].reshape(-1),
                            gripper_width_chunk[i],
                        ])
    
            # 设置返回结果
            with TimingContext("result_preparation"):
                result_dict["robot_state"] = robot_state
                result_dict["proprio_state"] = proprio_state
                result_dict["action"] = action_chunk
    
            return result_dict

    def update_obs_normalize_params(self, obs_normalize_params):
        self.OBS_NORMALIZE_PARAMS = copy.deepcopy(obs_normalize_params)

        self.pose_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["pose", "gripper_width"]
            ]
        )
        self.pose_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["pose", "gripper_width"]
            ]
        )

        self.proprio_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )
        self.proprio_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )

    def compute_normalize_stats(self, scale_eps=0.03):
        print("compute normalize stats...")
        # min and max scale
        joint_min, joint_max = None, None
        gripper_width_min, gripper_width_max = None, None
        pose_min, pose_max = None, None
        proprio_min, proprio_max = None, None

        def safe_minimum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.minimum(a, b)

        def safe_maximum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.maximum(a, b)

        def safe_min(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.min(a, axis=axis)

        def safe_max(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.max(a, axis=axis)

        for i in tqdm(range(len(self))):
            item_dict = self.get_unnormalized_item(i)
            pose = item_dict["robot_state"][:9]
            action_pose = item_dict["action"][~item_dict["is_pad"]][:, :9]
            gripper_width = item_dict["robot_state"][9:10]
            action_gripper_width = item_dict["action"][~item_dict["is_pad"]][:, 9:10]
            proprio_pose = item_dict["proprio_state"][:9]

            pose_min = safe_minimum(
                safe_minimum(pose_min, pose), safe_min(action_pose, axis=0)
            )
            pose_max = safe_maximum(
                safe_maximum(pose_max, pose), safe_max(action_pose, axis=0)
            )
            gripper_width_min = safe_minimum(
                safe_minimum(gripper_width_min, gripper_width),
                safe_min(action_gripper_width, axis=0),
            )
            gripper_width_max = safe_maximum(
                safe_maximum(gripper_width_max, gripper_width),
                safe_max(action_gripper_width, axis=0),
            )
            proprio_min = safe_minimum(
                proprio_min, proprio_pose
            )
            proprio_max = safe_maximum(
                proprio_max, proprio_pose
            )

        params = {}
        params["pose"] = {
            "mean": (pose_min + pose_max) / 2,
            "scale": np.maximum((pose_max - pose_min) / 2, scale_eps),
        }
        params["gripper_width"] = {
            "mean": (gripper_width_min + gripper_width_max) / 2,
            "scale": np.maximum((gripper_width_max - gripper_width_min) / 2, scale_eps),
        }
        params["proprio_state"] = {
            "mean": (proprio_min + proprio_max) / 2,
            "scale": np.maximum((proprio_max - proprio_min) / 2, scale_eps),
        }
        return params

    # 其他方法与原始Dataset相同
    def __len__(self):
        return self.cum_steps_list[-1]

    def _locate(self, index):
        assert index < len(self)
        traj_idx = np.where(self.cum_steps_list > index)[0][0]
        steps_before = self.cum_steps_list[traj_idx - 1] if traj_idx > 0 else 0
        start_ts = index - steps_before
        return traj_idx, start_ts

    def __getitem__(self, index: int):
        with TimingContext("getitem_total"):  #time
            with TimingContext("get_unnormalized_item"):    #time
                result = self.get_unnormalized_item(index)

            robot_state = result["robot_state"]
            proprio_state = result["proprio_state"]
            is_pad = result["is_pad"]
            action_chunk = result["action"]
            robot_state = (
                                robot_state - self.pose_gripper_mean
                        ) / self.pose_gripper_scale
            action_chunk[~is_pad] = (
                                            action_chunk[~is_pad] - np.expand_dims(self.pose_gripper_mean, axis=0)
                                    ) / np.expand_dims(self.pose_gripper_scale, axis=0)
            proprio_state = (
                                    proprio_state - self.proprio_gripper_mean
                            ) / self.proprio_gripper_scale

            result["robot_state"] = torch.from_numpy(robot_state)
            result["proprio_state"] = torch.from_numpy(proprio_state)
            result["action"] = torch.from_numpy(action_chunk)
            result["is_pad"] = torch.from_numpy(is_pad)

            # 图像已经在DALI中处理，这里可以简化
            # result["images"] = torch.from_numpy(result["images"])

            return result


def step_collate_fn(samples):
    batch = {}
    for key in samples[0].keys():
        if key != "lang":
            # print(key, samples[0][key].shape)
            batched_array = torch.stack([sample[key] for sample in samples], dim=0)
            batch[key] = batched_array
    batch["lang"] = [sample["lang"] for sample in samples]
    return batch


def load_sim2sim_data(data_roots, num_seeds, train_batch_size, val_batch_size, chunk_size, **kwargs):
    # construct dataset and dataloader
    train_dataset = Sim2SimEpisodeDatasetEff(
        data_roots,
        num_seeds,
        split="train",
        chunk_size=chunk_size,
        norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"),
        **kwargs
    )
    val_dataset = Sim2SimEpisodeDatasetEff(
        data_roots,
        num_seeds,
        split="val",
        chunk_size=chunk_size,
        norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"),
        **kwargs
    )
    train_num_workers = 32 #8
    val_num_workers = 32 #8
    print(
        f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        collate_fn=step_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
        collate_fn=step_collate_fn,
    )
    return train_dataloader, val_dataloader, train_dataset.OBS_NORMALIZE_PARAMS, val_dataset.OBS_NORMALIZE_PARAMS


if __name__ == "__main__":
    # 添加简单的测试代码来验证时间统计功能
    print("Testing timing functionality...")
    data_roots =["/home/zhouzhiting/Data/panda_data/cano_policy_efficient"]
    
    try:
        train_loader, _, _, _ = load_sim2sim_data(
            data_roots=data_roots,
            num_seeds= 10,
            train_batch_size=128,
            val_batch_size=128,
            chunk_size=20,
            usages = ["obs"],
            camera_names=["third"]  #, "wrist"
        )

        # 处理3个批次后停止
        for i, batch in enumerate(train_loader):
            print(f"Processed batch {i}")
            if i >= 2:  
                break
                
        # 打印最终统计结果
        print_timing_stats()
        
    except Exception as e:
        print(f"Error testing timing: {e}")
        import traceback
        traceback.print_exc()

