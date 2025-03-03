## Sim2SimEpisodeDataset
-  result_dict["action"] 
    - 这里是已经做了chunk处理的action_chunk，和原来数据集中的action不一样（原有的action是用来step的，而且是可选择用不用的）
    - ndarray, (chunk_size, 10d)
    - use_desired_action: 则是用action相对上一目标pose计算出的作为target pose; 如果不用，则是用tcp_pose作为target pose
    - 最后都是相对tcp_pose_at_obs来计算的delta pose
- result_dict["proprio_state"]  
    - proprio_state, 一直都是观察帧的tcp pose，重复chunk个用来对齐
    - ndarray, (chunk_size, 10d)
-  result_dict["robot_state"] 
    - _pose_relative = np.eye(4)
    - robot_state[:9] = np.concatenate(
            [_pose_relative[:3, 3], _pose_relative[:3, :2].reshape(-1)]
        )
- 训的都是10d的（还原之后都是tcp pose）



## norm
- 注意是做完了chunk处理之后的norm操作

- [proprio_state]
    - data["tcp_pose"] + gripper_width,  6+1d --> 10d 
    - 9d, 是绝对pose概念
- [pose]
    - action 的最小值 和 robot_state的最小值 中的最小值 
    - 9d, 都是delta概念

-   params["pose"] = {
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

- 先对读入的数据进行norm操作之后进行训练；所以在验证的时候也需要unnorm
    - unnorm: 
        - 仿真得到的pose_at_obs是7d，转10d输入，没有归一化
        - 得到的action 10d -> 4*4d，然后和输入时的pose_at_obs转的4 *4d计算后，得到desired delta pose(7d+2d空)作为step需要的，没有归一化 