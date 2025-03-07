import os
from tqdm import tqdm

def check_total_steps(root_path):
    """检查指定路径下 total_steps.npz 文件的存在情况"""
    
    # 获取所有数据文件夹
    all_data_roots = [
        os.path.join(root_path, d) 
        for d in os.listdir(root_path) 
        if os.path.isdir(os.path.join(root_path, d))
    ]
    
    total_dirs = 0
    processed_dirs = 0
    unprocessed_dirs = [] 

    print(f"\n检查路径: {root_path}")
    print("=" * 50)
    
    for data_root in tqdm(all_data_roots, desc="检查进度"):
        episode_dir = os.path.join(data_root, "ep_0")
        if not os.path.exists(episode_dir):
            continue
            
        total_dirs += 1
        npz_path = os.path.join(episode_dir, "total_steps.npz")
        
        if os.path.exists(npz_path):
            processed_dirs += 1
        else:
            unprocessed_dirs.append(data_root)
    
    print("\n统计结果:")
    print(f"总文件夹数: {total_dirs}")
    print(f"已处理文件夹数: {processed_dirs}")
    print(f"待处理文件夹数: {total_dirs - processed_dirs}")
    print(f"处理进度: {(processed_dirs/total_dirs*100):.2f}%")

    return unprocessed_dirs

if __name__ == "__main__":
    root_path = "/home/zhouzhiting/Data/panda_data/cano_policy_pd_2"
    unprocessed_dirs = check_total_steps(root_path)