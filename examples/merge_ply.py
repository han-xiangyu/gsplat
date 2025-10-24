import numpy as np
from plyfile import PlyData, PlyElement
import sys
import os

# --- 你需要修改的部分 ---
# 第一个PLY文件的路径 (例如 0-3000 帧的结果)
file1_path = "/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/tra2_3000keyframes_fps_3cam/ply/point_cloud_99999.ply"
file2_path = "/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/tra2_3000to6000keyframes_fps_3cam/ply/point_cloud_99999.ply" 
# 第二个PLY文件的路径 (例如 3000-6000 帧的结果)

# 合并后的输出文件路径
output_path = '/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/b1_b2_merge/merged_point_cloud.ply'
# ------------------------


def merge_ply_files(file1, file2, output_file):
    """
    合并两个 3DGS PLY 文件，保留所有属性。
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 找不到文件 {file1}")
        return
    if not os.path.exists(file2):
        print(f"错误: 找不到文件 {file2}")
        return

    try:
        # 读取第一个 PLY 文件
        print(f"正在读取文件 1: {file1}")
        ply1 = PlyData.read(file1)
        data1 = ply1['vertex'].data
        print(f"文件 1 顶点数: {len(data1)}")
    except Exception as e:
        print(f"读取文件 {file1} 时出错: {e}")
        return

    try:
        # 读取第二个 PLY 文件
        print(f"正在读取文件 2: {file2}")
        ply2 = PlyData.read(file2)
        data2 = ply2['vertex'].data
        print(f"文件 2 顶点数: {len(data2)}")
    except Exception as e:
        print(f"读取文件 {file2} 时出错: {e}")
        return

    # --- 属性一致性检查 (非常重要) ---
    props1 = [p.name for p in ply1['vertex'].properties]
    props2 = [p.name for p in ply2['vertex'].properties]
    
    if props1 != props2:
        print("错误: 两个PLY文件的顶点属性不一致！")
        print("这可能意味着它们来自不同的训练设置或版本。")
        print(f"文件 1 属性: {props1}")
        print(f"文件 2 属性: {props2}")
        print("无法合并。")
        return
    
    print("文件属性一致，可以安全合并。")

    # --- 合并数据 ---
    # `data1` 和 `data2` 是 NumPy 结构化数组
    # 我们可以使用 np.concatenate 来合并它们
    combined_data = np.concatenate((data1, data2))
    total_vertices = len(combined_data)
    print(f"合并后总顶点数: {total_vertices}")

    # --- 创建新的 PLY 结构 ---
    # 1. 从合并后的数据创建一个新的 PlyElement
    #    PlyElement.describe 会自动从 structured array 推断出所有属性
    combined_vertex_element = PlyElement.describe(combined_data, 'vertex')
    
    # 2. 检查原始文件中是否有其他元素 (3DGS 通常没有, 但以防万一)
    other_elements = [el for el in ply1.elements if el.name != 'vertex']
    
    # 3. 创建一个新的 PlyData 对象
    new_ply_data = PlyData([combined_vertex_element] + other_elements)
    
    # 继承第一个文件的格式（例如是 'binary_little_endian' 还是 'ascii'）
    is_text = ply1.is_text
    byte_order = ply1.byte_order

    # --- 写入新的 PLY 文件 ---
    print(f"正在写入合并后的文件: {output_path}")
    new_ply_data.write(output_path, text=is_text, byte_order=byte_order)
    
    print("="*30)
    print(f"合并完成！文件已保存至: {output_path}")
    print("="*30)


if __name__ == "__main__":
    # 运行合并函数
    merge_ply_files(file1_path, file2_path, output_path)