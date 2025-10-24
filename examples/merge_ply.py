import numpy as np
from plyfile import PlyData, PlyElement
import sys
import os
import time
from tqdm import tqdm

# --- 你的文件路径 (已从你的代码中复制) ---
file1_path = "/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/tra2_3000keyframes_fps_3cam/ply/point_cloud_99999.ply"
file2_path = "/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/tra2_3000to6000keyframes_fps_3cam/ply/point_cloud_99999.ply" 
output_path = '/lustre/fsw/portfolios/nvr/users/ymingli/datasets/citygs/models/b1_b2_merge/merged_point_cloud.ply'
# ------------------------------------


def merge_ply_files(file1, file2, output_file):
    """
    合并两个 3DGS PLY 文件，保留所有属性，并显示基于步骤的进度。
    """
    
    # 检查输出目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except Exception as e:
            print(f"创建目录 {output_dir} 失败: {e}")
            return
            
    # 检查输入文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 找不到文件 {file1}")
        return
    if not os.path.exists(file2):
        print(f"错误: 找不到文件 {file2}")
        return

    # --- TQDM 进度条设置 ---
    total_steps = 5
    pbar = tqdm(total=total_steps, desc="开始合并", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")
    
    total_vertices = 0

    try:
        # --- 步骤 1: 读取文件 1 ---
        pbar.set_description("步骤 1/5: 读取文件 1")
        pbar.set_postfix_str(f"{os.path.basename(file1)} (可能需几分钟)")
        start_time = time.time()
        
        ply1 = PlyData.read(file1)
        data1 = ply1['vertex'].data
        
        pbar.update(1)
        read_time1 = time.time() - start_time
        tqdm.write(f"\n[步骤 1/5] 文件 1 读取完毕: {len(data1)} 个顶点 (用时: {read_time1:.2f} 秒)")


        # --- 步骤 2: 读取文件 2 ---
        pbar.set_description("步骤 2/5: 读取文件 2")
        pbar.set_postfix_str(f"{os.path.basename(file2)} (可能需几分钟)")
        start_time = time.time()

        ply2 = PlyData.read(file2)
        data2 = ply2['vertex'].data
        
        pbar.update(1)
        read_time2 = time.time() - start_time
        tqdm.write(f"[步骤 2/5] 文件 2 读取完毕: {len(data2)} 个顶点 (用时: {read_time2:.2f} 秒)")

        # --- 属性一致性检查 ---
        props1 = [p.name for p in ply1['vertex'].properties]
        props2 = [p.name for p in ply2['vertex'].properties]
        
        if props1 != props2:
            tqdm.write("\n" + "="*30)
            tqdm.write("错误: 两个PLY文件的顶点属性不一致！")
            tqdm.write(f"文件 1 属性: {props1}")
            tqdm.write(f"文件 2 属性: {props2}")
            tqdm.write("无法合并。")
            tqdm.write("="*30 + "\n")
            return

        # --- 步骤 3: 合并数据 (内存操作) ---
        pbar.set_description("步骤 3/5: 合并数据")
        pbar.set_postfix_str("np.concatenate (内存操作)")
        start_time = time.time()

        combined_data = np.concatenate((data1, data2))
        total_vertices = len(combined_data)
        
        pbar.update(1)
        concat_time = time.time() - start_time
        tqdm.write(f"[步骤 3/5] 数据合并完毕: 共 {total_vertices} 个顶点 (用时: {concat_time:.2f} 秒)")

        # --- 步骤 4: 创建新的 PLY 结构 ---
        pbar.set_description("步骤 4/5: 准备 PLY 结构")
        pbar.set_postfix_str("创建 PlyElement")
        start_time = time.time()

        combined_vertex_element = PlyElement.describe(combined_data, 'vertex')
        other_elements = [el for el in ply1.elements if el.name != 'vertex']
        
        #
        # *********** 修正点 ***********
        # 继承第一个文件的格式
        is_text = ply1.text
        byte_order = ply1.byte_order
        
        # 在创建 PlyData 对象时传入格式参数
        new_ply_data = PlyData([combined_vertex_element] + other_elements,
                               text=is_text,
                               byte_order=byte_order)
        # ****************************
        
        pbar.update(1)
        prep_time = time.time() - start_time
        tqdm.write(f"[步骤 4/5] PLY 结构创建完毕 (用时: {prep_time:.2f} 秒)")

        # --- 步骤 5: 写入新的 PLY 文件 (最耗时) ---
        pbar.set_description("步骤 5/5: 写入文件")
        pbar.set_postfix_str(f"写入 {os.path.basename(output_file)} (此步最久!)")
        start_time = time.time()

        #
        # *********** 修正点 ***********
        # .write() 方法只接受文件名作为参数
        new_ply_data.write(output_path)
        # ****************************
        
        pbar.update(1)
        write_time = time.time() - start_time
        tqdm.write(f"[步骤 5/5] 文件写入完毕 (用时: {write_time:.2f} 秒)")

    except Exception as e:
        tqdm.write(f"\n合并过程中发生错误: {e}")
        import traceback
        tqdm.write(traceback.format_exc())
    finally:
        pbar.close()
        if total_vertices > 0 and os.path.exists(output_file):
            print("\n" + "="*30)
            print(f"合并完成！文件已保存至: {output_path}")
            print(f"总计 {total_vertices} 个顶点。")
            print("="*30)
        else:
            print("\n合并未完成或失败。")


if __name__ == "__main__":
    # 运行合并函数
    merge_ply_files(file1_path, file2_path, output_path)