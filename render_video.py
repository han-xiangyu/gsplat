import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
from collections import defaultdict

def resize_and_crop_image(image, target_size):
    return image  # 原样返回，因为未使用

def add_label_to_image(image, label):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)
    font_path = "/home/neptune/Downloads/optima/OPTIMA.TTF"  # 根据实际字体路径修改
    try:
        font = ImageFont.truetype(font_path, size=160)
    except IOError:
        print(f"无法加载字体 {font_path}，使用默认字体")
        font = ImageFont.load_default()

    text = label
    try:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width, text_height = font.getsize(text)

    x = 10
    y = image_pil.height - text_height - 25
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    image_with_label = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image_with_label

def parse_folder(folder_path):
    """
    解析GT文件夹，用正则解析 loc_<loc_id>_trav_<trav_id>_channel_<channel_id>_img_<frame_id>.(png|jpg)
    返回结构: parsed[trav_id][frame_id][channel_id] = path
    """
    # match an optional loc_ prefix, then trav, channel and img, with either .png or .jpg
    # pattern = r'^(?:loc_\d+_)?trav_(\d+)_channel_(\d+)_img_(\d+)\.(?:png|jpe?g)$'
    pattern = r'^trav_(\d+)_channel_(\d+)_img_(\d+)\.(?:png|jpe?g)$'
    parsed = defaultdict(lambda: defaultdict(dict))
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    matched_count = 0
    unmatched_files = []
    for f in image_files:
        m = re.match(pattern, f)
        if m:
            trav_id   = int(m.group(1))
            channel_id= int(m.group(2))
            frame_id  = int(m.group(3))
            full_path = os.path.join(folder_path, f)
            parsed[trav_id][frame_id][channel_id] = full_path
            matched_count += 1
        else:
            unmatched_files.append(f)

    print(f"[DEBUG] Parsing folder: {folder_path}")
    print(f" - Total images: {len(image_files)}, Matched: {matched_count}, Unmatched: {len(unmatched_files)}")
    if unmatched_files:
        print(" - Unmatched files:", unmatched_files)
    return parsed


def calculate_valid_frame_indices(gt_folder_path):
    """
    从GT文件夹中解析有效帧。有效帧定义：front/left/right都存在且可读。
    不依赖于log分配和复杂逻辑，找到所有 valid_frames 后，
    构建GT的全局image list，对每个 valid_frame 的front/left/right在此list中查找其索引。
    """
    gt_parsed = parse_folder(gt_folder_path)

    print("[DEBUG] In calculate_valid_frame_indices:")
    trav_ids = sorted(gt_parsed.keys())
    total_valid_frames = 0

    valid_frames = []
    for t_id in trav_ids:
        frame_ids = sorted(gt_parsed[t_id].keys())
        print(f"   - trav_id={t_id}: {len(frame_ids)} frames")
        for frame_id in frame_ids:
            channels = gt_parsed[t_id][frame_id]
            if all(ch in channels for ch in [1,2,3]):
                front_path = channels[1]
                left_path = channels[2]
                right_path = channels[3]
                front_img = cv2.imread(front_path)
                left_img = cv2.imread(left_path)
                right_img = cv2.imread(right_path)
                if front_img is not None and left_img is not None and right_img is not None:
                    valid_frames.append({
                        'trav_id': t_id,
                        'frame_id': frame_id,
                        'front': front_path,
                        'left': left_path,
                        'right': right_path
                    })

    print(f" - Total valid frames from GT: {len(valid_frames)}")
    if valid_frames:
        print(" - Example:", valid_frames[0])
    else:
        print(" - No valid frames found.")
        return [], None, [], []

    # 确定 target_size（以第一张front图为准）
    first_image_path = valid_frames[0]['front']
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print("   - Cannot read the first front image for sizing.")
        return [], None, [], []
    target_size = first_image.shape[:2]

    # 获取GT文件夹的所有图片列表，并构建路径->索引的映射
    gt_image_files = [f for f in os.listdir(gt_folder_path) if f.lower().endswith(('.jpg', '.png'))]
    gt_image_files.sort()
    # 构建全路径到index的映射
    path_to_index = {os.path.join(gt_folder_path, f): i for i, f in enumerate(gt_image_files)}

    valid_global_indices = []
    # 为每个有效帧的front/left/right分配全局索引（根据GT文件夹的排序）
    for frame_data in valid_frames:
        front_idx = path_to_index[frame_data['front']]
        left_idx = path_to_index[frame_data['left']]
        right_idx = path_to_index[frame_data['right']]
        frame_data['front_idx'] = front_idx
        frame_data['left_idx'] = left_idx
        frame_data['right_idx'] = right_idx
        valid_global_indices.extend([front_idx, left_idx, right_idx])

    return valid_frames, target_size, valid_global_indices, gt_image_files

def read_and_concatenate_images(base_paths, valid_frames, baseline_labels, target_size, valid_global_indices):
    all_concatenated_images = []
    all_image_lists = []

    # 为每个baseline文件夹获取已排序的图片列表
    for base_path in base_paths:
        image_files = [f for f in os.listdir(base_path) if f.lower().endswith(('.jpg', '.png'))]
        image_files.sort()
        all_image_lists.append(image_files)

    total_frames = len(valid_frames)
    print(f"[DEBUG] Total valid frames from GT: {total_frames}")

    for i in range(total_frames):
        row_images = []
        skip_frame = False

        # 对每帧，根据valid_global_indices获取front/left/right的索引
        front_idx = valid_global_indices[3*i]
        left_idx = valid_global_indices[3*i + 1]
        right_idx = valid_global_indices[3*i + 2]

        for idx, (base_path, baseline_label, image_files) in enumerate(zip(base_paths, baseline_labels, all_image_lists)):
            # 检查索引范围
            if (front_idx >= len(image_files) or left_idx >= len(image_files) or right_idx >= len(image_files)):
                print(f"[WARNING] {base_path} 中图片数量不足，无法获取第 {i} 帧的front/left/right图片，跳过该帧。")
                skip_frame = True
                break

            front_img_path = os.path.join(base_path, image_files[front_idx])
            left_img_path = os.path.join(base_path, image_files[left_idx])
            right_img_path = os.path.join(base_path, image_files[right_idx])

            front_img = cv2.imread(front_img_path)
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)

            if front_img is None or left_img is None or right_img is None:
                print(f"[WARNING] 第 {i} 帧无法读取图片：{front_img_path}, {left_img_path}, {right_img_path}，跳过该帧。")
                skip_frame = True
                break

            front_img = cv2.resize(front_img, (target_size[1], target_size[0]))
            left_img = cv2.resize(left_img, (target_size[1], target_size[0]))
            right_img = cv2.resize(right_img, (target_size[1], target_size[0]))

            left_img_with_label = add_label_to_image(left_img, baseline_label)
            concatenated_row = np.hstack((left_img_with_label, front_img, right_img))
            row_images.append(concatenated_row)

        if skip_frame or len(row_images) != len(base_paths):
            continue

        concatenated_img = np.vstack(row_images)
        all_concatenated_images.append(concatenated_img)

    print(f"[DEBUG] Total concatenated frames generated: {len(all_concatenated_images)}")
    return all_concatenated_images

def generate_video_from_folders(base_paths, output_video_path, baseline_labels, fps=30):
    gt_folder = base_paths[0]
    valid_frames, target_size, valid_global_indices, gt_image_files = calculate_valid_frame_indices(gt_folder)

    if not valid_frames:
        print("没有有效的帧可用于生成视频。")
        return

    all_concatenated_images = read_and_concatenate_images(
        base_paths, valid_frames, baseline_labels, target_size, valid_global_indices
    )

    if not all_concatenated_images:
        print("没有有效的图片用于生成视频。")
        return

    height, width, layers = all_concatenated_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img in all_concatenated_images:
        video_writer.write(img)

    video_writer.release()
    print(f"视频已保存到 {output_video_path}")

if __name__ == "__main__":
    model_folder = "/lustre/fsw/portfolios/nvr/users/ymingli/gaussian/models/long_video_frames6000_full_autoresume_distributed8GPU/"
    baseline_labels = ["Ground truth", "GrendelGS"]

    train_paths = [
        os.path.join(model_folder, "train/ours_99993/gt"),
        os.path.join(model_folder, "train/ours_99993/renders"),
    ]

    output_video_path = os.path.join(model_folder, "train_set_video.mp4")
    generate_video_from_folders(train_paths, output_video_path, baseline_labels=baseline_labels, fps=10)

    # test_paths = [
    #     os.path.join(model_folder, "test/ours_80000/gt"),
    #     os.path.join(model_folder, "test/ours_80000/renders"),
    # ]

    # output_video_path = os.path.join(model_folder, "test_set_video.mp4")
    # generate_video_from_folders(test_paths, output_video_path, baseline_labels=baseline_labels, fps=7)
