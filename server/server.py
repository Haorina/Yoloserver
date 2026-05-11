import os
import sys
import cv2
import time
import json
import torch
import warnings
import traceback
import numpy as np

# 設置 Server 為無 GUI 的模式(避免圖表的生成)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import *
from flask_cors import *
from collections import *
from datetime import datetime
from mpl_toolkits.mplot3d.axes3d import *

# 獲取當前資料夾與父資料夾路徑，雙重保險加入系統路徑
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir not in sys.path: sys.path.insert(1, parent_dir)

# 引入自定義的模組
from src.mediapipe_lib.base import PoseResult, ResultAnalyzer
from src.value import KPT_LIST, KPT_IDX_DICT, coco_line
from src.model_wrappers import YOLOPoseEstimator, MotionBERTEstimator

# 創建 Server
server = Flask(__name__)
CORS(server)

# 設置 Server 的相關配置
server.config['UPLOAD_FOLDER'] = 'Uploads/'
server.config['ALLOWED_EXTENSIONS'] = {'mp4'}
server.config['KPT_FOLDER'] = 'kpt/'
server.config['JOINT_ANGLES_FOLDER'] = 'jointangles/'
server.config['OVERLAY_FOLDER'] = 'overlay/'

# 創建 Server 相關的資料夾
os.makedirs(server.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('pose_change', exist_ok=True)
os.makedirs(server.config['KPT_FOLDER'], exist_ok=True)
os.makedirs(server.config['JOINT_ANGLES_FOLDER'], exist_ok=True)
os.makedirs(server.config['OVERLAY_FOLDER'], exist_ok=True)

warnings.filterwarnings("ignore")

# 仰角降到 10 度 (平視)，視角只微轉 10 度 (保留正面的直覺)
default_view = (10, -80, 0)
#轉側面
#default_view = (10, 0, 0)
fig_width = 15
fig_height = 15

def get_a_3d_pose_plot_image(kpts_3d_dict: dict, output_path: str, figsize=(fig_width, fig_height), view=default_view, dot_size=15, colors=("#000", "#f00", "#0f0", "#00f")):
    fig = plt.figure(figsize=figsize)
    ax: Axes3D = fig.add_subplot(projection="3d")
    plot_range = 1.0 
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range) 
    ax.set_zlim(-plot_range, plot_range) 
    ax.set_box_aspect([1, 1, 1])
            
    ax.view_init(view[0], view[1], view[2])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    
    if len(kpts_3d_dict) > 0:
        for name, pos in kpts_3d_dict.items():
            if pos is not None and len(pos) == 3:
                ax.scatter(pos[0], pos[2], -pos[1], c=colors[0], s=dot_size)
        def draw_lines_by_pairs(pairs, color):
            for pt1_name, pt2_name in pairs:
                if pt1_name in kpts_3d_dict and pt2_name in kpts_3d_dict:
                    p1, p2 = kpts_3d_dict[pt1_name], kpts_3d_dict[pt2_name]
                    if p1 and p2:
                        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], c=color)

        draw_lines_by_pairs(coco_line.left_kpt, colors[1])   
        draw_lines_by_pairs(coco_line.center_kpt, colors[2]) 
        draw_lines_by_pairs(coco_line.right_kpt, colors[3])  
    
    plt.savefig(output_path, transparent=True)
    plt.close(fig)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in server.config['ALLOWED_EXTENSIONS']

@server.route('/')
def index():
    return render_template('index.html')

@server.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
        file_path = os.path.join(server.config['UPLOAD_FOLDER'], f"{timestamp}.mp4")
        file.save(file_path)
        result = process_video(file_path, timestamp)
        response = server.response_class(
            response=json.dumps(result, ensure_ascii=False),
            mimetype='application/json'
        )
        return response

def process_video(video_path, timestamp):
    checkPersonInScreen = False
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[{timestamp}] Using compute device: {device}")

        yolo_estimator = YOLOPoseEstimator(model_path='yolov8n-pose.pt', device=device)
        mbert_path = os.path.join(current_dir, 'models', 'motionbert_h36m.bin')
        if not os.path.exists(mbert_path): raise FileNotFoundError(f"找不到 MotionBERT 模型檔: {mbert_path}")
        mbert_estimator = MotionBERTEstimator(model_path=mbert_path, device=device)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ⭐ 嘗試獲取手機錄影的旋轉元數據 (Metadata)
        rotation_angle = 0
        try:
            meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if meta in [90, 180, 270]:
                rotation_angle = int(meta)
        except Exception:
            pass

        # ⭐ 如果有旋轉，長寬度必須跟著對調，否則輸出會變形
        if rotation_angle == 90 or rotation_angle == 270:
            video_width, video_height = video_height, video_width

        # 階段一
        print(f"[{timestamp}] Stage 1: Extracting 2D Poses & Creating Overlay...")
        timestamp_overlay_folder = os.path.join(server.config['OVERLAY_FOLDER'], timestamp)
        os.makedirs(timestamp_overlay_folder, exist_ok=True)
        overlay_video_path = os.path.join(timestamp_overlay_folder, f"{timestamp}_overlay.mp4")
        overlay_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overlay_video = cv2.VideoWriter(overlay_video_path, overlay_fourcc, fps, (video_width, video_height))

        all_2d_poses = [] 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # ⭐ 根據前面抓到的角度，把每一幀轉正！
            if rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            kpts_2d, annotated_frame = yolo_estimator.predict(frame)
            all_2d_poses.append(kpts_2d)
            overlay_video.write(annotated_frame)
            
        cap.release()
        overlay_video.release()
        if len(all_2d_poses) == 0: raise ValueError("No frames processed or no person detected in video.")
        seq_2d_matrix = np.array(all_2d_poses)

        # 階段二
        print(f"[{timestamp}] Stage 2: Lifting to 3D with MotionBERT...")
        all_3d_poses_sequence = mbert_estimator.predict(seq_2d_matrix, video_width, video_height)

        # 階段三
        print(f"[{timestamp}] Stage 3: Ergonomics Evaluation (KIM-LHC)...")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        A_array, B_array, C_array, D_array = [], [], [], []
        frame_counts, labels, joint_angles_per_frame = [], [], []

        pose_mapping = {'A1': 1, 'A2': 2, 'A3-1': 3, 'A3-2': 4, 'A4-1': 5, 'A4-2': 6, 'A5-1': 7, 'A5-2': 8, 'A5-3': 9}
        pose_label_reverse = {v: k for k, v in pose_mapping.items()}
        fine_to_coarse = {1:1, 2:2, 3:3, 4:3, 5:4, 6:4, 7:5, 8:5, 9:5}
        coarse_to_rep_fine = {1: 1, 2: 2, 3: 3, 4: 5, 5: 7}
        coarse_name = {1: "A1", 2: "A2", 3: "A3", 4: "A4", 5: "A5"}
        coarse_bg_colors = {1: "#FFF2CC", 2: "#E2EFDA", 3: "#D9E1F2", 4: "#F8CECC", 5: "#EAD1DC"}

        def add_horizontal_group_background(ax, alpha=0.8):
            bands = [(1,1,1), (2,2,2), (3,4,3), (5,6,4), (7,9,5)]
            for y0, y1, coarse_id in bands:
                ax.axhspan(y0 - 0.5, y1 + 0.5, facecolor=coarse_bg_colors.get(coarse_id, "#eeeeee"), alpha=alpha, linewidth=0, zorder=0)

        pose_scores = {
            (1, 1): 0,  (1, 2): 3,  (1, 3): 3,  (1, 4): 7,  (1, 5): 9,
            (2, 2): 5,  (2, 3): 5,  (2, 4): 10, (2, 5): 13,
            (3, 3): 5,  (3, 4): 10, (3, 5): 13,
            (4, 4): 15, (4, 5): 18, (5, 5): 20
        }

        timestamp_folder = os.path.join(server.config['KPT_FOLDER'], timestamp)
        os.makedirs(timestamp_folder, exist_ok=True)
        kpt_video_path = os.path.join(timestamp_folder, "video")
        os.makedirs(kpt_video_path, exist_ok=True)
        kpt_image_path = os.path.join(timestamp_folder, "images")
        os.makedirs(kpt_image_path, exist_ok=True)
        kpt_video_name = os.path.join(kpt_video_path, f"{timestamp}.mp4")
        joint_angles_image_path = os.path.join(server.config['JOINT_ANGLES_FOLDER'], timestamp)
        os.makedirs(joint_angles_image_path, exist_ok=True)

        kpt_video = None
        kpt_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= len(all_3d_poses_sequence): break
            current_3d_matrix = all_3d_poses_sequence[frame_count]
            current_3d_dict = {KPT_LIST[i]: current_3d_matrix[i].tolist() for i in range(17)}
            
            landmarks_file = os.path.join(timestamp_folder, f"landmarks_{frame_count}.txt")
            with open(landmarks_file, "w") as f:
                for name, pos in current_3d_dict.items():
                    f.write(f"{name}: x={pos[0]:.10f}, y={pos[1]:.10f}, z={pos[2]:.10f}\n")

            pose_result = PoseResult(current_3d_dict)
            analyzer = ResultAnalyzer(pose_result)
            is_3d = True

            label = analyzer.get_lhc_label(is_3d)
            frame_counts.append(frame_count)
            labels.append(pose_mapping.get(label, 0))

            is_person_detected = np.any(current_3d_matrix)
            if is_person_detected:
                joint_angles = analyzer.get_all_joint_angles_by_name(is_3d)
                
                # 同步真實軀幹重力角度給圖表與 JSON
                try:
                    shoulder_l = np.array(analyzer.pose_result.get_kpt_pos_by_name("left_shoulder", True))
                    shoulder_r = np.array(analyzer.pose_result.get_kpt_pos_by_name("right_shoulder", True))
                    hip_l = np.array(analyzer.pose_result.get_kpt_pos_by_name("left_hip", True))
                    hip_r = np.array(analyzer.pose_result.get_kpt_pos_by_name("right_hip", True))
                    mid_shoulder = (shoulder_l + shoulder_r) / 2.0
                    mid_hip = (hip_l + hip_r) / 2.0
                    trunk_vec = mid_shoulder - mid_hip
                    norm_trunk = np.linalg.norm(trunk_vec)
                    if norm_trunk > 0:
                        cos_a = np.clip(np.dot(trunk_vec, np.array([0, -1, 0])) / norm_trunk, -1.0, 1.0)
                        true_hip_angle = 180.0 - np.degrees(np.arccos(cos_a))
                        joint_angles["left_hip"] = true_hip_angle
                        joint_angles["right_hip"] = true_hip_angle
                except Exception as e:
                    pass

                joint_angles_per_frame.append({
                    "frame": frame_count,
                    "angles": {
                        "left_shoulder": joint_angles.get("left_shoulder", None),
                        "right_shoulder": joint_angles.get("right_shoulder", None),
                        "left_hip": joint_angles.get("left_hip", None),
                        "right_hip": joint_angles.get("right_hip", None),
                        "left_knee": joint_angles.get("left_knee", None),
                        "right_knee": joint_angles.get("right_knee", None),
                    }
                })
                A_array.append(analyzer.check_if_trunk_is_twisted_or_lateral_inclination(is_3d))
                B_array.append(analyzer.check_if_hands_at_a_distance(is_3d))
                C_array.append(analyzer.check_if_arms_raised(is_3d))
                D_array.append(analyzer.check_if_hands_above_shoulder(is_3d))
            else:
                joint_angles_per_frame.append({
                    "frame": frame_count,
                    "angles": {k: None for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"]}
                })
                A_array.append(False); B_array.append(False); C_array.append(False); D_array.append(False)

            kpt_image_name = os.path.join(kpt_image_path, f"{timestamp}_{frame_count}.png")
            get_a_3d_pose_plot_image(current_3d_dict, kpt_image_name)

            frame_img = cv2.imread(kpt_image_name)
            if frame_img is not None:
                if kpt_video is None:
                    h, w = frame_img.shape[:2]
                    kpt_video = cv2.VideoWriter(kpt_video_name, kpt_fourcc, fps, (w, h))
                kpt_video.write(frame_img)
            frame_count += 1
        
        cap.release()
        if kpt_video is not None: kpt_video.release()
        total_frames = frame_count

        if joint_angles_per_frame:
            plt.figure(figsize=(20, 6))
            frames_x = [frame["frame"] for frame in joint_angles_per_frame]
            for joint in ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"]:
                angles = [frame["angles"][joint] for frame in joint_angles_per_frame]
                plt.plot(frames_x, angles, label=joint)
            plt.title('Frame-by-frame Joint Angles')
            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
            plt.grid(True)
            plt.xlim(0, len(frame_counts))
            plt.ylim(0, 180)
            plt.xticks(range(0, len(frame_counts), max(1, int(len(frame_counts) / 10))))
            plt.savefig(os.path.join(joint_angles_image_path, f"{timestamp}_jointangles.png"))
            plt.close()

        plt.figure(figsize=(20, 6))
        ax_before = plt.gca()
        add_horizontal_group_background(ax_before, alpha=0.8)
        plt.step(frame_counts, labels, marker='o', color='blue', markersize=6, label='Frames')
        plt.title('Frame-by-frame pose change diagram (before)')
        plt.text(x=5, y=10.0, s=f'Total frames: {total_frames}', fontsize=12, bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round, pad=0.5'))
        plt.legend(loc='upper right', fontsize=12, shadow=True)
        plt.xlabel('Frame')
        plt.ylabel('Pose')
        plt.xlim(0, len(frame_counts))
        plt.yticks(range(1, 10), ['A1', 'A2', 'A3-1', 'A3-2', 'A4-1', 'A4-2', 'A5-1', 'A5-2', 'A5-3'])
        plt.ylim(0, 10.5)
        plt.xticks(range(0, len(frame_counts), max(1, int(len(frame_counts) / 10))))
        plt.grid(True)
        plt.savefig(os.path.join('pose_change', f'before.png'))
        plt.close()

        coarse_labels = [fine_to_coarse.get(x, 1) if x is not None else None for x in labels]
        csv_time = fps * 3

        def get_con(coarse_label):
            match coarse_label:
                case 1: return 20
                case 2: return 10
                case 3: return 15
                case 4: return 10
                case 5: return 5
                case _: return 1
        def majority_fine_label(start_idx, end_idx_exclusive, coarse):
            seq = [labels[j] for j in range(start_idx, end_idx_exclusive) if labels[j] in fine_to_coarse and fine_to_coarse[labels[j]] == coarse]
            if not seq: return coarse_to_rep_fine.get(coarse, None)
            cnt = Counter(seq)
            return sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]

        label_changes_coarse = []
        label_changes_fine = []
        
        if len(coarse_labels) > 0:
            current_coarse = coarse_labels[0]
            con = get_con(current_coarse)
            consecutive_count = 0
            segment_start = 0
            previous_coarse = None
            previous_count = 0
            previous_fine = None
            isFirst = False

            for i in range(len(coarse_labels)):
                cl = coarse_labels[i]
                if cl == current_coarse: consecutive_count += 1
                elif consecutive_count < con:
                    for j in range(1, consecutive_count + 1):
                        idx = i - j
                        A_array[idx] = B_array[idx] = C_array[idx] = D_array[idx] = 0
                        labels[idx] = None
                    total_frames -= consecutive_count
                    current_coarse = cl
                    con = get_con(current_coarse)
                    consecutive_count = 1
                    segment_start = i
                else:
                    fine_rep = majority_fine_label(segment_start, i, current_coarse)
                    if not isFirst:
                        previous_coarse = current_coarse
                        previous_count = consecutive_count
                        previous_fine = fine_rep
                        isFirst = True
                    elif current_coarse == previous_coarse:
                        previous_count += consecutive_count
                    else:
                        label_changes_coarse.append(previous_coarse)
                        label_changes_fine.append(previous_fine)
                        tmp = previous_count
                        while tmp >= csv_time:
                            label_changes_coarse.append(previous_coarse)
                            label_changes_fine.append(previous_fine)
                            tmp -= csv_time
                        previous_coarse = current_coarse
                        previous_count = consecutive_count
                        previous_fine = fine_rep

                    current_coarse = cl
                    con = get_con(current_coarse)
                    consecutive_count = 1
                    segment_start = i

                if i == len(coarse_labels) - 1:
                    if consecutive_count < con:
                        for j in range(len(coarse_labels) - consecutive_count, len(coarse_labels)):
                            A_array[j] = B_array[j] = C_array[j] = D_array[j] = 0
                            labels[j] = None
                        total_frames -= consecutive_count
                        if isFirst:
                            label_changes_coarse.append(previous_coarse)
                            label_changes_fine.append(previous_fine)
                            tmp = previous_count
                            while tmp >= csv_time:
                                label_changes_coarse.append(previous_coarse)
                                label_changes_fine.append(previous_fine)
                                tmp -= csv_time
                    else:
                        fine_rep_end = majority_fine_label(segment_start, i + 1, current_coarse)
                        if not isFirst:
                            label_changes_coarse.append(current_coarse)
                            label_changes_fine.append(fine_rep_end)
                            tmp = consecutive_count
                            while tmp >= csv_time:
                                label_changes_coarse.append(current_coarse)
                                label_changes_fine.append(fine_rep_end)
                                tmp -= csv_time
                        else:
                            if previous_coarse == current_coarse:
                                total_len = previous_count + consecutive_count
                                label_changes_coarse.append(previous_coarse)
                                label_changes_fine.append(previous_fine)
                                tmp = total_len
                                while tmp >= csv_time:
                                    label_changes_coarse.append(previous_coarse)
                                    label_changes_fine.append(previous_fine)
                                    tmp -= csv_time
                            else:
                                label_changes_coarse.append(previous_coarse)
                                label_changes_fine.append(previous_fine)
                                tmp = previous_count
                                while tmp >= csv_time:
                                    label_changes_coarse.append(previous_coarse)
                                    label_changes_fine.append(previous_fine)
                                    tmp -= csv_time
                                label_changes_coarse.append(current_coarse)
                                label_changes_fine.append(fine_rep_end)
                                tmp = consecutive_count
                                while tmp >= csv_time:
                                    label_changes_coarse.append(current_coarse)
                                    label_changes_fine.append(fine_rep_end)
                                    tmp -= csv_time

        best_pair = (None, None)
        max_score = 0
        fine_start = None
        fine_end = None

        if len(label_changes_coarse) == 1:
            pair = (label_changes_coarse[0], label_changes_coarse[0])
            max_score = pose_scores.get(pair, 0)
            best_pair = pair
        elif len(label_changes_coarse) >= 2:
            for i in range(len(label_changes_coarse) - 1):
                pair1 = (label_changes_coarse[i], label_changes_coarse[i + 1])
                pair2 = (label_changes_coarse[i + 1], label_changes_coarse[i])
                score = pose_scores.get(pair1, pose_scores.get(pair2, 0))
                if score > max_score:
                    max_score = score
                    best_pair = pair1

        worst_fine_text = "N/A"
        if best_pair[0] is not None and best_pair[1] is not None:
            for i in range(len(label_changes_coarse) - 1):
                if (label_changes_coarse[i], label_changes_coarse[i + 1]) == best_pair:
                    fine_start = label_changes_fine[i]
                    fine_end = label_changes_fine[i + 1]
                    break
            if fine_start is None: fine_start = coarse_to_rep_fine.get(best_pair[0])
            if fine_end is None: fine_end = coarse_to_rep_fine.get(best_pair[1])
            if fine_start is not None and fine_end is not None:
                worst_fine_text = f"{pose_label_reverse.get(fine_start, f'A{fine_start}')} to {pose_label_reverse.get(fine_end, f'A{fine_end}')}"

        temp_list = labels.copy()
        for i in range(len(temp_list)):
            if (temp_list[i] is None) and (i > 0) and (temp_list[i - 1] is not None): temp_list[i] = temp_list[i - 1]
        for i in reversed(range(len(temp_list) - 1)):
            if (temp_list[i] is None) and (temp_list[i + 1] is not None): temp_list[i] = temp_list[i + 1]

        plt.figure(figsize=(20, 6))
        ax_after = plt.gca()
        add_horizontal_group_background(ax_after, alpha=0.8)
        plt.step(frame_counts, temp_list, marker='o', color='red', markersize=6, label='Invalid frames')
        plt.step(frame_counts, labels, marker='o', color='blue', markersize=6, label='Valid frames')
        plt.title('Frame-by-frame pose change diagram (after)')
        pose_changes_text_coarse = ", ".join(coarse_name.get(x, f"A{x}") for x in label_changes_coarse) if label_changes_coarse else "N/A"
        pose_changes_text_fine = ", ".join(pose_label_reverse.get(x, f"A{x}") for x in label_changes_fine) if label_changes_fine else "N/A"
        worst_change_text_coarse = f'{coarse_name.get(best_pair[0], f"A{best_pair[0]}")} to {coarse_name.get(best_pair[1], f"A{best_pair[1]}")}' if best_pair[0] is not None else "N/A"
        plt.text(x=5, y=9.5, s=(f'Pose changes (coarse): {pose_changes_text_coarse}\nPose changes (fine):   {pose_changes_text_fine}\nThe worst pose change (coarse): {worst_change_text_coarse} ({max_score} scores)\nThe worst pose change (fine):   {worst_fine_text}\nValid frames: {total_frames}'), fontsize=12, bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round, pad=0.5'))
        plt.legend(loc='upper right', fontsize=12, shadow=True)
        plt.xlabel('Frame')
        plt.ylabel('Pose')
        plt.xlim(0, len(frame_counts))
        plt.yticks(range(1, 10), ['A1', 'A2', 'A3-1', 'A3-2', 'A4-1', 'A4-2', 'A5-1', 'A5-2', 'A5-3'])
        plt.ylim(0, 10.5)
        plt.xticks(range(0, len(frame_counts), max(1, int(len(frame_counts) / 10))))
        plt.grid(True)
        plt.savefig(os.path.join('pose_change', f'after.png'))
        plt.close()

        def get_frequency_score(ratio, none_score, occasional_score, frequent_score):
            if ratio < 1 / 9: return none_score
            elif ratio < 1 / 3: return occasional_score
            else: return frequent_score

        if total_frames > 0:
            A_ratio = sum(1 for x in A_array if x) / total_frames
            B_ratio = sum(1 for x in B_array if x) / total_frames
            C_ratio = sum(1 for x in C_array if x) / total_frames
            D_ratio = sum(1 for x in D_array if x) / total_frames
        else:
            A_ratio = B_ratio = C_ratio = D_ratio = 0.0

        A_score = get_frequency_score(A_ratio, 0, 1, 3)
        B_score = get_frequency_score(B_ratio, 0, 1, 3)
        C_score = get_frequency_score(C_ratio, 0, 0.5, 1)
        D_score = get_frequency_score(D_ratio, 0, 1, 2)
        extra_score = min(A_score + B_score + C_score + D_score, 6)

        plt.figure(figsize=(20, 6))
        plt.step(frame_counts, [1 if x else None for x in A_array], marker='o', linestyle='None', color='red', markersize=6)
        plt.step(frame_counts, [2 if x else None for x in B_array], marker='o', linestyle='None', color='orange', markersize=6)
        plt.step(frame_counts, [3 if x else None for x in C_array], marker='o', linestyle='None', color='blue', markersize=6)
        plt.step(frame_counts, [4 if x else None for x in D_array], marker='o', linestyle='None', color='green', markersize=6)
        plt.title('Frame-by-frame additional points diagram')
        plt.text(x=5, y=4.3, s=f'Total frames: {total_frames}\nA_counts: {sum(1 for x in A_array if x)}({A_ratio * 100:.2f}%)\nB_counts: {sum(1 for x in B_array if x)}({B_ratio * 100:.2f}%)\nC_counts: {sum(1 for x in C_array if x)}({C_ratio * 100:.2f}%)\nD_counts: {sum(1 for x in D_array if x)}({D_ratio * 100:.2f}%)', fontsize=12, bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round, pad=0.5'))
        plt.xlabel('Frame')
        plt.ylabel('Additional points')
        plt.xlim(0, len(frame_counts))
        plt.ylim(0, 5.7)
        plt.xticks(range(0, len(frame_counts), max(1, int(len(frame_counts) / 10))))
        plt.yticks(range(1, 5), ['A', 'B', 'C', 'D'])
        plt.grid(True)
        plt.savefig(os.path.join('pose_change', f'additional.png'))
        plt.close()

        total_score = max_score + extra_score
        worst_change_fine_pair = None
        if best_pair[0] is not None and best_pair[1] is not None:
            f_start, f_end = None, None
            for i in range(len(label_changes_coarse) - 1):
                if (label_changes_coarse[i], label_changes_coarse[i + 1]) == best_pair:
                    f_start = label_changes_fine[i]
                    f_end = label_changes_fine[i + 1]
                    break
            if f_start is None: f_start = coarse_to_rep_fine.get(best_pair[0])
            if f_end is None: f_end = coarse_to_rep_fine.get(best_pair[1])
            if f_start is not None and f_end is not None:
                worst_change_fine_pair = (pose_label_reverse.get(f_start), pose_label_reverse.get(f_end))

        # ==========================================
        # ⭐ 尋找整段影片的極值，直接放入 JSON 回傳！
        # ==========================================
        debug_min_knee = 180.0
        debug_min_hip = 180.0
        for data in joint_angles_per_frame:
            a = data["angles"]
            if a["left_knee"] is not None: debug_min_knee = min(debug_min_knee, a["left_knee"])
            if a["right_knee"] is not None: debug_min_knee = min(debug_min_knee, a["right_knee"])
            if a["left_hip"] is not None: debug_min_hip = min(debug_min_hip, a["left_hip"])
            if a["right_hip"] is not None: debug_min_hip = min(debug_min_hip, a["right_hip"])

        result = OrderedDict([
            ("totalFrame", total_frames),
            ("BodyPosturePoint", max_score),
            ("Addition Point", extra_score),
            ("twistAndLanternal", A_score),
            ("distance of body", B_score),
            ("arm raise", C_score),
            ("above shoulder", D_score),
            ("total_score", total_score),
            ("start", fine_start if fine_start is not None else 0),
            ("end", fine_end if fine_end is not None else 0),
            ("start_name", coarse_name.get(best_pair[0]) if best_pair[0] else None),
            ("end_name",   coarse_name.get(best_pair[1]) if best_pair[1] else None),
            ("pose_changes_coarse", [coarse_name.get(x) for x in label_changes_coarse] if label_changes_coarse else []),
            ("pose_changes_fine",   [pose_label_reverse.get(x) for x in label_changes_fine] if label_changes_fine else []),
            ("worst_change_fine",   worst_change_fine_pair),
            ("checkPersonInScreen", checkPersonInScreen),
            ("timeStamp", timestamp),
            # ⭐ 你的除錯好幫手：JSON 的最下方會多出這兩個數值！
            ("DEBUG_Lowest_Knee_Angle", round(debug_min_knee, 1)),
            ("DEBUG_Lowest_Hip_Angle", round(debug_min_hip, 1)),
        ])
        print(f"[{timestamp}] upload & process: success")
    except Exception as e:
        print(f'Error reasons:\n{traceback.print_exc()}')
        checkPersonInScreen = True
        result = OrderedDict([
            ("checkPersonInScreen", checkPersonInScreen),
            ("error", str(e))
        ])
        print(f"[{timestamp}] upload & process: fail")
    
    return result

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8080, debug=True)