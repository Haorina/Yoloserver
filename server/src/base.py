import numpy as np

# 明確從 calculator 引入，確保不會載入錯檔案
from src.calculator import (
    angles_to_lhc_label,
    get_angle_between_two_lines_position,
    get_angle_by_3_points,
    get_dist_between_points,
    get_triangle_gravity_position,
)

from src.value import (
    KPT_LIST,
    JOINT_NAME_DICT,
    JOINT_IDX_DICT,
    LineConnections,
    coco_line,  
)

class PoseResult:
    """通用 3D 姿勢物件結果 (適應 YOLO/MotionBERT 字典格式)"""
    def __init__(self, kpts_3d_dict: dict):
        self.kpts = kpts_3d_dict

    def get_kpt_pos_by_index(self, idx: int, get_3d: bool = True) -> list[float]:
        if 0 <= idx < len(KPT_LIST):
            kpt_name = KPT_LIST[idx]
            if kpt_name in self.kpts:
                return list(self.kpts[kpt_name])
        return None

    def get_kpt_pos_by_name(self, kpt_name: str, get_3d: bool = True) -> list[float]:
        if kpt_name in self.kpts:
            return list(self.kpts[kpt_name])
        return None

    def get_all_kpt_positions(self, get_3d: bool = True) -> list[list[float]]:
        positions = []
        for name in KPT_LIST:
            if name in self.kpts:
                positions.append(list(self.kpts[name]))
        return positions


class ResultAnalyzer:
    """姿勢結果分析器 (已針對 3D 空間門檻優化)"""
    def __init__(self, pose_result: PoseResult):
        self.pose_result = pose_result

    def get_center_position_by_2_hand(self, get_3d: bool = True) -> list[float]:
        left_hand = np.array(self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d))
        right_hand = np.array(self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d))
        return ((left_hand + right_hand) / 2.0).tolist()

    def get_body_gravity_position(self, get_3d: bool = True) -> list[float]:
        left_shoulder = self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)
        right_shoulder = self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)
        left_hip = self.pose_result.get_kpt_pos_by_name("left_hip", get_3d)
        right_hip = self.pose_result.get_kpt_pos_by_name("right_hip", get_3d)
        center_hip: list[float] = ((np.array(left_hip) + np.array(right_hip)) / 2.0).tolist()
        return get_triangle_gravity_position(left_shoulder, right_shoulder, center_hip)

    def get_joint_angle_by_name(self, center_joint_name: str, get_3d: bool = True) -> float:
        j1, j2 = JOINT_NAME_DICT[center_joint_name]
        pos1 = self.pose_result.get_kpt_pos_by_name(j1, get_3d)
        pos2 = self.pose_result.get_kpt_pos_by_name(j2, get_3d)
        center_pos = self.pose_result.get_kpt_pos_by_name(center_joint_name, get_3d)
        return get_angle_by_3_points(pos1, pos2, center_pos)

    def get_joint_angle_by_index(self, center_joint_idx: int, get_3d: bool = True) -> float:
        j1, j2 = JOINT_IDX_DICT[center_joint_idx]
        pos1 = self.pose_result.get_kpt_pos_by_index(j1, get_3d)
        pos2 = self.pose_result.get_kpt_pos_by_index(j2, get_3d)
        center_pos = self.pose_result.get_kpt_pos_by_index(center_joint_idx, get_3d)
        return get_angle_by_3_points(pos1, pos2, center_pos)

    def get_line_positions(self, line_type: LineConnections = coco_line, get_3d: bool = True) -> list[list, list, list]:
        left, center, right = [], [], []
        for pt1, pt2 in line_type.left_kpt:
            left.append([self.pose_result.get_kpt_pos_by_name(pt1, get_3d), self.pose_result.get_kpt_pos_by_name(pt2, get_3d)])
        for pt1, pt2 in line_type.center_kpt:
            center.append([self.pose_result.get_kpt_pos_by_name(pt1, get_3d), self.pose_result.get_kpt_pos_by_name(pt2, get_3d)])
        for pt1, pt2 in line_type.right_kpt:
            right.append([self.pose_result.get_kpt_pos_by_name(pt1, get_3d), self.pose_result.get_kpt_pos_by_name(pt2, get_3d)])
        return [left, center, right]

    def get_pose_shoulder_hip_staggered_angle_xz(self) -> float:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)
        return get_angle_between_two_lines_position(
            np.array([[shoulder_l[0], shoulder_l[2]], [shoulder_r[0], shoulder_r[2]]]),
            np.array([[hip_l[0], hip_l[2]], [hip_r[0], hip_r[2]]])
        )

    def get_pose_shoulder_hip_staggered_angle_xy(self) -> float:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)
        return get_angle_between_two_lines_position(
            np.array([[shoulder_l[0], shoulder_l[1]], [shoulder_r[0], shoulder_r[1]]]),
            np.array([[hip_l[0], hip_l[1]], [hip_r[0], hip_r[1]]])
        )

    def get_pose_shoulder_hip_staggered_angle_yz(self) -> float:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)
        return get_angle_between_two_lines_position(
            np.array([[shoulder_l[1], shoulder_l[2]], [shoulder_r[1], shoulder_r[2]]]),
            np.array([[hip_l[1], hip_l[2]], [hip_r[1], hip_r[2]]])
        )

    def get_a_hand_to_gravity_dist(self, is_left: bool, get_3d: bool = True) -> float:
        hand_kpt_name = "left_wrist" if is_left else "right_wrist"
        hand = np.array(self.pose_result.get_kpt_pos_by_name(hand_kpt_name, get_3d))
        gravity = np.array(self.get_body_gravity_position(get_3d))
        return get_dist_between_points(hand, gravity)

    def get_two_hands_center_to_gravity_dist(self, get_3d: bool = True) -> float:
        hand = np.array(self.get_center_position_by_2_hand(get_3d))
        gravity = np.array(self.get_body_gravity_position(get_3d))
        return get_dist_between_points(hand, gravity)

    def get_all_joint_angles_by_name(self, get_3d: bool = True) -> dict:
        all_angles = {}
        for center in JOINT_NAME_DICT.keys():
            all_angles[center] = self.get_joint_angle_by_name(center, get_3d)
        return all_angles

    def get_all_joint_angles_by_index(self, get_3d: bool = True) -> dict:
        all_angles = {}
        for center in JOINT_IDX_DICT.keys():
            all_angles[center] = self.get_joint_angle_by_index(center, get_3d)
        return all_angles

    # ==============================================================
    # ⭐ 核心修正區域：精準計算彎腰與照妖鏡機制
    # ==============================================================
    def get_lhc_label(self, get_3d: bool = True) -> str:
        angle_name_dict = self.get_all_joint_angles_by_name(get_3d)
        
        if get_3d:
            shoulder_l = np.array(self.pose_result.get_kpt_pos_by_name("left_shoulder", True))
            shoulder_r = np.array(self.pose_result.get_kpt_pos_by_name("right_shoulder", True))
            hip_l = np.array(self.pose_result.get_kpt_pos_by_name("left_hip", True))
            hip_r = np.array(self.pose_result.get_kpt_pos_by_name("right_hip", True))
            
            mid_shoulder = (shoulder_l + shoulder_r) / 2.0
            mid_hip = (hip_l + hip_r) / 2.0
            trunk_vec = mid_shoulder - mid_hip
            vertical_vec = np.array([0, -1, 0])
            
            norm_trunk = np.linalg.norm(trunk_vec)
            if norm_trunk > 0:
                cos_a = np.dot(trunk_vec, vertical_vec) / norm_trunk
                cos_a = np.clip(cos_a, -1.0, 1.0)
                trunk_bending_angle = np.degrees(np.arccos(cos_a)) 
                mapped_angle = 180.0 - trunk_bending_angle
                angle_name_dict["left_hip"] = mapped_angle
                angle_name_dict["right_hip"] = mapped_angle

        # ==========================================
        # 照妖鏡 (終端機列印)：請觀察伺服器執行時印出的這行字
        # ==========================================
        min_knee_val = min(angle_name_dict.get('left_knee', 180), angle_name_dict.get('right_knee', 180))
        min_hip_val = angle_name_dict.get('left_hip', 180)
        # 將原本的 print 替換成這行：
        print(f"[角度除錯] 目前影格 -> 膝蓋: {min_knee_val:.1f} 度 | 軀幹挺直度: {min_hip_val:.1f} 度", flush=True)

        return angles_to_lhc_label(angle_name_dict)

    def check_if_trunk_is_twisted_or_lateral_inclination(self, get_3d: bool = True) -> bool:
        return self.check_if_trunk_is_twisted(get_3d) or self.check_if_trunk_is_lateral_inclination(get_3d)

    def check_if_trunk_is_twisted(self, get_3d: bool = True) -> bool:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        return abs(shoulder_l[2] - shoulder_r[2]) > 0.1  

    def check_if_trunk_is_lateral_inclination(self, get_3d: bool = True) -> bool:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        return abs(shoulder_l[1] - shoulder_r[1]) > 0.08

    def check_if_hands_at_a_distance(self, get_3d: bool = True) -> bool:
        left_hand_to_gravity = self.get_a_hand_to_gravity_dist(is_left=True, get_3d=True)
        right_hand_to_gravity = self.get_a_hand_to_gravity_dist(is_left=False, get_3d=True)
        return left_hand_to_gravity > 0.25 or right_hand_to_gravity > 0.25

    def check_if_arms_raised(self, get_3d: bool = True) -> bool:
        left_shoulder_angle = self.get_joint_angle_by_name("left_shoulder", get_3d)
        left_elbow_y = self.pose_result.get_kpt_pos_by_name("left_elbow", get_3d)[1] * -1
        left_wrist_y = self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)[1] * -1
        left_result = (left_shoulder_angle > 60) and (left_wrist_y > left_elbow_y)

        right_shoulder_angle = self.get_joint_angle_by_name("right_shoulder", get_3d)
        right_elbow_y = self.pose_result.get_kpt_pos_by_name("right_elbow", get_3d)[1] * -1
        right_wrist_y = self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)[1] * -1
        right_result = (right_shoulder_angle > 60) and (right_wrist_y > right_elbow_y)

        return left_result or right_result

    def check_if_hands_above_shoulder(self, get_3d: bool = True) -> bool:
        left_wrist_y = self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)[1] * -1
        left_shoulder_y = self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)[1] * -1
        right_wrist_y = self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)[1] * -1
        right_shoulder_y = self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)[1] * -1
        return left_wrist_y > left_shoulder_y or right_wrist_y > right_shoulder_y