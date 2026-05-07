import numpy as np

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
    def __init__(self, pose_result: PoseResult):
        self.pose_result = pose_result

    def get_center_position_by_2_hand(self, get_3d: bool = True) -> list[float]:
        left_hand = np.array(self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d))
        right_hand = np.array(self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d))
        hand_center = (left_hand + right_hand) / 2
        return hand_center.tolist()

    def get_body_gravity_position(self, get_3d: bool = True) -> list[float]:
        left_shoulder = self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)
        right_shoulder = self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)
        left_hip = self.pose_result.get_kpt_pos_by_name("left_hip", get_3d)
        right_hip = self.pose_result.get_kpt_pos_by_name("right_hip", get_3d)

        center_hip: list[float] = ((np.array(left_hip) + np.array(right_hip)) / 2).tolist()
        gravity_position = get_triangle_gravity_position(left_shoulder, right_shoulder, center_hip)
        return gravity_position

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
            pos = [self.pose_result.get_kpt_pos_by_name(pt1, get_3d), self.pose_result.get_kpt_pos_by_name(pt2, get_3d)]
            left.append(pos)

        for pt1, pt2 in line_type.center_kpt:
            pos = [self.pose_result.get_kpt_pos_by_name(pt1, get_3d), self.pose_result.get_kpt_pos_by_name(pt2, get_3d)]
            center.append(pos)

        for pt1, pt2 in line_type.right_kpt:
            pos = [self.pose_result.get_kpt_pos_by_name(pt1, get_3d), self.pose_result.get_kpt_pos_by_name(pt2, get_3d)]
            right.append(pos)

        return [left, center, right]

    def get_pose_shoulder_hip_staggered_angle_xz(self) -> float:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array([[shoulder_l[0], shoulder_l[2]], [shoulder_r[0], shoulder_r[2]]])
        hip_vec = np.array([[hip_l[0], hip_l[2]], [hip_r[0], hip_r[2]]])
        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

    def get_pose_shoulder_hip_staggered_angle_xy(self) -> float:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array([[shoulder_l[0], shoulder_l[1]], [shoulder_r[0], shoulder_r[1]]])
        hip_vec = np.array([[hip_l[0], hip_l[1]], [hip_r[0], hip_r[1]]])
        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

    def get_pose_shoulder_hip_staggered_angle_yz(self) -> float:
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array([[shoulder_l[1], shoulder_l[2]], [shoulder_r[1], shoulder_r[2]]])
        hip_vec = np.array([[hip_l[1], hip_l[2]], [hip_r[1], hip_r[2]]])
        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

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
            angle = self.get_joint_angle_by_name(center, get_3d)
            all_angles.update({center: angle})
        return all_angles

    def get_all_joint_angles_by_index(self, get_3d: bool = True) -> dict:
        all_angles = {}
        for center in JOINT_IDX_DICT.keys():
            angle = self.get_joint_angle_by_index(center, get_3d)
            all_angles.update({center: angle})
        return all_angles

    def get_lhc_label(self, get_3d: bool = True) -> str:
        angle_name_dict = self.get_all_joint_angles_by_name(get_3d)
        return angles_to_lhc_label(angle_name_dict)

    def check_if_trunk_is_twisted_or_lateral_inclination(self, get_3d: bool = True) -> bool:
        return self.check_if_trunk_is_twisted(get_3d) or self.check_if_trunk_is_lateral_inclination(get_3d)

    def check_if_trunk_is_twisted(self, get_3d: bool = True) -> bool:
        TWISTED_ANGLE = 15
        return self.get_pose_shoulder_hip_staggered_angle_xz() > TWISTED_ANGLE

    def check_if_trunk_is_lateral_inclination(self, get_3d: bool = True) -> bool:
        HEIGHT_DIFF = 0.06
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        if shoulder_l is None or shoulder_r is None: return False
        shoulder_diff = np.linalg.norm(shoulder_l[1] - shoulder_r[1])
        return shoulder_diff > HEIGHT_DIFF

    def check_if_hands_at_a_distance(self, get_3d: bool = True) -> bool:
        DIST = 0.4
        wrist_l = self.pose_result.get_kpt_pos_by_name("left_wrist", True)
        wrist_r = self.pose_result.get_kpt_pos_by_name("right_wrist", True)
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)
        
        if any(x is None for x in [wrist_l, wrist_r, hip_l, hip_r]): return False

        gravity = np.mean([hip_l[0::2], hip_r[0::2]], axis=0)
        left_hand_to_gravity = np.linalg.norm(np.array(wrist_l[0::2]) - gravity)
        right_hand_to_gravity = np.linalg.norm(np.array(wrist_r[0::2]) - gravity)

        return left_hand_to_gravity > DIST or right_hand_to_gravity > DIST

    def check_if_arms_raised(self, get_3d: bool = True) -> bool:
        RAISED_ANGLE = 60
        left_shoulder_angle = self.get_joint_angle_by_name("left_shoulder", get_3d)
        
        ls = self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)
        le = self.pose_result.get_kpt_pos_by_name("left_elbow", get_3d)
        lw = self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)
        
        left_result = False
        if ls and le and lw and left_shoulder_angle > RAISED_ANGLE:
            if (ls[1] * -1) > (lw[1] * -1) > (le[1] * -1):
                left_result = True

        right_shoulder_angle = self.get_joint_angle_by_name("right_shoulder", get_3d)
        rs = self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)
        re = self.pose_result.get_kpt_pos_by_name("right_elbow", get_3d)
        rw = self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)

        right_result = False
        if rs and re and rw and right_shoulder_angle > RAISED_ANGLE:
            if (rs[1] * -1) > (rw[1] * -1) > (re[1] * -1):
                right_result = True

        return left_result or right_result

    def check_if_hands_above_shoulder(self, get_3d: bool = True) -> bool:
        lw = self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)
        ls = self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)
        rw = self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)
        rs = self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)
        
        if any(x is None for x in [lw, ls, rw, rs]): return False
        
        return (lw[1] * -1) > (ls[1] * -1) or (rw[1] * -1) > (rs[1] * -1)
