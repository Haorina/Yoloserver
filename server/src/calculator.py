import numpy as np
import math

# ==========================================
# 3D 空間向量運算 (防呆與高精度版)
# ==========================================
def get_angle_by_3_points(pos1: list, pos2: list, center_pos: list) -> float:
    if pos1 is None or pos2 is None or center_pos is None:
        return 0.0
    p1, p2, c = np.array(pos1), np.array(pos2), np.array(center_pos)
    vec1, vec2 = p1 - c, p2 - c
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(vec1, vec2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def get_angle_between_two_lines_position(line1, line2) -> float:
    pos1a, pos1b = np.array(line1[0]), np.array(line1[1])
    pos2a, pos2b = np.array(line2[0]), np.array(line2[1])
    vec1, vec2 = pos1a - pos1b, pos2a - pos2b
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(vec1, vec2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def get_dist_between_points(*point_list) -> float:
    dist = 0.0
    if len(point_list) > 1:
        pt1 = np.array(point_list[0])
        for pt2 in point_list[1:]:
            dist += np.linalg.norm(pt1 - np.array(pt2))
            pt1 = np.array(pt2)
    return float(dist)

def get_triangle_gravity_position(pos1, pos2, pos3) -> list:
    return ((np.array(pos1) + np.array(pos2) + np.array(pos3)) / 3.0).tolist()

# ==========================================
# LHC 姿勢判定邏輯 (細分項版本)
# ==========================================
def angles_to_lhc_label(angles: dict) -> str:
    """
    將角度轉換成 LHC 姿勢細分標籤
    """
    if len(angles) <= 0:
        return ""
    
    # 自動抓取左右兩側最嚴重的角度來進行評分
    min_knee = min(angles.get("left_knee", 180), angles.get("right_knee", 180))
    min_hip = min(angles.get("left_hip", 180), angles.get("right_hip", 180))
    max_shoulder = max(angles.get("left_shoulder", 0), angles.get("right_shoulder", 0))

    # A5: 蹲姿、跪姿、跪坐姿勢
    if min_knee <= 30: 
        return "A5-3"
    elif min_knee <= 45: 
        return "A5-2"
    elif min_knee <= 130:  # 稍微放寬一般蹲姿的門檻，避免 3D 空間漏判
        return "A5-1"

    # A4: 嚴重彎腰
    elif min_hip <= 100: 
        return "A4-2"
    elif min_hip <= 120: 
        return "A4-1"

    # A3: 輕微彎腰
    elif min_hip <= 140: 
        return "A3-2"
    elif min_hip <= 160: 
        return "A3-1"

    # A2: 手臂抬高
    elif max_shoulder > 90: 
        return "A2"

    # A1: 正常站立
    else: 
        return "A1"

def angles_to_lhc_label_by_one_side(angles: dict, is_left=False) -> str:
    """
    單側角度轉換成 LHC 姿勢細分標籤
    """
    if len(angles) <= 0: return ""
    
    knee = angles.get("left_knee", 180) if is_left else angles.get("right_knee", 180)
    hip = angles.get("left_hip", 180) if is_left else angles.get("right_hip", 180)
    shoulder = angles.get("left_shoulder", 0) if is_left else angles.get("right_shoulder", 0)

    if knee <= 30: return "A5-3"
    elif knee <= 45: return "A5-2"
    elif knee <= 130: return "A5-1"

    elif hip <= 100: return "A4-2"
    elif hip <= 120: return "A4-1"

    elif hip <= 140: return "A3-2"
    elif hip <= 160: return "A3-1"

    elif shoulder > 90: return "A2"
    
    else: return "A1"

# ==========================================
# 標籤陣列處理邏輯
# ==========================================
def label_list_to_label_list_rule(labels: list) -> list:
    result = []
    buffer = None  
    num = 1
    for lab in labels:
        if buffer is None: buffer = lab
        elif buffer == lab: num += 1
        else:
            result.append([buffer, num]) 
            buffer = lab
            num = 1
    if buffer is not None:
        result.append([buffer, num]) 
    return result

def label_list_rule_to_label_list(label_list_rule: list) -> list:
    result = []
    for label, num in label_list_rule:
        result += [label] * int(num)
    return result