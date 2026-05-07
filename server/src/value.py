import numpy as np

# 新增 COCO 17 關鍵點列表 (取代原有的 33 點)
KPT_LIST: list = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
]

# 關鍵點索引值字典
KPT_IDX_DICT: dict = {idx: name for idx, name in enumerate(KPT_LIST)}

# 關節角度名稱對照字典 (KIM-LHC 角度計算需要)
JOINT_NAME_DICT: dict = {
    "left_shoulder": ["left_elbow", "left_hip"],
    "right_shoulder": ["right_elbow", "right_hip"],
    "left_hip": ["left_shoulder", "left_knee"],
    "right_hip": ["right_shoulder", "right_knee"],
    "left_knee": ["left_hip", "left_ankle"],
    "right_knee": ["right_hip", "right_ankle"],
}

# 關節角度索引值對照字典
JOINT_IDX_DICT: dict = {
    5: [7, 11],   # 左肩: 左肘, 左髖
    6: [8, 12],   # 右肩: 右肘, 右髖
    11: [5, 13],  # 左髖: 左肩, 左膝
    12: [6, 14],  # 右髖: 右肩, 右膝
    13: [11, 15], # 左膝: 左髖, 左踝
    14: [12, 16], # 右膝: 右髖, 右踝
}


# 線條連接點類別
class LineConnections:
    """線條連接點"""

    left_kpt: list[list[str]] = []  # 左邊關鍵點連接列表
    right_kpt: list[list[str]] = []  # 右邊關鍵點連接列表
    center_kpt: list[list[str]] = []  # 中間關鍵點連接列表
    full_kpt: list[list[str]] = []  # 全關鍵點連接列表

    def __init__(
        self, left: list[list[str]], right: list[list[str]], center: list[list[str]]
    ) -> None:
        """初始化物件
        Args:
            left (list[list[str]]): 左邊連接列表
            right (list[list[str]]): 右邊連接列表
            center (list[list[str]]): 中間連接列表
        """
        
        self.left_kpt = left
        self.right_kpt = right
        self.center_kpt = center
        
        # 避免空陣列時發生維度不匹配的問題
        if len(left) > 0 and len(right) > 0:
            lr_kpt = np.append(left, right, axis=0)
            if len(center) > 0:
                full_kpt = np.append(lr_kpt, center, axis=0)
            else:
                full_kpt = lr_kpt
        else:
            full_kpt = np.array([])
            
        self.full_kpt = full_kpt.tolist()


# 取代原本 blazepose_line 的 COCO 線條組合列表
coco_line = LineConnections(
    left=[
        # 頭部
        ["nose", "left_eye"], 
        ["left_eye", "left_ear"],
        # 手部
        ["left_shoulder", "left_elbow"], 
        ["left_elbow", "left_wrist"],
        # 腰部、腳
        ["left_shoulder", "left_hip"], 
        ["left_hip", "left_knee"], 
        ["left_knee", "left_ankle"]
    ],
    right=[
        # 頭部
        ["nose", "right_eye"], 
        ["right_eye", "right_ear"],
        # 手部
        ["right_shoulder", "right_elbow"], 
        ["right_elbow", "right_wrist"],
        # 腰部、腳
        ["right_shoulder", "right_hip"], 
        ["right_hip", "right_knee"], 
        ["right_knee", "right_ankle"]
    ],
    center=[
        # 身體中線連接
        ["left_shoulder", "right_shoulder"], 
        ["left_hip", "right_hip"]
    ]
)