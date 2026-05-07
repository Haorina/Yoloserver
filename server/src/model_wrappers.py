import torch
import numpy as np
from ultralytics import YOLO
from lib.model.DSTformer import DSTformer

class YOLOPoseEstimator:
    # ⭐ 這裡把 yolov8n-pose.pt 改成 yolo11n-pose.pt
    def __init__(self, model_path='yolo11n-pose.pt', device='cpu'):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.device = device
        self.last_kpts = None

    def predict(self, frame):
        results = self.model(frame, verbose=False, device=self.device)
        annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()
        
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts_2d = results[0].keypoints.xy.cpu().numpy()[0]
            if np.sum(kpts_2d) > 0:
                self.last_kpts = kpts_2d
                return kpts_2d, annotated_frame
        
        if self.last_kpts is not None:
            return self.last_kpts, annotated_frame
            
        return np.zeros((17, 2)), annotated_frame

class MotionBERTEstimator:
    def __init__(self, model_path, device='cpu'):
        print(f"Loading MotionBERT model from {model_path}...")
        self.device = device
        self.model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, 
                               depth=5, num_heads=8, mlp_ratio=2, 
                               maxlen=243, num_joints=17)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, seq_2d_coco: np.ndarray, width: int, height: int) -> np.ndarray:
        if self.model is None: raise NotImplementedError("模型尚未載入")

        h36m_2d = np.zeros((seq_2d_coco.shape[0], 17, 2))
        h36m_2d[:, 0] = (seq_2d_coco[:, 11] + seq_2d_coco[:, 12]) / 2 
        h36m_2d[:, 1] = seq_2d_coco[:, 12] 
        h36m_2d[:, 2] = seq_2d_coco[:, 14] 
        h36m_2d[:, 3] = seq_2d_coco[:, 16] 
        h36m_2d[:, 4] = seq_2d_coco[:, 11] 
        h36m_2d[:, 5] = seq_2d_coco[:, 13] 
        h36m_2d[:, 6] = seq_2d_coco[:, 15] 
        h36m_2d[:, 8] = (seq_2d_coco[:, 5] + seq_2d_coco[:, 6]) / 2   
        h36m_2d[:, 7] = (h36m_2d[:, 0] + h36m_2d[:, 8]) / 2           
        h36m_2d[:, 9] = seq_2d_coco[:, 0]  
        eye_center = (seq_2d_coco[:, 1] + seq_2d_coco[:, 2]) / 2
        h36m_2d[:, 10] = h36m_2d[:, 9] - (h36m_2d[:, 8] - eye_center) 
        h36m_2d[:, 11] = seq_2d_coco[:, 5] 
        h36m_2d[:, 12] = seq_2d_coco[:, 7] 
        h36m_2d[:, 13] = seq_2d_coco[:, 9] 
        h36m_2d[:, 14] = seq_2d_coco[:, 6] 
        h36m_2d[:, 15] = seq_2d_coco[:, 8] 
        h36m_2d[:, 16] = seq_2d_coco[:, 10]

        scale = max(width, height)
        h36m_2d[..., 0] = (h36m_2d[..., 0] / scale) * 2 - (width / scale)
        h36m_2d[..., 1] = (h36m_2d[..., 1] / scale) * 2 - (height / scale)

        conf_scores = np.ones((h36m_2d.shape[0], 17, 1))
        h36m_3dim_input = np.concatenate((h36m_2d, conf_scores), axis=-1)

        maxlen = 243
        total_frames = h36m_3dim_input.shape[0]
        predicted_3d_list = []

        with torch.no_grad():
            for start_idx in range(0, total_frames, maxlen):
                end_idx = min(start_idx + maxlen, total_frames)
                chunk = h36m_3dim_input[start_idx:end_idx]
                seq_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
                predicted_chunk = self.model(seq_tensor)
                predicted_3d_list.append(predicted_chunk.squeeze(0).cpu().numpy())

        seq_3d_h36m = np.concatenate(predicted_3d_list, axis=0)

        coco_3d = np.zeros((seq_3d_h36m.shape[0], 17, 3))
        coco_3d[:, 0] = seq_3d_h36m[:, 9]  
        coco_3d[:, 1:5] = seq_3d_h36m[:, 9:10] 
        coco_3d[:, 5] = seq_3d_h36m[:, 11] 
        coco_3d[:, 6] = seq_3d_h36m[:, 14] 
        coco_3d[:, 7] = seq_3d_h36m[:, 12] 
        coco_3d[:, 8] = seq_3d_h36m[:, 15] 
        coco_3d[:, 9] = seq_3d_h36m[:, 13] 
        coco_3d[:, 10]= seq_3d_h36m[:, 16] 
        coco_3d[:, 11]= seq_3d_h36m[:, 4]  
        coco_3d[:, 12]= seq_3d_h36m[:, 1]  
        coco_3d[:, 13]= seq_3d_h36m[:, 5]  
        coco_3d[:, 14]= seq_3d_h36m[:, 2]  
        coco_3d[:, 15]= seq_3d_h36m[:, 6]  
        coco_3d[:, 16]= seq_3d_h36m[:, 3]  

        return coco_3d