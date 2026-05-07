import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_DIR = f"./addition/test" 


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ 資料 ------------------
labels_dict = {
    "V1": [5,5,5,5,5,5,5,5,5,5,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    "V2": [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "V3": [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    "V4": [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    "V5": [5,5,5,5,5,5,5,4,4,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    "V6": [5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    "V7": [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    "V8": [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4],
    "V9": [5,5,5,5,5,5,5,5,5,4,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "V10": [4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    
}
filmName = f"V{input('Enter film number (1-10): ')}"
# labels_original = [5,5,5,5,5,5,5,5,5,5,3,3,3,3,4,4,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # V1
# labels_original = [5,5,5,5,5,5,5,4,4,3,3,3,3,3,3,3,3,3,3,3,1,1,1,3,3,1,1,1,1,1,2,2,3,3,2,3,3,2,2,2,2,2,2,3,2,3,3,3,2,2,2,2,2,2,2,2,2,3,3,3,2,2,2] # V5
labels_original = labels_dict[filmName]

frame_counts = labels_original
total_frames = len(frame_counts)
print("Total frames:", total_frames)

pose_scores = {
    (1, 1): 0, (1, 2): 3, (1, 3): 3, (1, 4): 7, (1, 5): 9,
    (2, 2): 5, (2, 3): 5, (2, 4): 10, (2, 5): 13,
    (3, 3): 5, (3, 4): 10, (3, 5): 13,
    (4, 4): 15, (4, 5): 18,
    (5, 5): 20
}
# ------------------ 畫原圖 ------------------
x = np.arange(len(labels_original))
plt.figure(figsize=(8, 4.5))
plt.plot(x, labels_original, marker='o', linewidth=1)
plt.xlabel('frame')
plt.ylabel('pose', rotation='vertical')
plt.grid(True)
step = 30
plt.xticks(x[::step], rotation=45)
plt.yticks([1, 2, 3, 4, 5], ['A1', 'A2', 'A3', 'A4', 'A5'])
plt.ylim(0, 6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{filmName}_original.png"))
plt.close()

# ============================================================
# 第一階段：段落切割 + barrier 區間提升 keep（解決「短段被拆開但總長其實達標」）
# 規則：
# 1) 第一段、最後段必保留
# 2) 中間段落 < con 預設要刪
# 3) 若同 label 在「兩個 barrier 之間」的總長度 >= con，則該 label 在該區間的所有段落提升為保留
# ============================================================
con = 12
csv = 90

labels = labels_original.copy()
removed_points = []
label_changes = []
valid_frames = len(labels)

# 1) 切成連續段落
segments = []  # dict: {start,end,label,len}
start = 0
for i in range(1, len(labels) + 1):
    if i == len(labels) or labels[i] != labels[start]:
        end = i - 1
        segments.append({
            "start": start,
            "end": end,
            "label": labels[start],
            "len": i - start
        })
        start = i

nseg = len(segments)

# 2) keep 初始：第一段、最後段必保留；len>=con 也必保留（barrier）
keep = set()
for idx, seg in enumerate(segments):
    if idx == 0 or idx == nseg - 1 or seg["len"] >= con:
        keep.add(idx)

# 3) 迭代提升 keep：在 barrier 區間內，如果某 label 的總長 >= con，則該 label 在區間內全部保留
changed = True
while changed:
    changed = False
    barriers = sorted(keep)
    if len(barriers) < 2:
        break

    for bi in range(len(barriers) - 1):
        L = barriers[bi]
        R = barriers[bi + 1]

        idxs = [k for k in range(L + 1, R) if k not in keep]
        if not idxs:
            continue

        totals = {}
        by_label = {}
        for k in idxs:
            lab = segments[k]["label"]
            totals[lab] = totals.get(lab, 0) + segments[k]["len"]
            by_label.setdefault(lab, []).append(k)

        for lab, total_len in totals.items():
            if total_len >= con:
                for k in by_label[lab]:
                    if k not in keep:
                        keep.add(k)
                        changed = True

# 4) 最終要刪的段落：非 keep 且 len < con
remove = set()
for idx, seg in enumerate(segments):
    if idx not in keep and seg["len"] < con:
        remove.add(idx)

# 5) 套用刪除：段落整段設 None
for idx in remove:
    s, e = segments[idx]["start"], segments[idx]["end"]
    seg_len = e - s + 1
    valid_frames -= seg_len
    for j in range(s, e + 1):
        labels[j] = None
        removed_points.append((j, labels_original[j]))

# ============================================================
# 第二階段：迭代式消抖（大量跳動處理）
# 做法：多輪掃描「短段落」，把短段落回填成更穩定的鄰居
# - 左右同 label：直接回填為該 label（等效合併）
# - 左右不同 label：選較長的那側（你也可以改成用 pose_scores 選成本較小者）
# ============================================================
max_iters = con         # 依你的想法：最多迭代到 con 輪
min_len = 6            # 要消掉的「短段」門檻：2~4 都常見（越大越狠）
protect_ends = True     # 保護第一段/最後段不動（延續規則1）

def build_segments_skip_none(arr):
    segs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and arr[j + 1] == arr[i]:
            j += 1
        segs.append({"start": i, "end": j, "label": arr[i], "len": j - i + 1})
        i = j + 1
    return segs

for _ in range(max_iters):
    segs = build_segments_skip_none(labels)
    if len(segs) <= 2:
        break

    changed = False

    for k, seg in enumerate(segs):
        if protect_ends and (k == 0 or k == len(segs) - 1):
            continue

        if seg["len"] >= min_len:
            continue

        left = segs[k - 1]
        right = segs[k + 1]

        # 只有「左右同 label」才判定為尖峰抖動，直接刪除
        if left["label"] == right["label"]:
            for i in range(seg["start"], seg["end"] + 1):
                if labels[i] is not None:
                    removed_points.append((i, labels_original[i]))
                    labels[i] = None
                    valid_frames -= 1
            changed = True

    if not changed:
        break

# ------------------ 清理 None 並保持連線 ------------------
filtered_x = [i for i, v in enumerate(labels) if v is not None]
filtered_y = [v for v in labels if v is not None]

# ------------------ 產生 label_changes（以消抖後的結果重算段落）------------------
label_changes = []
merged = []
prev = None
run_len = 0
for v in labels:
    if v is None:
        continue
    if v == prev:
        run_len += 1
    else:
        if prev is not None:
            merged.append((prev, run_len))
        prev = v
        run_len = 1
if prev is not None:
    merged.append((prev, run_len))

for lab, ln in merged:
    if ln >= con:
        label_changes.append(lab)
    if ln >= csv:
        label_changes.append(lab)

if merged:
    if len(label_changes) == 0 or label_changes[-1] != merged[-1][0]:
        label_changes.append(merged[-1][0])

# ------------------ 繪製過濾後圖 ------------------
plt.figure(figsize=(8, 4.5))
plt.plot(filtered_x, filtered_y, marker='o', linewidth=1)
plt.xlabel('frame')
plt.ylabel('pose', rotation='vertical')
plt.grid(True)
plt.xticks(x[::step], rotation=45)
plt.yticks([1, 2, 3, 4, 5], ['A1', 'A2', 'A3', 'A4', 'A5'])
plt.ylim(0, 6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{filmName}_filtered.png"))
plt.close()

print("valid_frames:", valid_frames)
print("label_changes:", label_changes)
print("removed_points_count:", len(removed_points))
