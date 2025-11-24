#!/usr/bin/env python3
"""
mask2labelme.py
將 U-Net 輸出的 mask (單通道灰階) 轉換為 Labelme 可用的 JSON 標註格式。
每個灰階值代表一個類別。
"""
import os, cv2, json
import numpy as np
from PIL import Image
from datetime import datetime

from utils import preprocess_mask

def load_label_names(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names

def mask_to_shapes(mask, label_names):
    """將 mask 圖像轉成 Labelme 的 shapes"""
    shapes = []
    h, w = mask.shape[:2]
    num_classes = len(label_names)

    for class_id in range(1, num_classes):  # 跳過背景 (0)
        label = label_names[class_id]
        binary = (mask == class_id).astype(np.uint8)
        if np.sum(binary) == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = cnt.squeeze(1).astype(float).tolist()
            shape = {
                "label": label,
                "points": pts,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
    return shapes

def convert_mask_to_labelme(mask_path, image_dir, out_dir, label_names):
    base = os.path.splitext(os.path.basename(mask_path))[0]
    img_path = os.path.join(image_dir, base + ".jpg")
    if not os.path.exists(img_path):
        for ext in [".png", ".jpeg"]:
            p = os.path.join(image_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
    if not os.path.exists(img_path):
        print(f"[WARN] 找不到對應圖片：{base}")
        return

    img = Image.open(img_path)
    w, h = img.size
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = preprocess_mask(mask, straighten=True, simplify=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    # normalize if mask is gray-scaled from 0~255
    unique_vals = np.unique(mask)
    if np.max(unique_vals) > len(label_names)-1:
        # 灰階轉為離散 id
        scaled = np.zeros_like(mask)
        sorted_vals = sorted([v for v in unique_vals if v > 0])
        for i, v in enumerate(sorted_vals, start=1):
            scaled[mask == v] = i
        mask = scaled

    shapes = mask_to_shapes(mask, label_names)

    data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "time": datetime.now().isoformat()
    }

    out_path = os.path.join(out_dir, base + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {base}.png → {base}.json ({len(shapes)} shapes)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", required=True, help="U-Net 輸出的 mask 資料夾 (灰階或 class id)")
    parser.add_argument("--image_dir", required=True, help="對應原始圖片的資料夾")
    parser.add_argument("--label_names", required=True, help="label_names.txt 檔路徑")
    parser.add_argument("--out_dir", required=True, help="輸出 JSON 資料夾")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    label_names = load_label_names(args.label_names)
    mask_files = [f for f in os.listdir(args.mask_dir) if f.lower().endswith((".png", ".jpg"))]

    for f in mask_files:
        mask_path = os.path.join(args.mask_dir, f)
        convert_mask_to_labelme(mask_path, args.image_dir, args.out_dir, label_names)

if __name__ == "__main__":
    main()
