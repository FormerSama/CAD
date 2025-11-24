#!/usr/bin/env python3
"""
labelme2unet_masks.py
將 Labelme JSON 批次轉換成 U-Net 的 mask：
  - 灰階版（訓練用，資料夾：masks/）
  - 彩色版（可選，用於報告顯示，資料夾：masks_color/）

新增功能：
  --skip_empty   跳過 shape 為空的 JSON
  --with_color   若啟用，才輸出彩色 mask
"""

import os
import json
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw
from collections import OrderedDict

def shape_to_polygon(shape):
    st = shape.get("shape_type", "").lower()
    pts = shape.get("points", [])

    if st == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    elif st in ["polygon", "polyline", "linestrip", "line"]:
        return pts

    elif st == "circle" and len(pts) >= 2:
        cx, cy = pts[0]
        rx, ry = pts[1]
        r = ((cx - rx)**2 + (cy - ry)**2)**0.5
        num = 40
        return [[cx + r*np.cos(2*np.pi*i/num), cy + r*np.sin(2*np.pi*i/num)] for i in range(num)]

    return pts

def ensure_folder(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def random_color():
    return tuple(random.randint(50, 255) for _ in range(3))

def main(args):
    json_dir = args.json_dir
    out_dir = args.out_dir
    resize = args.resize
    copy_images = args.copy_images
    skip_empty = args.skip_empty
    with_color = args.with_color

    # === 輸出資料夾結構 ===
    masks_dir = os.path.join(out_dir, "masks")
    ensure_folder(out_dir)
    ensure_folder(masks_dir)

    if with_color:
        masks_color_dir = os.path.join(out_dir, "masks_color")
        ensure_folder(masks_color_dir)

    if copy_images:
        images_out = os.path.join(out_dir, "images")
        ensure_folder(images_out)

    # === 建立 label map ===
    label_map = OrderedDict({"__background__": 0})
    json_files = [f for f in sorted(os.listdir(json_dir)) if f.lower().endswith(".json")]

    if not json_files:
        print("❌ No JSON found.")
        return

    for jf in json_files:
        with open(os.path.join(json_dir, jf), "r", encoding="utf-8") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label and label not in label_map:
                label_map[label] = len(label_map)

    num_classes = len(label_map)
    print(f"🔍 Found classes = {num_classes}: {list(label_map.keys())}")

    # === 顏色表 ===
    color_map = {k: random_color() for k in label_map.keys()}
    color_map["__background__"] = (0, 0, 0)

    # === 主流程 ===
    for jf in json_files:
        jp = os.path.join(json_dir, jf)
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])

        # --- 判斷是否空 JSON ---
        def has_points(s):
            pts = s.get("points", [])
            return isinstance(pts, list) and len(pts) >= 2

        non_empty_shapes = [s for s in shapes if has_points(s)]

        if skip_empty and len(non_empty_shapes) == 0:
            print(f"⏭️ Skip empty JSON: {jf}")
            continue

        # --- 找圖片 ---
        img_path = data.get("imagePath", "")
        img_file = os.path.join(json_dir, img_path) if img_path else None

        if not img_file or not os.path.exists(img_file):
            base = os.path.splitext(jf)[0]
            for ext in [".jpg", ".jpeg", ".png", ".tif"]:
                cand = os.path.join(json_dir, base + ext)
                if os.path.exists(cand):
                    img_file = cand
                    break

        if not img_file:
            print(f"[WARN] Image not found for {jf}")
            continue

        img = Image.open(img_file)
        ow, oh = img.size
        tw, th = resize if resize else (ow, oh)

        # === 建立灰階 mask（訓練用） ===
        mask_gray = Image.new("L", (tw, th), 0)
        draw_gray = ImageDraw.Draw(mask_gray)

        # === 彩色 mask（可選） ===
        if with_color:
            mask_color = Image.new("RGB", (tw, th), (0, 0, 0))
            draw_color = ImageDraw.Draw(mask_color)

        # === 繪製 ===
        for shape in non_empty_shapes:
            label = shape.get("label", "").strip()
            if label not in label_map:
                continue

            pts = shape_to_polygon(shape)
            if not pts:
                continue

            if resize:
                sx, sy = tw/ow, th/oh
                pts = [(p[0]*sx, p[1]*sy) for p in pts]

            cid = label_map[label]
            gray_val = round(255 * cid / (num_classes-1)) if num_classes > 1 else 0

            draw_gray.polygon(pts, outline=gray_val, fill=gray_val)

            if with_color:
                draw_color.polygon(pts, outline=color_map[label], fill=color_map[label])

        # === 輸出 ===
        base = os.path.splitext(os.path.basename(img_file))[0]
        mask_gray.save(os.path.join(masks_dir, base + ".png"))

        if with_color:
            mask_color.save(os.path.join(masks_color_dir, base + ".png"))

        if copy_images:
            img_resize = img.resize((tw, th), Image.BILINEAR) if resize else img
            img_resize.save(os.path.join(images_out, os.path.basename(img_file)))

        print(f"✅ {jf} done.")

    print("\n🎉 完成輸出")
    print(f"灰階訓練 mask： {masks_dir}")
    if with_color:
        print(f"彩色版： {masks_color_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--with_color", action="store_true", help="輸出彩色版 mask")
    parser.add_argument("--resize", nargs=2, type=int)
    args = parser.parse_args()

    if args.resize:
        args.resize = (int(args.resize[0]), int(args.resize[1]))

    main(args)
