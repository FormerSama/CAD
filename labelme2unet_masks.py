#!/usr/bin/env python3
"""
labelme2unet_masks.py
將 Labelme JSON 批次轉成 U-Net 的 mask (single-channel PNG).
輸出形式:
  output_dir/
    images/   <- 可選，複製原圖 (若指定)
    masks/    <- mask png 單通道，值為 class id (0 背景)
    label_names.txt

用法範例:
  python labelme2unet_masks.py --json_dir ./labelme_jsons --out_dir ./dataset --copy_images --resize 960 960
"""
import os
import json
import argparse
from PIL import Image, ImageDraw
import numpy as np
import cv2
from collections import OrderedDict

# 支援的 shape types 轉成 polygon points
def shape_to_polygon(shape):
    st = shape.get("shape_type", "").lower()
    pts = shape.get("points", [])
    if st in ["polygon", "rectangle", "polyline", "linestrip", "line"]:
        # rectangle 在 labelme 會是四點或兩點，我們接受任何點數
        return pts
    elif st == "circle":
        # circle: [center, point_on_radius] -> convert to polygon approx
        if len(pts) >= 2:
            cx, cy = pts[0]
            rx, ry = pts[1]
            r = ((cx - rx)**2 + (cy - ry)**2)**0.5
            num = 40
            poly = [[cx + r * np.cos(2*np.pi*i/num), cy + r * np.sin(2*np.pi*i/num)] for i in range(num)]
            return poly
        else:
            return []
    else:
        # fallback: treat points as polygon
        return pts

def ensure_folder(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def main(args):
    json_dir = args.json_dir
    out_dir = args.out_dir
    copy_images = args.copy_images
    resize = args.resize  # tuple (w,h) or None
    label_map = OrderedDict()  # label -> id
    label_map["__background__"] = 0

    ensure_folder(out_dir)
    images_out = os.path.join(out_dir, "images")
    masks_out = os.path.join(out_dir, "masks")
    ensure_folder(masks_out)
    if copy_images:
        ensure_folder(images_out)

    json_files = [f for f in sorted(os.listdir(json_dir)) if f.lower().endswith(".json")]
    if len(json_files) == 0:
        print("No json files found in", json_dir)
        return

    for jf in json_files:
        jp = os.path.join(json_dir, jf)
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = data.get("imagePath") or data.get("imagePath", "")
        # try to find absolute image path if not present in same folder
        img_file = os.path.join(json_dir, img_path) if img_path and not os.path.isabs(img_path) else img_path
        if not img_file or not os.path.exists(img_file):
            # try same base name with common extensions
            base = os.path.splitext(jf)[0]
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp"]:
                cand = os.path.join(json_dir, base + ext)
                if os.path.exists(cand):
                    img_file = cand
                    break

        if not img_file or not os.path.exists(img_file):
            print(f"[WARN] image for {jf} not found, skipping.")
            continue

        # load image to get size
        img = Image.open(img_file)
        orig_w, orig_h = img.size

        # if resize specified, target size:
        if resize:
            target_w, target_h = resize
        else:
            target_w, target_h = orig_w, orig_h

        # create mask (single channel) initial background 0
        mask = Image.new("L", (target_w, target_h), 0)
        draw = ImageDraw.Draw(mask)

        shapes = data.get("shapes", [])
        for shape in shapes:
            label = shape.get("label", "").strip()
            if label == "":
                continue

            # assign id if not exist
            if label not in label_map:
                label_map[label] = len(label_map)  # next id

            cls_id = label_map[label]

            pts = shape_to_polygon(shape)
            if not pts:
                continue
            # scale points if resizing
            if resize:
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                pts_scaled = [(p[0]*scale_x, p[1]*scale_y) for p in pts]
            else:
                pts_scaled = [(p[0], p[1]) for p in pts]

            # convert floats to tuples of int
            poly = [tuple([float(x), float(y)]) for x,y in pts_scaled]
            # draw polygon filled with class id
            draw.polygon(poly, outline=cls_id, fill=cls_id)

        # save mask as PNG single-channel
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        mask_name = base_name + ".png"
        mask_path = os.path.join(masks_out, mask_name)
        mask.save(mask_path, format="PNG")

        # optionally copy / resize original image to images_out
        if copy_images:
            if resize:
                img_resized = img.resize((target_w, target_h), Image.BILINEAR)
            else:
                img_resized = img
            img_out_path = os.path.join(images_out, os.path.basename(img_file))
            img_resized.save(img_out_path)

        print(f"Processed {jf} -> {mask_path}")

    # write label_names.txt (one per line, index implied)
    label_names_path = os.path.join(out_dir, "label_names.txt")
    with open(label_names_path, "w", encoding="utf-8") as f:
        for k, v in label_map.items():
            f.write(k + "\n")

    print("Done. labels written to", label_names_path)
    print("Label -> id mapping:")
    for k, v in label_map.items():
        print(f"  {v}: {k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labelme jsons to single-channel masks for U-Net")
    parser.add_argument("--json_dir", required=True, help="Folder with labelme .json files (and images).")
    parser.add_argument("--out_dir", required=True, help="Output folder to save masks (and optionally images).")
    parser.add_argument("--copy_images", action="store_true", help="Copy & optionally resize source images into out_dir/images/")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W","H"), help="Resize output masks/images to this size, e.g. --resize 960 960")
    args = parser.parse_args()
    if args.resize:
        args.resize = (int(args.resize[0]), int(args.resize[1]))
    main(args)
