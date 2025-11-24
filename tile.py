import os
from utils import split_and_pad

# 參數設定
IMG_DIR = "test"       # 原始圖片資料夾
OUT_DIR = "test/images"   # 切割後的輸出資料夾
WINDOW = 960
STRIDE = 480

os.makedirs(OUT_DIR, exist_ok=True)



# 主程式：批次處理資料夾
for fname in os.listdir(IMG_DIR):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        
        split_and_pad(os.path.join(IMG_DIR, fname), OUT_DIR, WINDOW, STRIDE)

print("全部完成！")
