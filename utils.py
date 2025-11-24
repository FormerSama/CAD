from typing import Dict, Tuple, List
import cv2
import numpy as np
import os

from geometry import PolygonRegion

def otsu(img) -> int:
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu Threshold: {ret}")
    return th

def soble(img, ksize, alp, beta) -> np.ndarray:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)

    sobelx_abs = cv2.convertScaleAbs(sobelx)
    sobely_abs = cv2.convertScaleAbs(sobely)

    sobel_combined = cv2.addWeighted(sobelx_abs, alp, sobely_abs, beta, 0)
    return sobel_combined

def gaussianBlur(img, sigmaX, sigmaY) -> np.ndarray:
    blurredImage = cv2.GaussianBlur(img, sigmaX, sigmaY)
    return blurredImage

def grayScale(img) -> np.ndarray:
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

def closing(img, ksize) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closingImage = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closingImage

def opening(img, ksize) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    openingImage = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return openingImage

def preprocess_mask(mask, straighten=True, simplify=True):
    # 1. 二值化 (避免多餘灰階雜訊)
    mask = (mask > 0).astype(np.uint8) * 255

    # 2. 形態學平滑，去除鋸齒
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    if straighten:
        # 霍夫線修正主方向
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=30, maxLineGap=10)
        if lines is not None:
            out = np.zeros_like(mask)
            all_angles = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                all_angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            mean_angle = np.median(all_angles)
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if abs(mean_angle) < 45:  # 水平
                    y_mean = int((y1 + y2) / 2)
                    cv2.line(out, (x1, y_mean), (x2, y_mean), 255, 2)
                else:                      # 垂直
                    x_mean = int((x1 + x2) / 2)
                    cv2.line(out, (x_mean, y1), (x_mean, y2), 255, 2)
            mask = out

    if simplify:
        # 多邊形簡化：讓近似矩形的區塊更筆直
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 4:
                cv2.drawContours(out, [approx], -1, 255, -1)
        mask = out

    return mask

def split_and_pad(
    img_path: str,
    out_dir: str,
    window: int,
    stride: int,
    save: bool = True
) -> Tuple[List[np.ndarray], int, int]:
    """
    將「單張」影像切成多個 window x window 的 patch，
    不足的地方用白色(255)補齊。

    參數：
        img_path (str): 輸入影像路徑
        out_dir  (str): 輸出資料夾 (若 save=False 可以隨便給或設 None)
        window   (int): patch 的邊長 (window x window)
        stride   (int): 每次滑動的步幅
        save    (bool): 是否將切好的 patch 存成檔案，預設 True

    回傳：
        patches (list[np.ndarray]): 該張影像切出的所有 patch，shape=(H, W, 3)
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("讀取失敗:", img_path)
        return []

    h, w = img.shape[:2]
    fname = os.path.splitext(os.path.basename(img_path))[0]

    # 若要存檔，確保輸出資料夾存在
    if save and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    patches: List[np.ndarray] = []
    count = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 建立一張白色背景
            patch = np.full((window, window, 3), 255, dtype=np.uint8)

            # 擷取實際存在的區域（可能在邊界會比 window 小）
            crop = img[y:y+window, x:x+window]

            # 貼到白色背景左上角
            patch[0:crop.shape[0], 0:crop.shape[1]] = crop

            # 收集到 list 裡
            patches.append(patch)

            # 視需求輸出成檔案
            if save and out_dir is not None:
                out_path = os.path.join(out_dir, f"case2_{fname}_{count}.jpg")
                cv2.imwrite(out_path, patch)

            count += 1

    print(f"{img_path} 已切成 {count} 張 patch (save={save})")
    return patches, img.shape[0], img.shape[1]


def concat(
    patches: List[np.ndarray],
    original_shape: Tuple[int, int],
    window: int,
    stride: int
) -> np.ndarray:
    """
    將依照 sliding window 切出來、又經過模型預測後的 patches
    按照原本座標「拼回去」成一張大圖。

    假設 patches 的順序是：
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # 對應一個 patch

    參數：
        patches: list[np.ndarray]
            - 每個元素是單一 patch 的預測結果
            - shape 可以是 (h, w) 或 (h, w, c)
              其中 h,w 通常是 window，但邊界可能會比較小（你有做 padding 時，一般還是 window）
        original_shape: (H, W)
            - 原始大圖的高、寬（沒縮放前 or 你希望重建的大小）
        window: int
            - 當初切 patch 時用的 window 大小
        stride: int
            - 當初切 patch 時用的 stride

    回傳：
        full_pred: np.ndarray
            - 拼回來的大圖，shape 為 (H, W) 或 (H, W, C)
    """

    H, W = original_shape

    if len(patches) == 0:
        raise ValueError("patches 是空的，無法拼回大圖")

    # 檢查 patch 維度 (2D or 3D)
    sample = patches[0]
    if sample.ndim == 2:
        channels = 1
        full_pred = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
    elif sample.ndim == 3:
        h_p, w_p, c = sample.shape
        channels = c
        full_pred = np.zeros((H, W, c), dtype=np.float32)
        weight = np.zeros((H, W, c), dtype=np.float32)
    else:
        raise ValueError("patches 裡的 ndarray 維度必須是 2D 或 3D (H,W) 或 (H,W,C)")

    idx = 0
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            if idx >= len(patches):
                # 小心，patch 數量不夠
                break

            patch = patches[idx].astype(np.float32)
            idx += 1

            # 有可能最後一個 patch 會超出原始大小，所以算實際有效區域
            patch_h = min(window, H - y)
            patch_w = min(window, W - x)

            if channels == 1:
                # 2D
                full_pred[y:y+patch_h, x:x+patch_w] += patch[:patch_h, :patch_w]
                weight[y:y+patch_h, x:x+patch_w] += 1.0
            else:
                # 3D
                full_pred[y:y+patch_h, x:x+patch_w, :] += patch[:patch_h, :patch_w, :]
                weight[y:y+patch_h, x:x+patch_w, :] += 1.0

    # 避免除以 0
    if channels == 1:
        full_pred /= np.maximum(weight, 1e-6)
    else:
        full_pred /= np.maximum(weight, 1e-6)

    return full_pred

def detect_polygon(
    mask: np.ndarray,
    min_area: float = 1.0,
    approx_epsilon_ratio: float = 0.01
) -> List[PolygonRegion]:
    """
    從 binary mask 中取得每個白色區塊的多邊形表示、面積與中心點。

    參數：
        mask : 單通道或彩色影像，白色為前景
        min_area : 面積小於此值的區塊會被忽略
        approx_epsilon_ratio : 多邊形簡化程度，
                               epsilon = approx_epsilon_ratio * 周長
                               設 0 代表不用簡化，保留所有輪廓點

    回傳：
        List[PolygonRegion]
    """

    # 1. 確保是單通道
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()

    # 2. 確保是 0/255 binary
    _, binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3. 找輪廓（只找外輪廓即可）
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions: List[PolygonRegion] = []

    for idx, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 4. 算中心點 (質心)
        M = cv2.moments(cnt)
        if M["m00"] == 0:  # 避免除以 0
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 5. 多邊形頂點（視需求簡化）
        if approx_epsilon_ratio > 0:
            peri = cv2.arcLength(cnt, True)
            epsilon = approx_epsilon_ratio * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            approx = cnt  # 不簡化，保留全部輪廓點

        # approx 形狀是 (N,1,2)，把它轉成 [(x,y), ...]
        poly_points = [(int(p[0][0]), int(p[0][1])) for p in approx]

        regions.append(
            PolygonRegion(
                id=idx,
                area=float(area),
                centroid=(float(cx), float(cy)),
                polygon=poly_points,
            )
        )

    return regions

def crop_table_from_mask(img_bgr: np.ndarray,
                         mask_id: np.ndarray,
                         table_class_id: int = 2,
                         margin: int = 10) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    """
    根據 table 類別的 mask，在原圖中裁出整張表格區塊。

    參數:
        img_bgr        : 原始彩色圖 (H,W,3)
        mask_id        : class id mask (H,W)
        table_class_id : table 的 class id
        margin         : 裁切時四周預留的像素 (避免剛好切到邊界)

    回傳:
        table_bgr : 裁切出來的表格圖 (h', w', 3)
        bbox      : (x, y, w, h) 在原圖中的位置
    """
    assert img_bgr.shape[:2] == mask_id.shape, "影像與 mask 尺寸需相同"

    table_mask = (mask_id == table_class_id).astype(np.uint8)  # (H,W)

    ys, xs = np.where(table_mask == 1)
    if len(ys) == 0 or len(xs) == 0:
        # 沒有偵測到 table，回傳原圖 + 全圖 bbox
        H, W = img_bgr.shape[:2]
        return img_bgr.copy(), (0, 0, W, H)

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # 加 margin 並限制在圖內
    H, W = img_bgr.shape[:2]
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, W - 1)
    y2 = min(y2 + margin, H - 1)

    table_bgr = img_bgr[y1:y2+1, x1:x2+1].copy()
    bbox = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
    return table_bgr, bbox



