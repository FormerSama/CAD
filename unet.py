import os
import cv2
import torch
import numpy as np
from typing import Tuple, Dict, List, Union
from train_unet import UNet  # 或 AttentionUNet 依你訓練的模型而定
from utils import crop_table_from_mask, detect_polygon, split_and_pad
from utils import concat

# ========== 模型載入 ==========
def load_unet_model(weights_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet()  # 如果你用的是 AttentionUNet，就改成 AttentionUNet()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ========== 前處理 & 後處理 ==========
def preprocess_image(img: np.ndarray, resize: Tuple[int, int] = (960, 960)) -> torch.Tensor:
    """
    img: BGR (H, W, 3)
    回傳: Tensor (1, 3, H, W)，值在 [0,1]
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, resize)

    x = img.transpose(2, 0, 1)  # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0).float() / 255.0
    return x


def postprocess_pred(pred: torch.Tensor, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    pred: (1, 1, H, W)，值域 [0,1]
    回傳:
        prob_map: float32 (H,W) [0,1]
        mask    : uint8  (H,W) 0/255
    """
    prob = pred.squeeze().cpu().numpy().astype(np.float32)
    mask = (prob > threshold).astype(np.uint8) * 255
    return prob, mask

InputType = Union[str, List[np.ndarray]]

def predict(
    model: torch.nn.Module,
    src: InputType,
    device: str,
    resize: Tuple[int, int] = (960, 960),
    threshold: float = 0.5,
    save: bool = True,
    out_dir: str = "auto_masks",
    prefix: str = "pred"
):
    """
    根據 src 型態，做「方法 overloading」的效果：

    - 若 src 是 str：
        視為「資料夾路徑」，讀取其中所有 .jpg/.jpeg/.png 來做推論。
        回傳: dict[str, (prob_map, mask)]，key 是檔名

    - 若 src 是 list[np.ndarray]：
        視為「前面 split_and_pad() 回傳的 tiles」(每個 BGR patch)
        回傳: list[(prob_map, mask)]，index 對應 tile 順序

    參數:
        model : 已經 load 好權重、model.eval() 的模型
        src: 資料夾路徑或 tiles list
        device: "cuda" or "cpu"
        resize: 若 tiles 已經是 window size，可以設 None 不 resize
        threshold: 分割二值化門檻
        save: 是否將 mask 寫成檔案
        out_dir: 輸出資料夾 (save=True 時)
        prefix: tiles 輸出時用的檔名前綴
    """

    os.makedirs(out_dir, exist_ok=True) if save else None

    # ========== Case 1: 資料夾路徑 ==========
    if isinstance(src, str):
        src_dir = src
        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for f in os.listdir(src_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(src_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"讀取失敗: {img_path}")
                continue

            x = preprocess_image(img, resize=resize).to(device)

            with torch.no_grad():
                pred = model(x)

            prob_map, mask = postprocess_pred(pred, threshold=threshold)
            results[f] = (prob_map, mask)

            if save:
                base = os.path.splitext(f)[0]
                out_path = os.path.join(out_dir, f"{base}.png")
                cv2.imwrite(out_path, mask)
                print("saved mask:", out_path)

        return results

    # ========== Case 2: tiles list[np.ndarray] ==========
    elif isinstance(src, list) and (len(src) == 0 or isinstance(src[0], np.ndarray)):
        tiles: List[np.ndarray] = src
        results_tiles: List[Tuple[np.ndarray, np.ndarray]] = []

        for idx, tile in enumerate(tiles):
            img = tile  # BGR patch (H, W, 3)

            x = preprocess_image(img, resize=resize).to(device)

            with torch.no_grad():
                pred = model(x)

            prob_map, mask = postprocess_pred(pred, threshold=threshold)
            results_tiles.append((prob_map, mask))

            if save:
                out_path = os.path.join(out_dir, f"{prefix}_{idx}.png")
                cv2.imwrite(out_path, mask)
                print("saved tile mask:", out_path)

        return results_tiles

    else:
        raise TypeError("inputs 必須是資料夾路徑 (str) 或 list[np.ndarray] (tiles)")


if __name__ == '__main__':
    '''
    img_src = 'test/page_11.jpeg'
    patch, H, W = split_and_pad(img_path=img_src, out_dir=None, window=960, stride=480, save=False)
    model, device = load_unet_model(weights_path='unet_best.pth')
    
    pred_patchs = predict(model=model, src=patch, device=device, resize=None, threshold=0.5, save=False, out_dir=None, prefix='test')
    prob_patches = [pm[0] for pm in pred_patchs]  # 每個是 (h, w) 的 float32

    # 4. 用 concat 把所有 prob patch 拼回一張大圖
    full_prob = concat(
        patches=prob_patches,     # 注意：這裡是整包 list，不是 pred_patchs[1]
        original_shape=(H, W),
        window=960,
        stride=480
    )

    # 5. 變成二值 mask 再存檔（避免剛剛那個 imwrite warning）
    full_mask = (full_prob > 0.5).astype(np.uint8) * 255
    
    full_mask = cv2.imread('test.png')

    poly = detect_polygon(mask=full_mask)
    for p in poly:
        print(f"id: {p.id}")
        print(f"area: {p.area}")
        print(f"centroid: {p.centroid}")
        print(f"polygon: {p.polygon}")
    '''

    mask = cv2.cvtColor(cv2.imread('test/masks/page_10.png'), cv2.COLOR_RGB2BGR)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    img = cv2.imread('test/images/page_10.jpeg')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    num_classes = 3  # 背景, symbol, table
    mask_id = np.rint(mask_gray / 255.0 * (num_classes - 1)).astype(np.int64)
    table_img, bbox = crop_table_from_mask(img_bgr, mask_id, table_class_id=2, margin=10)
    print("table bbox:", bbox)
    cv2.imwrite("table_cropped.png", table_img)
    #cv2.imwrite('test/test.png', full_mask)
    
