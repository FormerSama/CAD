import json
import os

# 載入 PaddleOCR 輸出
dirPath = 'output/label'
allFilePath = os.listdir(dirPath)
with open("train_label.txt", "w", encoding="utf-8") as outputFile:
    for e in allFilePath:
        filePath = os.path.join(dirPath, e)
        print(filePath)
        with open(filePath, "r", encoding="utf-8") as f:
            data = json.load(f)
            image_path = data["input_path"].replace('\\', '/')
            polys = data["dt_polys"]
            texts = data['rec_texts']

        # 模擬 OCR 辨識結果（若已有文字）
        results = []
        for index in range(len(polys)):
            results.append({
                "transcription": texts[index], 
                "points": polys[index],
                "difficult": False
            })
        # 輸出為 PaddleOCR 訓練格式
        out_line = f"{image_path}\t{json.dumps(results, ensure_ascii=False)}"
        outputFile.write(out_line + "\n")    



    
