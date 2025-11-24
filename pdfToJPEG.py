from pdf2image import convert_from_path

# 將 PDF 轉成圖片
pages = convert_from_path("【02】1140903-黑白掃描-共63張-合併檔.pdf", dpi=300)

# 每一頁存成 JPEG
for i, page in enumerate(pages):
    page.save(f"page_{i+1}.jpeg", "JPEG")
