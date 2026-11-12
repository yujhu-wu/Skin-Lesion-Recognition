# 網站開發版本紀錄

網站從只能分類的單一頁面，逐步加入病灶分割、相機拍攝、HTTPS與正式部署，以下保留各階段的程式碼。最終版本在[`web/`](../)。

| 版本 | 新增功能 | 說明 |
|---|---|---|
| [v1-classification](v1-classification/) | 疾病分類 | 第一版網站：上傳影像後以Swin Transformer預測疾病類別，結果直接顯示在頁面上 |
| [v2-segmentation](v2-segmentation/) | 病灶分割 | 單獨測試FAT-Net：上傳影像後回傳原圖與預測遮罩的對照圖 |
| [v3-classification-and-mask](v3-classification-and-mask/) | 分類+分割整合 | 同一次上傳同時完成疾病分類與病灶分割；加入遮罩歸一化、高斯模糊與二值化，改善遮罩邊界模糊、顏色過淡的問題 |
| [v4-camera-capture](v4-camera-capture/) | 相機拍攝 | 前端加入`Capture Photo`：以`getUserMedia`開啟相機，拍照後轉成檔案送出，可重拍 |
| [v5-https-camera](v5-https-camera/) | 疾病說明、HTTPS | 結果圖加入7種疾病的中文說明；相機只能在安全連線下使用，因此以自簽憑證啟動HTTPS測試 |
| [最終版](../) | 正式上線 | 前後鏡頭切換、上傳前預覽、未選擇圖片時停用上傳按鈕、就醫提醒；以IIS反向代理部署，改用正式憑證與自有網域 |

## 執行方式

每個版本都是獨立的Flask程式，模型放在該版本資料夾的`model_files/`中(檔名與最終版相同)：

```bash
cd v3-classification-and-mask
python app.py
```

`v5-https-camera`需要先在同一資料夾產生`server.crt`與`server.key`憑證檔(未收錄)。
