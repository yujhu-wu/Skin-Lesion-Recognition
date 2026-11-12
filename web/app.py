"""Skin Scan：皮膚病變辨識網站後端

使用者上傳(或用相機拍攝)皮膚影像後：
1. 以Swin Transformer判斷屬於7種皮膚病變中的哪一種
2. 以FAT-Net產生病灶區域的分割遮罩
3. 將原圖、遮罩與疾病說明繪製成一張結果圖回傳給前端
"""
import io
import os
import textwrap

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from flask import Flask, render_template, request, send_file
from PIL import Image
from torch import nn
from torchvision import models as resnet_model
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "swin_classifier.pth")
SEGMENTER_PATH = os.path.join(MODEL_DIR, "fatnet_segmentation.pth.tar")
# 結果圖中的中文字型，可用環境變數CJK_FONT指定字型檔路徑
FONT_PATH = os.environ.get("CJK_FONT", "C:/Windows/Fonts/msyh.ttc")

# 疾病類別
label_2_id = {
    'MEL': 0,
    'NV': 1,
    'BCC': 2,
    'AKIEC': 3,
    'BKL': 4,
    'DF': 5,
    'VASC': 6
}
id_2_label = {v: k for k, v in label_2_id.items()}

# 疾病說明
disease_info = {
    'MEL': "惡性黑色素瘤 [Melanoma] 是皮膚癌中最危險的一種，主要發生在皮膚表面的黑色素細胞。症狀通常包括快速生長的異常黑色斑點或痣，並且可能伴隨瘙癢、出血或潰爛。",
    'NV': "黑色素細胞痣 [Melanocytic Nevus] 是常見的良性皮膚增生，呈圓形或橢圓形，顏色範圍從肉色到深褐色，一般無需治療。",
    'BCC': "基底細胞癌 [Basal Cell Carcinoma] 是最常見的皮膚癌，主要出現在暴露於陽光的部位，如臉部和頸部，通常生長緩慢，且較少轉移。",
    'AKIEC': "鱗狀細胞癌 [Actinic Keratosis and Intraepithelial Carcinoma] 是一種侵犯性較強的皮膚癌，由長期紫外線暴露引起的皮膚病變，可能擴散至其他器官，通常表現為粗糙或鱗狀的皮膚斑塊。",
    'BKL': "良性角化病變 [Benign Keratosis] 表面粗糙或呈蠟質，顏色從黃色到黑色不等。通常是無害的老年性斑點，不需要治療。",
    'DF': "皮膚纖維瘤 [Dermatofibroma] 是一種良性纖維組織增生，通常出現在四肢，摸起來堅硬，顏色從紅棕色到暗褐色不等，常伴隨瘙癢或疼痛，一般無需治療。",
    'VASC': "血管病變 [Vascular Lesion] 包括血管瘤、血管畸形等，通常呈現為紅色或紫色斑塊，大部分是良性的，但需定期檢查。"
}

app = Flask(__name__)

SIZE = 224  # 分割模型輸入大小
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================================
# 疾病分類模型：Swin Transformer
# ==================================================================
model_name = 'swin_base_patch4_window12_384'
num_classes = 10  # 與訓練時的輸出層大小一致，實際使用前7類

model_pred = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model_pred.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE)['model'])
model_pred.to(DEVICE)
model_pred.eval()


# ==================================================================
# 病灶分割模型：FAT-Net
# ==================================================================
class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(FAT_Net, self).__init__()

        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, 14, 14)
        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out


model = FAT_Net().to(DEVICE)
model.load_state_dict(torch.load(SEGMENTER_PATH, map_location=DEVICE)['state_dict'])
model.eval()

# 分割模型的影像轉換(與訓練時一致)
transform = A.Compose([
    A.Resize(height=SIZE, width=SIZE),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
])


# ==================================================================
# 疾病分類
# ==================================================================
def preprocess_image(image):
    """分類模型的前處理：調整為384×384並正規化"""
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # 增加batch維度
    return image


def infer_image(image):
    """回傳預測的疾病代碼"""
    image = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        outputs = model_pred(image)
        probs = torch.softmax(outputs, dim=1)
        pred_id = probs.argmax(dim=1).item()
        return id_2_label.get(pred_id, "Unknown")


# ==================================================================
# 病灶分割
# ==================================================================
def generate_mask(img):
    """輸入RGB影像(numpy)，回傳處理後的影像與二值化遮罩"""
    img = transform(image=img)["image"]

    # 轉成張量並歸一化
    img_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

    with torch.no_grad():
        generated_mask = model(img_tensor).squeeze().cpu().numpy()

    # 歸一化模型輸出，避免遮罩邊界模糊、顏色過淡
    mask_min, mask_max = np.min(generated_mask), np.max(generated_mask)
    generated_mask_normalized = (generated_mask - mask_min) / (mask_max - mask_min)

    # 高斯模糊平滑邊緣後二值化，讓遮罩邊界更清楚
    threshold_value = 0.6
    generated_mask_blurred = cv2.GaussianBlur(generated_mask_normalized, (5, 5), 0)
    generated_mask_binary = (generated_mask_blurred > threshold_value).astype(np.float32)
    generated_mask_resized = cv2.resize(generated_mask_binary, (img.shape[1], img.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
    return img, np.stack((generated_mask_resized,) * 3, axis=-1)


def render_result(img, mask, pred_label):
    """把原圖、遮罩與疾病說明畫成一張PNG"""
    prop = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None
    plt.rcParams['axes.unicode_minus'] = False

    wrapped_text = textwrap.fill(
        f'Picture prediction results: {pred_label}\n{disease_info[pred_label]}', width=45)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(mask)
    ax[1].set_title('Generated Mask')
    fig.text(0.5, 0.2, wrapped_text, color='black', fontsize=18, ha='center', va='top', fontproperties=prop)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            pred_label = infer_image(file)

            file.stream.seek(0)
            img = np.array(Image.open(file).convert('RGB'))
            img, mask = generate_mask(img)

            return send_file(render_result(img, mask, pred_label), mimetype='image/png')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
