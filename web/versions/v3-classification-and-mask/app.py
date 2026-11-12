from flask import Flask, request, render_template, send_file
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import albumentations as A
import io

import torch
from torchvision import models as resnet_model
from torch import nn

from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import timm

# 标签映射字典
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

app = Flask(__name__)

# 設定模型的相關參數
SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
model_name = 'swin_base_patch4_window12_384'
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_pred = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model_pred.load_state_dict(torch.load('model_files/swin_classifier.pth')['model'])
model_pred.to(device)
model_pred.eval()

####################################################################
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
####################################################################

# 定义图片预处理步骤
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加一个维度以匹配批量维度
    return image

# 进行推理
def infer_image(image):
    image = preprocess_image(image)
    image = image.to(device)
    with torch.no_grad():
        outputs = model_pred(image)
        probs = torch.softmax(outputs, dim=1)
        pred_id = probs.argmax(dim=1).item()
        pred_label = id_2_label.get(pred_id, "Unknown")
        return pred_label

# 加载模型
model = FAT_Net().to(DEVICE)  # 使用自定义模型类
model.load_state_dict(torch.load("model_files/fatnet_segmentation.pth.tar")['state_dict'])
model.eval()

# 图像转换（与训练时一致）
transform = A.Compose([ 
    A.Resize(height=SIZE, width=SIZE),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
])

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            #呼叫預測圖片
            pred_label = infer_image(file)

            # 读取并处理图像
            img = Image.open(file).convert('RGB')
            img = np.array(img)
            
            # 应用与训练一致的变换
            transformed = transform(image=img)
            img = transformed["image"]
            
            # 将图像转换为 PyTorch 张量并进行归一化
            img_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            img_tensor = img_tensor / 255.0  # 进行归一化处理
            
            print("Image Tensor:", img_tensor.shape, img_tensor.device)

            # 生成模型的預測遮罩
            with torch.no_grad():
                generated_mask = model(img_tensor).squeeze().cpu().numpy()

            print("Generated Mask:", generated_mask.shape, np.min(generated_mask), np.max(generated_mask))

            # 归一化生成的遮罩
            mask_min, mask_max = np.min(generated_mask), np.max(generated_mask)
            generated_mask_normalized = (generated_mask - mask_min) / (mask_max - mask_min)

            # 應用二值化處理
            threshold_value = 0.6  # 定義閾值
            generated_mask_binary = (generated_mask_normalized > threshold_value).astype(np.float32)
            
            # 调整生成遮罩的大小（如有必要）
            generated_mask_blurred = cv2.GaussianBlur(generated_mask_normalized, (5, 5), 0)
            generated_mask_binary = (generated_mask_blurred > threshold_value).astype(np.float32)
            generated_mask_resized = cv2.resize(generated_mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            generated_mask_stacked = np.stack((generated_mask_resized,) * 3, axis=-1)
            
            # 将结果图像保存到字节流中
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img)
            ax[0].set_title('Original Image')
            ax[1].imshow(generated_mask_stacked)
            ax[1].set_title('Generated Mask')
            fig.text(
                0.5, 0.06, f'Picture prediction results: {pred_label}', 
                color='black', fontsize=24, ha='center', va='top'
            )
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            # 返回生成的图像作为响应
            return send_file(buf, mimetype='image/png')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
