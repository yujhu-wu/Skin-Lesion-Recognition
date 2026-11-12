from flask import Flask, request, render_template
import torch
import timm
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# ID 到标签的映射
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

#swin_base_patch4_window12_384
#resnext50_32x4d

def load_model(device):
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=7)
    model.load_state_dict(torch.load('model_files/swin_classifier.pth', map_location=device),strict = False)
    #model.load_state_dict(torch.load(model_path),strict = False)
    model.to(device)
    model.eval()
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # 添加批量维度
    return image

def get_prediction(image, model, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        prediction = output.cpu().numpy()
    predicted_label = prediction.argmax(1)[0]
    return id_2_label[predicted_label]

# 加载模型并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            image = transform_image(image)
            prediction = get_prediction(image, model, device)
            return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
