from mobilenetv2 import MobileNetV2
import torch

model = MobileNetV2()
model.load_state_dict(torch.load(open('mobilenet_v2.pth.tar', 'rb')))
print(model)