from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
import numpy as np

def get_classes(file_path):
    # Create an instance of 'myModel' imported above
    resnet = models.resnet101(pretrained=True)

    img = Image.open(file_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    preprocessed = preprocess(img)
    img_tensor = torch.unsqueeze(preprocessed, 0)

    preds = resnet(img_tensor)
    #currPath = os.path.dirname(__file__)
    #txtfile = os.path.join()
    with open('./imagenetclasses.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    #top3_prob, top3_idx = torch.topk(preds,largest=True, k=3)
    #_,index = torch.max(preds, 1)
    percentage = torch.nn.functional.softmax(preds, dim=1)[0]*100
    #print(labels[index[0]], percentage[index[0]].item())
    _,indices = torch.sort(preds, descending=True)
    top5_labels = [(labels[idx]) for idx in indices[0][:3]]

    top5_prob = [(percentage[idx].item()) for idx in indices[0][:3]]
    print("start\n")
    print(top5_prob)
    print("\n")
    print(top5_labels)
    print("end\n")
    #[print(idx) for idx in top3_idx[0]]
    return (top5_labels,top5_prob)
