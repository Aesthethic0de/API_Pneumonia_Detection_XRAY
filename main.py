from fastapi import FastAPI, UploadFile
import torch
from model.model_resnet import Resnet_Pneumonia
import uvicorn
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.model_resnet import Resnet_Pneumonia
from model.Loader import XrayDataset
import glob
from config.config import Path
app = FastAPI()
model = Resnet_Pneumonia(pretrained=True)

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(size=(500,500)),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
@app.post("/xray")
async def image_test(file : UploadFile):
    y_pred = []
    file = file.file.read()
    test_folder = glob.glob(Path.test_folder)
    path_test = test_folder
    with open(Path.img_save, "wb") as f:
        f.write(file)
    test_dataset = XrayDataset(path_test ,transform=transform_test)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    model.load_state_dict(torch.load(Path.weight_location ,map_location=torch.device('cpu')))
    model.eval()
    for i, tensors in enumerate(test_data_loader):
        with torch.no_grad():
            predictions = model(tensors.cpu())
            predictions = predictions.sigmoid()
            predictions = predictions > 0.5
            y_pred.append(predictions)
            y_pred = torch.cat(y_pred)
            y_pred = y_pred.cpu().numpy()
            y_pred = y_pred.astype(np.int64)
            y_pred = y_pred.reshape(-1)
            result = str(y_pred)
            if result[1] == '0':
                return {"Result" : "you are Normal"}
            else:
                return {"Result" : "you have pneumonia"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")

