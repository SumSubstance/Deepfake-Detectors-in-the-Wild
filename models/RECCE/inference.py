import cv2
import torch
import argparse
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import random
import numpy as np

sys.path.insert(0, 'RECCE/Pytorch_Retinaface')
from detect import extract_faces

sys.path.insert(0, 'RECCE')
from model.network import Recce
from model.common import freeze_weights
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def preprocess(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    compose = Compose([Resize(height=299, width=299),
                       Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                       ToTensorV2()])
    img = compose(image=img)['image']
    return img


class RECCE_dataset(Dataset):
    def __init__(self, input_dir: str, batch_size: int, faces_dir: str):
        if faces_dir is None:
            faces_dir = f"{input_dir}_faces"
        if os.path.exists(faces_dir) and len(os.listdir(faces_dir)) == len(os.listdir(input_dir)):
            print('skipping detection')
        else:
            extract_faces(input_dir, faces_dir, batch_size)
        self.imgs = [os.path.join(faces_dir, file) for file in os.listdir(faces_dir)
                     if (file.endswith(".jpg") or file.endswith(".png"))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_path = self.imgs[index]
        img = preprocess(img_path)
        return img, img_path


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = eval("Recce")(num_classes=1)
    weights = torch.load('RECCE/model_params_ffpp_c40.bin', map_location="cpu")["model"]
    model.load_state_dict(weights)
    model = model.to(device)
    freeze_weights(model)
    model.eval()

    dataset = RECCE_dataset(args.input_dir, args.batch_size_detect, args.faces_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_preds = list()
    for imgs, pt in tqdm(loader, total=len(loader), desc='RECCE'):
        imgs = imgs.to(device)
        with torch.no_grad():
            predictions = model(imgs)
            predictions = torch.sigmoid(predictions).cpu()
        all_preds.extend(predictions.squeeze().tolist())

    df = pd.DataFrame({'File': dataset.imgs, 'Pred': all_preds})
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_csv", type=str, required=True, help="csv to store results")
    parser.add_argument("--faces_dir", type=str, default=None, help="Directory containing extracted faces")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--batch_size_detect", type=int, default=32, help="batch size for retinaface")
    parser.add_argument("--num_workers", type=int, default=8, help="num of workers for dataloader")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")

    args = parser.parse_args()
    main(args)
