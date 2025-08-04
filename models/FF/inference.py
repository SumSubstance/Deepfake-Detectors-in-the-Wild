import os
import argparse
import cv2
import dlib
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

sys.path.insert(0, 'FaceForensics')
from network.models import model_selection
from detect_from_video import preprocess_image, get_boundingbox

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import seed_everything


class FF_dataset(Dataset):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.imgs = [file for file in os.listdir(self.input_dir) if (file.endswith('.jpg') or file.endswith('.png'))]
        self.face_detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.input_dir, self.imgs[index])
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        if len(faces):
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            return preprocess_image(cropped_face, cuda=False)

        return preprocess_image(image, cuda=False)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model = torch.load('FaceForensics/faceforensics++_models_subset/full/xception/full_c40.p')
    model = model.to(device)
    model.eval()
    post_function = nn.Softmax(dim=1)

    dataset = FF_dataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    all_preds = []
    for images in tqdm(loader, desc='FF++'):
        images = images.to(device)
        output = model(images)
        output = post_function(output)
        all_preds.extend(output[:, 1].cpu().tolist())

    df = pd.DataFrame({'File': dataset.imgs, 'Pred': all_preds})
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_csv", type=str, required=True, help="csv to store results")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num of workers for dataloader")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")

    args = parser.parse_args()
    main(args)
