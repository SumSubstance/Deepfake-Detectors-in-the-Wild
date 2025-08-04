import sys
import os
import torch
import cv2
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from albumentations import CenterCrop,Compose,Resize,RandomCrop, Normalize
from albumentations.pytorch.transforms import ToTensorV2

sys.path.insert(0, 'multiple-attention/preprocessing')
sys.path.insert(0, 'multiple-attention/retinaface')
from face_utils import FaceDetector
from slave_crop import crop_unalinged

sys.path.remove('multiple-attention/retinaface')
sys.path.insert(0, 'multiple-attention')
import importlib
import models
importlib.reload(models)
import utils
importlib.reload(utils)
from models.MAT import MAT

sys.path.remove('multiple-attention')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
importlib.reload(utils)
from utils import seed_everything


def _extract_face_photo(input_file, output_file, aligned_image_size=380,
                        unaligned_padding=0.2, unaligned_image_size=(380, 380),
                        det=None):
    image = cv2.imread(input_file)
    if det is None:
        if image.shape[0] >= aligned_image_size and image.shape[1] >= aligned_image_size:
            image = Image.fromarray(cv2.resize(image, unaligned_image_size))
            image.save(output_file)
            return
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            image = torch.nn.functional.pad(image,
                                            (0, unaligned_image_size[1] - image.shape[1], 0,
                                             unaligned_image_size[1] - image.shape[2]))
            image = image.numpy().transpose((1, 2, 0))
            image = Image.fromarray(cv2.resize(image, unaligned_image_size))
            image.save(output_file)
            return

    box = det[:4]
    img_crop = crop_unalinged(image, box, unaligned_padding, unaligned_image_size)
    img_crop.save(output_file)
    return


def calc_num_images(input_dir):
    return len([file for file in os.listdir(input_dir) if (file.endswith('jpg') or file.endswith('.png'))])


class MAT_Dataset(Dataset):
    def __init__(self, input_dir: str, batch_size: int, faces_dir: str,
                 resize=(380, 380)):
        self.faces_dir = faces_dir if faces_dir is not None else f"{input_dir}_faces"
        if os.path.exists(self.faces_dir) and calc_num_images(self.faces_dir) == calc_num_images(input_dir):
            print('skipping detection')
        else:
            face_detector = FaceDetector(device='cuda:0')
            face_detector.load_checkpoint('multiple-attention/retinaface/Resnet50_Final.pth')
            self.face_detector = face_detector
            self._preproc(input_dir, self.faces_dir, batch_size)
            self.face_detector = None  # should free up GPU memory after working

        self.imgs = [file for file in os.listdir(self.faces_dir) if (file.endswith(".jpg") or file.endswith(".png"))]
        self.trans = Compose([CenterCrop(*resize),
                              Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ToTensorV2()])

    def _preproc(self, input_dir, faces_dir, batch_size):
        os.makedirs(faces_dir, exist_ok=True)
        all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)
                     if (file.endswith(".jpg") or file.endswith(".png"))]
        for i in tqdm(range(0, len(all_files), batch_size), desc='Face Detection'):
            files = all_files[i:i+batch_size]
            now_img_files = []
            for file in files:
                image = cv2.imread(file)
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                now_img_files.append(image)
            det = self.face_detector.detect(now_img_files)
            det_new = [face[0] if len(face) != 0 else None for face in det]  # takes only one face in detect

            for file, det in zip(files, det_new):
                output_file = os.path.join(faces_dir, os.path.basename(file))
                _extract_face_photo(file, output_file, det=det)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.faces_dir, self.imgs[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.trans(image=np.array(img))['image']


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MAT_Dataset(args.input_dir, args.batch_size_detect, args.faces_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = MAT(net='efficientnet-b4', pretrained='multiple-attention/pretrained/ff_c40.pth')
    model.to(device)
    model.eval()

    all_preds = list()
    for images in tqdm(loader, desc='MAT'):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            pred = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            all_preds.extend(pred.cpu().tolist())

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
