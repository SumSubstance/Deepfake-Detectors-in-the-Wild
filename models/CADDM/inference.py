import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import dlib
from collections import OrderedDict
from torchvision.transforms import CenterCrop, Resize
from imutils import face_utils
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import argparse

sys.path.insert(0, 'CADDM')
from lib.data_preprocess.preprocess import prepare_test_input
from lib.util import load_config
import model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import seed_everything


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = k
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


class CADMM_dataset(Dataset):
    def __init__(self, images_path: str, config: dict):
        self.config = config
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor("CADDM/lib/shape_predictor_81_face_landmarks.dat")
        self.imgs = [file for file in os.listdir(images_path) if (file.endswith('.jpg') or file.endswith('.png'))]
        self.images_path = images_path
        self.crop = CenterCrop(224)
        self.resize = Resize(224)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.images_path, self.imgs[index])
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Landmarks
        faces = self.face_detector(frame, 1)
        if len(faces) == 0:
            if frame.shape[0] < 224:
                return self.resize(torch.Tensor(frame.transpose(2, 0, 1))), 1
            return self.crop(torch.Tensor(frame.transpose(2, 0, 1))), 1

        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        for face_idx in range(len(faces)):
            landmark = self.face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        # save the landmark with the biggest face
        landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

        # Cropping
        img, label_dict = prepare_test_input(
                [frame], landmarks, 1, config=self.config
        )
        return torch.Tensor(img[0].transpose(2, 0, 1)), 0


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = load_config('CADDM/configs/caddm_test.cfg')
    net = model.get(backbone=cfg['model']['backbone'])
    net = load_checkpoint(cfg['model']['ckpt'], net, device)
    net = net.to(device)
    net.eval()

    dataset = CADMM_dataset(args.input_dir, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    all_preds = list()
    all_errors = list()
    for batch, error_labels in tqdm(loader):
        batch = batch.to(device)
        with torch.no_grad():
            preds = net(batch)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_errors.extend(error_labels)

    print(f"face wasn't detected on {(np.array(all_errors) == 1).sum() / len(all_errors)} % images")
    df = pd.DataFrame({'File': dataset.imgs, 'Pred': list(np.array(all_preds)[:, 1])})
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
