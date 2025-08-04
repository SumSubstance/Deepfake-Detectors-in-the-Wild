from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import sys
import argparse
import random
from tqdm.auto import tqdm
import pandas as pd
import torch
from PIL import Image
import albumentations

sys.path.insert(0, 'M2TR')
from M2TR.datasets.utils import get_augmentations_from_list
from M2TR.models.m2tr import M2TR

sys.path.insert(0, '../RECCE/RECCE/Pytorch_Retinaface')
from detect import extract_faces


DETECT_WEIGHTS = 'M2TR/Resnet50_Final.pth'


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

def calc_num_images(input_dir):
    return len([file for file in os.listdir(input_dir) if (file.endswith('jpg') or file.endswith('.png'))])


class M2TRDataset(Dataset):
    def __init__(self, input_dir: str, batch_size: int, faces_dir: str):
        self.faces_dir = faces_dir if faces_dir is not None else f"{input_dir}_faces"
        if os.path.exists(self.faces_dir) and calc_num_images(self.faces_dir) == calc_num_images(input_dir):
            print('skipping detection')
        else:
            extract_faces(input_dir, self.faces_dir, batch_size, DETECT_WEIGHTS)

        self.imgs = [file for file in os.listdir(self.faces_dir) if (file.endswith(".jpg") or file.endswith(".png"))]

        aug_cfg = {'IMG_SIZE': 320, 'COMPOSE': ['Resize', 'Normalize'],
                   'NORMALIZE_PARAMS': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                   'RESIZE_PARAMS': (320, 320)}
        ops = get_augmentations_from_list(aug_cfg['COMPOSE'], aug_cfg)
        ops.append(albumentations.pytorch.transforms.ToTensorV2())
        self.augmentations = albumentations.Compose(ops, p=1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.faces_dir, self.imgs[index])
        img = Image.open(img_path)
        img = np.asarray(img)
        img = self.augmentations(image=img)['image']
        return {'img': img}


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_cfg = {"IMG_SIZE": 320, "BACKBONE": 'efficientnet-b4',
                 "TEXTURE_LAYER": 'b2', "FEATURE_LAYER": 'final',
                 "DEPTH": 4, "NUM_CLASSES": 2, "DROP_RATIO": 0.5,
                 "HAS_DECODER": False}
    model = M2TR(model_cfg)
    post_func = torch.nn.Softmax(dim=1).to(device)
    model.load_state_dict(torch.load('M2TR/M2TR_FFDF_c40.pyth', map_location=device)['model_state'])
    model.to(device)
    dataset = M2TRDataset(args.input_dir, args.batch_size_detect, args.faces_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_preds = list()
    for images in tqdm(loader, desc='M2TR'):
        images['img'] = images['img'].to(device)
        with torch.no_grad():
            logits = model(images)['logits']
            logits = post_func(logits)
            all_preds.extend(logits[:, 1].cpu().tolist())

    df = pd.DataFrame({'File': dataset.imgs, 'Pred': all_preds})
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_csv", type=str, required=True, help="csv to store results")
    parser.add_argument("--faces_dir", type=str, default=None, help="Directory containing extracted faces")

    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--batch_size_detect", type=int, default=32, help="batch size for retinaface")
    parser.add_argument("--num_workers", type=int, default=8, help="num of workers for dataloader")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")

    args = parser.parse_args()
    main(args)
