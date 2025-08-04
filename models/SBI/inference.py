import torch
import os
import sys
from SelfBlendedImages.inference.model import Detector
import argparse
from tqdm import tqdm
import cv2
from retinaface.pre_trained_models import get_model
from SelfBlendedImages.inference.preprocess import extract_face
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import seed_everything


def pred(input_image, face_detector, model, device):
    frame = cv2.imread(input_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_list = extract_face(frame, face_detector)

    with torch.no_grad():
        img = torch.tensor(face_list).to(device).float()/255
        pred = model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    return max(pred)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load('SelfBlendedImages/weights/FFc23.tar')["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=args.max_size_image,
                              device=device)
    face_detector.eval()

    results = {}
    all_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir)]
    tqdm_bar = tqdm(total=len(all_files))
    for file in all_files:
        if not (file.endswith('.jpg') or file.endswith('.png')):
            continue
        filename = os.path.basename(file)
        try:
            results[filename] = pred(file, face_detector, model, device)
        except Exception as e:
            results[filename] = 0.5
        tqdm_bar.update(1)

    df = pd.DataFrame(list(results.items()), columns=['File', 'Pred'])
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_csv", type=str, required=True, help="csv to store results")
    parser.add_argument("--max_size_image", type=int, default=1024, help="max size of image for face detector")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")

    args = parser.parse_args()
    main(args)
