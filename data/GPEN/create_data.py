import os
import cv2
from tqdm import tqdm
import argparse
import torch
import sys
sys.path.insert(0, 'GPEN')
from face_enhancement import FaceEnhancement


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


class Options:
    def __init__(self):
        self.model = 'GPEN-BFR-512'
        self.task = 'FaceEnhancement'
        self.key = None
        self.in_size = 512
        self.out_size = None
        self.channel_multiplier = 2
        self.narrow = 1
        self.alpha = 1
        self.use_sr = 'store_true'
        self.use_cuda = 'store_true'
        self.save_face = 'store_true'
        self.aligned = 'store_true'
        self.sr_model = 'realesrnet'
        self.sr_scale = 2
        self.tile_size = 0
        self.ext = '.jpg'
        self.seed = 3407


def main(args):
    opt = Options()
    seed_everything(opt.seed)
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    all_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir)]
    faceenhancer = FaceEnhancement(opt, in_size=opt.in_size, model=opt.model, use_sr=opt.use_sr,
                                   device='cuda' if opt.use_cuda else 'cpu', base_dir='GPEN')

    errors = 0
    tqdm_bar = tqdm(total=len(all_files))
    for file in all_files:
        if not (file.endswith('.jpg') or file.endswith('.png')):
            continue
        filename = os.path.basename(file)
        img = cv2.imread(file, cv2.IMREAD_COLOR)

        try:
            img_out, orig_faces, enhanced_faces = faceenhancer.process(img, aligned=False)
            cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_GPEN.jpg'), img_out)
            tqdm_bar.update(1)
        except Exception as e:
            errors += 1

    tqdm_bar.close()
    print(f'done with {errors} errors')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output files")

    args = parser.parse_args()
    main(args)
