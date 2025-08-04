import cv2
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import sys

sys.path.insert(0, 'SimSwap')
from models.models import create_model
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
import argparse
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import random
from tqdm import tqdm
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.swap_logic import lfw_logic, celeba_hq_logic, fairface_logic, LFW_SIZE, CELEBA_HQ_SIZE, FAIRFACE_SIZE


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class Options:
    def __init__(self):
        self.name = 'people'
        self.Arc_path = 'SimSwap/arcface_model/arcface_checkpoint.tar'
        self.gpu_ids = '0'
        self.isTrain = False
        self.checkpoints_dir = 'SimSwap/checkpoints'
        self.resize_or_crop = 'scale_width'
        self.verbose = False
        self.which_epoch = 'latest'


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def main(args):
    opt = Options()
    assert args.dataset in ['lfw', 'celeba-hq', 'fairface'], "Invalid dataset name"
    seed_everything(args.seed)
    if args.dataset == 'lfw':
        gen = lfw_logic(args.female_names, args.male_names, args.input_dir, args.output_dir)
        num_of_images = LFW_SIZE
    elif args.dataset == 'celeba-hq':
        gen = celeba_hq_logic(args.identity_file, args.input_dir, args.output_dir)
        num_of_images = CELEBA_HQ_SIZE
    elif args.dataset == 'fairface':
        gen = fairface_logic(args.train_csv, args.val_csv, args.input_dir, args.output_dir)
        num_of_images = FAIRFACE_SIZE

    crop_size = args.crop_size
    opt.crop_size = args.crop_size
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    logoclass = None
    model = create_model(opt)
    model.eval()

    spNorm = SpecificNorm()
    app = Face_detect_crop(name='antelope', root='SimSwap/insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('SimSwap/parsing_model/checkpoint', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    errors = 0
    tqdm_bar = tqdm(total=num_of_images)
    while len(os.listdir(args.output_dir)) < num_of_images:
        source_path, target_path, output_path = next(gen)
        try:
            with torch.no_grad():
                img_a_whole = cv2.imread(source_path)
                img_a_align_crop, _ = app.get(img_a_whole, crop_size)
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
                img_a = transformer_Arcface(img_a_align_crop_pil)
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

                # convert numpy to tensor
                img_id = img_id.cuda()

                # create latent id
                img_id_downsample = F.interpolate(img_id, size=(112, 112))
                latend_id = model.netArc(img_id_downsample)
                latend_id = F.normalize(latend_id, p=2, dim=1)

                ############## Forward Pass ######################
                img_b_whole = cv2.imread(target_path)

                img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
                # detect_results = None
                swap_result_list = []

                b_align_crop_tenor_list = []

                for b_align_crop in img_b_align_crop_list:
                    b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].cuda()

                    swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
                    swap_result_list.append(swap_result)
                    b_align_crop_tenor_list.append(b_align_crop_tenor)

                reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole,
                                   logoclass, output_path, no_simswaplogo=True, pasring_model=net, use_mask=True,
                                   norm=spNorm)

                tqdm_bar.update(1)
        except Exception as e:
            errors += 1

    tqdm_bar.close()
    print(f'done with {errors} errors')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='lfw', help="name of dataset [lfw, celeba-hq or fairface]")
    parser.add_argument("--crop_size", type=int, default=512, help="resolution for SimSwap 224 or 512")
    parser.add_argument("--female_names", type=str, default="/root/female_names.txt",
                        help="Path to file with LFW female names")
    parser.add_argument("--male_names", type=str, default="/root/male_names.txt",
                        help="Path to file with LFW male names")
    parser.add_argument("--identity_file", type=str, default="/root/roop-main/CelebA_HQ/CelebA-HQ-identity.txt",
                        help="Path to file with CELEBA-HQ identity")
    parser.add_argument("--input_dir", type=str, default="lfw", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="lfw_roop", help="Directory to store output files")
    parser.add_argument("--seed", type=int, default=3407, help='random seed')
    parser.add_argument("--train_csv", type=str, default="fairface_label_train.csv",
                        help="FairFace train csv with labels")
    parser.add_argument("--val_csv", type=str, default="fairface_label_val.csv",
                        help="FairFace val csv with labels")

    args = parser.parse_args()

    main(args)
