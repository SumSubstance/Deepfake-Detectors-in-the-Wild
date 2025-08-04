import os
from os.path import join
import numpy as np
import cv2
import torch
from tqdm import tqdm

from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode

np.random.seed(0)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu, device):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(device)
    return model


def detect(img_list, output_dir, base_names, args, net, cfg, device, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    im_height, im_width, _ = img_list[0].shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img_x = torch.stack(img_list, dim=0).permute([0, 3, 1, 2])
    scale = scale.to(device)

    # forward times
    f_times = img_x.shape[0] // batch_size
    if img_x.shape[0] % batch_size != 0:
        f_times += 1
    locs_list = list()
    confs_list = list()
    for _ in range(f_times):
        if _ != f_times - 1:
            batch_img_x = img_x[_ * batch_size:(_ + 1) * batch_size]
        else:
            batch_img_x = img_x[_ * batch_size:]  # last batch
        batch_img_x = batch_img_x.to(device).float()
        l, c, _ = net(batch_img_x)
        locs_list.append(l)
        confs_list.append(c)
    locs = torch.cat(locs_list, dim=0)
    confs = torch.cat(confs_list, dim=0)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    img_cpu = img_x.permute([0, 2, 3, 1]).cpu().numpy()
    i = 0
    for img, loc, conf, name in zip(img_cpu, locs, confs, base_names):
        boxes = decode(loc.data, prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        if len(dets) == 0:
            cv2.imwrite(join(output_dir, name), img)
            print(name)
            continue
        det = list(map(int, dets[0]))
        x, y, size_bb_x, size_bb_y = get_boundingbox(det, img.shape[1], img.shape[0])
        cropped_img = img[y:y + size_bb_y, x:x + size_bb_x, :] + (104, 117, 123)

        cv2.imwrite(join(output_dir, name), cropped_img)
        i += 1
    pass


def get_boundingbox(bbox, width, height, scale=1.3, minsize=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    size_bb_x = int((x2 - x1) * scale)
    size_bb_y = int((y2 - y1) * scale)
    if minsize:
        if size_bb_x < minsize:
            size_bb_x = minsize
        if size_bb_y < minsize:
            size_bb_y = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb_x // 2), 0)
    y1 = max(int(center_y - size_bb_y // 2), 0)
    # Check for too big bb size for given x, y
    size_bb_x = min(width - x1, size_bb_x)
    size_bb_y = min(height - y1, size_bb_y)
    return x1, y1, size_bb_x, size_bb_y


def extract_faces(input_dir, output_dir, batch_size, pretrained_weights=None):
    class Args:
        keep_top_k = 1
        top_k = 5
        nms_threshold = 0.4
        confidence_threshold = 0.05

    args = Args()
    torch.set_grad_enabled(False)
    cfg = cfg_re50
    pretrained_weights = pretrained_weights if pretrained_weights is not None\
        else 'RECCE/Pytorch_Retinaface/weights/Resnet50_Final.pth'

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, pretrained_weights, "cuda:0", "cuda:0")
    net.eval()

    strange_tensor = torch.tensor([104, 117, 123])

    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)
                 if (file.endswith(".jpg") or file.endswith(".png"))]
    for i in tqdm(range(0, len(all_files), batch_size), desc='Face Detection'):
        files = all_files[i:i+batch_size]
        now_images = []
        for file in files:
            image = cv2.imread(file)
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            now_images.append(torch.tensor(image) - strange_tensor)
        base_names = [os.path.basename(file) for file in files]
        detect(now_images, output_dir, base_names, args, net, cfg, device, batch_size)
