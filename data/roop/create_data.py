import os
import sys
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
from typing import List
import platform
import shutil
import argparse
import onnxruntime
import tensorflow

sys.path.insert(0, 'roop')
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import clean_temp
from tqdm import tqdm
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.swap_logic import lfw_logic, celeba_hq_logic, fairface_logic, LFW_SIZE, CELEBA_HQ_SIZE, FAIRFACE_SIZE

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def seed_tf(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tensorflow.random.set_seed(seed)


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        # update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        # update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def main(args):
    assert args.dataset in ['lfw', 'celeba-hq', 'fairface'], "Invalid dataset name"
    seed_tf(args.seed)
    if args.dataset == 'lfw':
        gen = lfw_logic(args.female_names, args.male_names, args.input_dir, args.output_dir)
        num_of_images = LFW_SIZE
    elif args.dataset == 'celeba-hq':
        gen = celeba_hq_logic(args.identity_file, args.input_dir, args.output_dir)
        num_of_images = CELEBA_HQ_SIZE
    elif args.dataset == 'fairface':
        gen = fairface_logic(args.train_csv, args.val_csv, args.input_dir, args.output_dir)
        num_of_images = FAIRFACE_SIZE

    roop.globals.execution_providers = [args.device]
    roop.globals.frame_processors = ['face_swapper']
    roop.globals.many_faces = False
    roop.globals.headless = True
    roop.globals.reference_face_position = 0
    roop.globals.similar_face_distance = 0.85
    roop.globals.max_memory = None
    roop.globals.execution_threads = suggest_execution_threads()

    limit_resources()

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            print('smth went wrong')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    errors = 0
    tqdm_bar = tqdm(total=num_of_images)
    while len(os.listdir(args.output_dir)) < num_of_images:
        source_path, target_path, output_path = next(gen)
        roop.globals.source_path = source_path
        roop.globals.target_path = target_path
        roop.globals.output_path = output_path
        try:
            shutil.copy2(roop.globals.target_path, roop.globals.output_path)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            tqdm_bar.update(1)
        except Exception as e:
            # print(f"Error: {e}")
            # print(source_path, target_path, output_path)
            os.remove(roop.globals.output_path)
            errors += 1

    tqdm_bar.close()
    print(f'done with {errors} errors')

    frame_processor.post_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='lfw', help="name of dataset [lfw, celeba-hq or fairface]")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cpu', 'cuda')")
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
