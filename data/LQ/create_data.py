from PIL import Image
import argparse
import os
from tqdm import tqdm


def downscale_image(input_image_path, output_image_path, size, quality):
    """
    :param input_image_path:
    :param output_image_path:
    :param size: resolution of output image. BICUBIC algorithm
    :param quality: The image quality, on a scale from 0 (worst) to 95 (best), or the string keep
    :return: None. saving image
    """
    with Image.open(input_image_path) as img:
        if size is not None:
            img.thumbnail(size, resample=Image.Resampling.BICUBIC)
        img.save(output_image_path, "JPEG", quality=quality)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    all_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir)]
    tqdm_bar = tqdm(total=len(all_files))
    size = args.size
    quality = args.quality
    for file in all_files:
        if not (file.endswith('.jpg') or file.endswith('.png')):
            continue
        filename = os.path.basename(file)
        out_filename = os.path.join(args.output_dir, f"{filename[:-4]}.jpg")
        downscale_image(file, out_filename, size, quality)
        tqdm_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output files")
    parser.add_argument("--size", nargs='+', type=int, default=None, help="Resolution of output. If none - perform wo"
                                                                          "resizing")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")

    args = parser.parse_args()
    main(args)
