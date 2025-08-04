import torch
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="pickle2bin utility")
    parser.add_argument("--path", "-p",
                        type=str,
                        required=True,
                        help="Specify the path of the pickle file to "
                             "translate into bin file.")
    return parser.parse_args()


if __name__ == '__main__':
    arg = arg_parser()

    print(f"Loading pickle file from '{arg.path}'.")
    model_params = torch.load(arg.path, map_location="cpu")

    torch.save({
        "model": model_params,
        "step": -1,
        "best_step": -1,
        "best_metric": torch.tensor(-1.),
        "eval_metric": "Unknown"
    }, arg.path.replace(".pickle", ".bin"))
    print(f"Converted pickle file to bin file stored in "
          f"'{arg.path.replace('.pickle', '.bin')}'.")

