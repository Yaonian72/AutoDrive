import numpy as np
import argparse
import os
import pickle


parser = argparse.ArgumentParser(description="convert mmdetection3d output to visualization kitti format")
parser.add_argument(
    "--split_file",
    type=str,
    required=True,
    help="split sample index file"
)

parser.add_argument(
    "--pkl_file",
    type=str,
    required=True,
    help="mmdetection3d output file"
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="prediction output directory"
)

args = parser.parse_args()

split_file = args.split_file
out_file = args.pkl_file
pred_dir = args.out_dir

with open(split_file, 'r') as f:
    split = f.readlines()
split = [name.strip() for name in split]

with open(out_file, 'rb') as f:
    out_dict = pickle.load(f)

for pred, filename in zip(out_dict, split):
    pred["dimensions"][:, [0, 2]] = pred["dimensions"][:, [2, 0]]# swap h and l
    with open(os.path.join(pred_dir, filename+'.txt'), 'w') as f:
        for i in range(len(pred['name'])):
            obj = []
            for k, v in pred.items():
                if not isinstance(v[i], np.ndarray):
                    obj.append(v[i])
                else:
                    obj.extend(list(v[i]))
            obj = [str(field) for field in obj]
            f.write(' '.join(obj)+'\n')