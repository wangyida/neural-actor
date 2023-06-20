import simplejson as json
import ruamel.yaml
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm

def json2yml(prefix, path, mode):
    f_json = open(join(path, prefix + '.json'), 'r')
    jsonData = json.load(f_json)
    f_json.close()

    f_yml = open(join(path, prefix + '.yml'), 'w+')
    if mode == 'opencv':
        ruamel.yaml.safe_dump(jsonData, f_yml, indent=3)
    else:
        # NOTE: the other yaml format which does not fitting OpenCV's loader
        import yaml
        yaml.dump(jsonData, f_yml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--mode', type=str, default='opencv', choices=['opencv', 'others'], help="differs for yml formats")
    args = parser.parse_args()

    path = args.path
    mode = args.mode
    # converting camera extrinsics from json to yml 
    json2yml('extri', path=args.path, mode=args.mode)
    # converting camera intrinsics from json to yml 
    json2yml('intri', path=args.path, mode=args.mode)    
