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

def nerf_format(intri, extri, path, mode):
    # Data to be written
    # NOTE as posted in https://github.com/yenchenlin/nerf-pytorch/issues/41
    # focal = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
    f_intri = open(join(path, intri + '.json'), 'r')
    jsonData = json.load(f_intri)
    f_intri.close()
    """
    # The matrix looks like
    f_x  s    c_x
    0    f_y  c_y
    0    0    1
    """
    image_width = jsonData[1]['K_1']['data'][2] * 2
    focal = jsonData[1]['K_1']['data'][0]
    camera_angle_x = np.arctan((0.5 * image_width) / focal) / 0.5

    f_extri = open(join(path, extri + '.json'), 'r')
    jsonData = json.load(f_extri)
    f_extri.close()

    dictionary = {
	"camera_angle_x": camera_angle_x,
	"frames": []
    }
    for i in tqdm(range(0, len(jsonData), 3)):
        name_idx = str(i // 3 + 1)
        array4x4 = []
        temp = jsonData[i]['Rot_'+name_idx]['data'][0:3]
        temp.append(jsonData[i+1]['T_'+name_idx]['data'][0])
        array4x4.append(temp)
        temp = jsonData[i]['Rot_'+name_idx]['data'][3:6]
        temp.append(jsonData[i+1]['T_'+name_idx]['data'][1])
        array4x4.append(temp)
        temp = jsonData[i]['Rot_'+name_idx]['data'][6:9]
        temp.append(jsonData[i+1]['T_'+name_idx]['data'][2])
        array4x4.append(temp)
        array4x4.append([0.0, 0.0, 0.0, 1.0])


        # NOTE need to change the world2cv matrix into cv2world matrix for Nerf
        array4x4 = np.linalg.inv(np.array(array4x4)).tolist()


        dictionary["frames"].append({'file_path': './'+str(name_idx)+'/'+'0001', "rotation": 0.012566370614359171, 'transform_matrix': array4x4})

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
     
    # Writing to sample.json
    with open("transforms_train.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--mode', type=str, default='opencv', choices=['opencv', 'others'], help="differs for yml formats")
    args = parser.parse_args()

    path = args.path
    mode = args.mode
    """
    # converting camera extrinsics from json to yml 
    json2yml('extri', path=args.path, mode=args.mode)
    # converting camera intrinsics from json to yml 
    json2yml('intri', path=args.path, mode=args.mode)    
    """
    nerf_format('intri', 'extri', path=args.path, mode=args.mode)
