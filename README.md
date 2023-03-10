## Setup

Notice that working on Nvidia 3090 requires certain pytorch and torchvision versions

```bash
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Blender

```bash
/snap/bin/blender ~/Downloads/rp_aliyah_4d_004_dancing_BLD/rp_aliyah_4d_004_dancing_2k.blend --background --python 'render/renderpeople.py' -- --with_images --start 1 --end 150
```

### Format the camera parameters
```bash
python3 format_convert/json2yaml.py ~/Documents/datasets
```
Then add `%YAML:1.0` as header for the newly generated yml file so that OpenCV can parse it

```vim
:%s/-\ \ //g
:%s/\ \ \ \ \ \ /\ \ \ /g
```

### Mocap

Detect hand and face keypoints using OpenPose
```bash
python3 scripts/preprocess/extract_video.py ~/Documents/datasets --openpose ~/Downloads/openpose --handface --ext png --with_img --end 1
```

```bash
python3 apps/demo/mv1p.py ~/Documents/datasets --out ~/Documents/datasets/output/smpl --vis_det --vis_repro --undis --vis_smpl --end 1
```
