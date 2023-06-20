## Rendering, Animating and Meshing Actors with NeRF
A library for rendering neural actors, and benchmarking dynamic NeRF

### Examplar

#### Point Cloud Completion
![](readmes/na1.gif)

#### Neural Rendering
![](readmes/na2.gif)

#### Dynamic Meshing
![](readmes/na3.gif)

#### SMPL Fitting
![](readmes/na4.gif)

## Setup

Notice that working on NVidia 3090 requires certain pytorch and torchvision versions


```bash
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Or download the wheel and install with in case pytorch site is not included in the pip source
```bash
pip install torch-1.11.0+cu113-cp37-cp37m-linux_x86_64.whl
```

As for *COLMAP*, version 3.8 (the lastest in early 2023) has problen with CUDA 11.X, version 3.7 fits the best for SIFT feature extraction and matching with CUDA.https://github.com/colmap/colmap/tree/3.7

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

### Citation
```
@misc{rama2023wang,
Author = {Yida Wang},
Year = {2023},
Note = {https://github.com/wangyida/neural-actor},
Title = {Rendering, Animating and Meshing Actors with NeRF}
}
```
