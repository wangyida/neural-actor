## Setup

Notice that working on Nvidia 3090 requires certain pytorch and torchvision versions

```bash
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Blender

```bash
/snap/bin/blender ~/Downloads/rp_aliyah_4d_004_dancing_BLD/rp_aliyah_4d_004_dancing_2k.blend --background --python 'render/renderpeople.py' -- --with_images --start 1 --end 150
```
