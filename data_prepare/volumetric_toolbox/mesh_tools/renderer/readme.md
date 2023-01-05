# mesh renderer

## Example build command for Ubuntu

```bash

# assuming you are already in the `mesh_renderer` directory
cd third_party

# install system dependencies
sudo apt install libassimp-dev libopencv-dev

# build pangolin dependency
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
cmake -B build
cmake --build build

# build renderer
cd ../..
cmake -B build
cmake --build build
```
