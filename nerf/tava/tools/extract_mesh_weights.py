import torch
import os
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tava.utils.training import resume_from_ckpt
from tava.models.basic.mipnerf import cylinder_to_gaussian

device = "cuda:0"

# args for data
ARGS_ZJU_5 = [
    "dataset=zju", "dataset.subject_id=5", "dataset.root_fp=/home/ubuntu/Documents/gitfarm/research-interns-2022/nerf/tava/data/zju/",
    "hydra.run.dir=/home/ubuntu/Documents/gitfarm/research-interns-2022/nerf/tava/outputs/dynamic_mipnerf/zju/5/snarf/cfg\=dataset.subject_id\=5\,dataset.training_view\=10\,loss_bone_offset_mult\=0.1\,loss_bone_w_mult\=1.0/",
]

# args for method
ARGS_TAVA_ZJU = ["pos_enc=snarf", "loss_bone_w_mult=1.0", "pos_enc.offset_net_enabled=true", "model.shading_mode=implicit_AO"]

# here we set the arguments for ZJU_313 as an example.
overrides = ["resume=true"] + ARGS_ZJU_5 + ARGS_TAVA_ZJU

# create the cfg
with initialize(config_path="../configs"):
    cfg = compose(config_name="mipnerf_dyn", overrides=overrides, return_hydra_config=True)
    OmegaConf.resolve(cfg.hydra)
    save_dir = cfg.hydra.run.dir
    ckpt_dir = os.path.join(save_dir, "checkpoints")

# initialize model and load ckpt
model = instantiate(cfg.model).to(device)
_ = resume_from_ckpt(
    path=ckpt_dir, model=model, step=cfg.resume_step, strict=True,
)
print(ckpt_dir)
assert os.path.exists(ckpt_dir), "ckpt does not exist! Please check."

# initialize dataset
dataset = instantiate(
    cfg.dataset, split="train", num_rays=None, cache_n_repeat=None,
)
meta_data = dataset.build_pose_meta_info()

# create a color pattern for visualization
torch.manual_seed(412)
colorbases = torch.rand((cfg.dataset.n_transforms + 1, 3)) * 255

# As the Mip-NeRF requires a covariance for density & color querying,
# we here *estimate* a `cov` based on the size of the subject and the
# number of sampled points. It can be estimated in other ways such as
# average of cov during training.

radii = dataset[0]["rays"].radii.mean().to(device)
if isinstance(cfg.dataset.subject_id, str):  # animal
    rest_verts = torch.from_numpy(
        dataset.parser.load_meta_data(dataset.parser.actions[0])["rest_verts"]
    ).to(device)
else:
    rest_verts = torch.from_numpy(
        dataset.parser.load_meta_data()["rest_verts"]
    ).to(device)
bboxs_min = rest_verts.min(dim=0).values
bboxs_max = rest_verts.max(dim=0).values
subject_size = torch.prod(bboxs_max - bboxs_min) ** (1./3.)
t0, t1 = 0, subject_size / model.num_samples
d = torch.tensor([0., 0., 1.]).to(device)
_, cov = cylinder_to_gaussian(d, t0, t1, radii, model.diag)

# Create a grid coordinates to be queried in canonical.

def create_grid3D(min, max, steps, device="cpu"):
    if type(min) is int:
        min = (min, min, min) # (x, y, z)
    if type(max) is int:
        max = (max, max, max) # (x, y)
    if type(steps) is int:
        steps = (steps, steps, steps) # (x, y, z)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).to(device)
    arrangeY = torch.linspace(min[1], max[1], steps[1]).to(device)
    arrangeZ = torch.linspace(min[2], max[2], steps[2]).to(device)
    gridX, girdY, gridZ = torch.meshgrid([arrangeX, arrangeY, arrangeZ], indexing="ij")
    coords = torch.stack([gridX, girdY, gridZ]) # [3, steps[0], steps[1], steps[2]]
    coords = coords.view(3, -1).t() # [N, 3]
    return coords
_center = (bboxs_max + bboxs_min) / 2.0
_scale = (bboxs_max - bboxs_min) / 2.0
bboxs_min_large = _center - _scale * 1.5
bboxs_max_large = _center + _scale * 1.5
res = 512
coords = create_grid3D(bboxs_min_large, bboxs_max_large, res)

# Query the density grid in the canonical

from tqdm import tqdm
from tava.utils.bone import closest_distance_to_points
from tava.utils.structures import namedtuple_map

bones_rest = namedtuple_map(lambda x: x.to(device), meta_data["bones_rest"])
chunk_size = 20480
densities = []
with torch.no_grad():
    for i in tqdm(range(0, coords.shape[0], chunk_size)):
        coords_chunk = coords[i: i + chunk_size].to(device)

        dists = closest_distance_to_points(bones_rest, coords_chunk).min(dim=-1).values
        selector = dists <= cfg.dataset.cano_dist

        posi_enc = model.pos_enc.posi_enc((coords_chunk, cov.expand_as(coords_chunk)))
        sigma = model.mlp.query_sigma(posi_enc, masks=selector, return_feat=False)
        density = model.density_activation(sigma + model.density_bias)
        densities.append(density.cpu())
densities = torch.cat(densities, dim=0).reshape(res, res, res)
print (densities.min(), densities.max(), densities.median())

# Marching cube to get the mesh. We use the threshold 5.0 for all cases. You might want
# to adjust that with your own data. Note installing torchmcubes would take some time.

# excute this in juputer to install: "!pip install git+https://github.com/tatsy/torchmcubes.git"
from torchmcubes import marching_cubes

verts, faces = marching_cubes(densities, 5.0)
verts = verts[..., [2, 1, 0]] / res * (bboxs_max_large - bboxs_min_large).cpu() + bboxs_min_large.cpu()
print (verts.shape)

# Export Mesh with Skinning Weights

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

chunk_size = 8192
weights = []
with torch.no_grad():
    for i in range(0, verts.shape[0], chunk_size):
        verts_chunk = verts[i: i + chunk_size].to(device)
        weights_chunk = model.pos_enc.query_weights(verts_chunk)
        weights.append(weights_chunk.cpu())
weights = torch.cat(weights, dim=0)

colors = (weights[..., None] * colorbases).sum(dim=-2)
save_obj_mesh_with_color(
    os.path.join(save_dir, "extracted_cano_mesh_soft_weight.obj"),
    verts.cpu().numpy(), faces.cpu().numpy(), colors.cpu().numpy() / 255.0
)

colors = colorbases[weights.argmax(dim=-1)]
save_obj_mesh_with_color(
    os.path.join(save_dir, "extracted_cano_mesh_hard_weight.obj"),
    verts.cpu().numpy(), faces.cpu().numpy(), colors.cpu().numpy() / 255.0
)

