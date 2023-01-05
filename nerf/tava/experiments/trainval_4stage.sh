CUDA_VISIBLE_DEVICES=$3 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=$1 \
    pos_enc=narf resume=true dataset.training_view=$2 max_steps=20_000 eval_every=20_000 eval_cache_dir=eval_imgs_otf_20000 save_every=20000

CUDA_VISIBLE_DEVICES=$3 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=$1 \
    pos_enc=narf resume=true dataset.training_view=$2 max_steps=40_000 eval_every=40_000 eval_cache_dir=eval_imgs_otf_40000 save_every=40000

CUDA_VISIBLE_DEVICES=$3 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=$1 \
    pos_enc=narf resume=true dataset.training_view=$2 max_steps=60_000 eval_every=60_000 eval_cache_dir=eval_imgs_otf_60000 save_every=60000

CUDA_VISIBLE_DEVICES=$3 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=$1 \
    pos_enc=narf resume=true dataset.training_view=$2 max_steps=120_000 eval_every=120_000 eval_cache_dir=eval_imgs_otf_120000 save_every=120000
