CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=narf resume=true dataset.training_view=10 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=narf resume=true dataset.training_view=20 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=narf resume=true dataset.training_view=40 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=narf resume=true dataset.training_view=80 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=narf resume=true dataset.training_view=130 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 resume=true dataset.training_view=10 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 resume=true dataset.training_view=20 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 resume=true dataset.training_view=40 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 resume=true dataset.training_view=80 engine=evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 launch.py --config-name=mipnerf_dyn \
    dataset=zju \
    dataset.subject_id=2 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 resume=true dataset.training_view=130 engine=evaluator
