colmap feature_extractor \
    --database_path ~/Documents/datasets/tower_cam_colmap/database.db \
    --image_path ~/Documents/datasets/tower_cam_colmap/images

colmap exhaustive_matcher \
    --database_path ~/Documents/datasets/tower_cam_colmap/database.db # --SiftMatching.use_gpu=false

colmap mapper --database_path ~/Documents/datasets/tower_cam_colmap/database.db --image_path ~/Documents/datasets/tower_cam_colmap/images --output_path ~/Documents/datasets/tower_cam_colmap/sparse
