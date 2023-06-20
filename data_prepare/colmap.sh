DATASET_PATH=/home/yidaw/Documents/datasets/$1

# colmap feature_extractor --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --mask_path $DATASET_PATH/masks # --SiftExtraction.use_gpu=false

# colmap exhaustive_matcher --database_path $DATASET_PATH/database.db # --SiftMatching.use_gpu=false

# mkdir $DATASET_PATH/sparse
# colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse

mkdir $DATASET_PATH/undistort
colmap image_undistorter --image_path $DATASET_PATH/images --input_path $DATASET_PATH/sparse/0 --output_path $DATASET_PATH/undistort --output_type COLMAP --max_image_size 5472
