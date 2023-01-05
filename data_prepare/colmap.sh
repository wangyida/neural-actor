colmap feature_extractor --database_path /Users/yidawang/Documents/dataset_static/database.db --image_path /Users/yidawang/Documents/dataset_static/images --SiftExtraction.use_gpu=false
colmap exhaustive_matcher --database_path /Users/yidawang/Documents/dataset_static/database.db --SiftMatching.use_gpu=false
colmap mapper --database_path /Users/yidawang/Documents/dataset_static/database.db --image_path /Users/yidawang/Documents/dataset_static/images --output_path /Users/yidawang/Documents/dataset_static/sparse

colmap image_undistorter --image_path /Users/yidawang/Documents/dataset_static/images --input_path /Users/yidawang/Documents/dataset_static/sparse/0 --output_path /Users/yidawang/Documents/dataset_static/dense --output_type COLMAP --max_image_size 2000
