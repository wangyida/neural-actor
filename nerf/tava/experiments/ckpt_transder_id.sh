# the first model
mipnerf_path="./outputs/dynamic_mipnerf/zju"
cp -r "$mipnerf_path/$1" "$mipnerf_path/$2"
search_dir="$HOME/Documents/gitfarm/research-interns-2022/nerf/tava/outputs/dynamic_mipnerf/zju/$2/snarf"
for entry in `ls $search_dir`; do
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/subject_id=$1/subject_id=$2/)"
done

# other two models
nerf_path="./outputs/dynamic_nerf/zju"
cp -r "$nerf_path/$1" "$nerf_path/$2"
search_dir="$HOME/Documents/gitfarm/research-interns-2022/nerf/tava/outputs/dynamic_mipnerf/zju/$2/narf"
for entry in `ls $search_dir`; do
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/subject_id=$1/subject_id=$2/)"
done
search_dir="$HOME/Documents/gitfarm/research-interns-2022/nerf/tava/outputs/dynamic_nerf/zju/$2/snarf"
for entry in `ls $search_dir`; do
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/subject_id=$1/subject_id=$2/)"
done
