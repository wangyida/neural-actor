search_dir='/Users/yidawang/Documents/datasets/depth'
for entry in `ls $search_dir`; do
    # for file in $search_dir/$entry/Image*; do mv "$file" "${file#Image}";done;
    for file in $search_dir/$entry/Image*; do mv "$file" "$(echo "$file" | sed s/Image//)";done;
done
