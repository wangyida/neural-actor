search_dir='/home/ywang/Documents/datasets_synthesia_5/videos_msk'
for entry in `ls $search_dir`; do
    # for file in $search_dir/$entry/$entry_mask00*; do mv "$file" "$(echo "$file" | sed s/${entry}_mask00//)"; done;
    # for file in $search_dir/$entry/$entry_Cam00*; do mv "$file" "$(echo "$file" | sed s/${entry}_00//)"; done;
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/_mask//)"
done
for entry in `ls $search_dir`; do
    # for file in $search_dir/$entry/$entry_mask00*; do mv "$file" "$(echo "$file" | sed s/${entry}_mask00//)"; done;
    # for file in $search_dir/$entry/$entry_Cam00*; do mv "$file" "$(echo "$file" | sed s/${entry}_00//)"; done;
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/Cam//)"
done
for entry in `ls $search_dir`; do
    mv "$search_dir/$entry" "$search_dir/$(echo "$entry" | sed -e 's:^0*::')"
done
