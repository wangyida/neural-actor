search_dir="$HOME/Documents/datasets_synthesia_$1/masks"
for entry in `ls $search_dir`; do
    for file in $search_dir/$entry/$entry_mask00*; do mv "$file" "$(echo "$file" | sed s/${entry}_mask00//)"; done;
    # for file in $search_dir/$entry/$entry_Cam00*; do mv "$file" "$(echo "$file" | sed s/${entry}_00//)"; done;
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/Cam//)"
done
for entry in `ls $search_dir`; do
    mv "$search_dir/$entry" "$search_dir/$(echo "$entry" | sed -e 's:^0*::')"
done


search_dir="$HOME/Documents/datasets_synthesia_$1/images"
for entry in `ls $search_dir`; do
    for file in $search_dir/$entry/$entry_rgb00*; do mv "$file" "$(echo "$file" | sed s/${entry}_rgb00//)"; done;
    # for file in $search_dir/$entry/$entry_Cam00*; do mv "$file" "$(echo "$file" | sed s/${entry}_00//)"; done;
    mv "$search_dir/$entry" "$(echo "$search_dir/$entry" | sed s/Cam//)"
done
for entry in `ls $search_dir`; do
    mv "$search_dir/$entry" "$search_dir/$(echo "$entry" | sed -e 's:^0*::')"
done
