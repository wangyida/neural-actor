FILES=$(find  -name "*eval*")
# for f in $FILES

for f in $(find outputs/dynamic_mipnerf/zju/ -name "*eval*")
do
  echo "$f"
  ls $f/val_ood/rgb | wc -l
done
