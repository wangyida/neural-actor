for file in tmp_depth/*; do
  convert $file -define png:color-type=2 $file
done
