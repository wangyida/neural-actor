for (( num=5; num<=8; num++ ))
do
  echo `printf %d/%04d $num $num+1`
  # cp `printf ~/Documents/datasets/images/%d/%04d.png $num $num+1` ./data_rp/images/`printf %05d $num-1`.jpg
  # cp `printf ~/Documents/datasets/images/%d/%04d.png 20 $num+1` ./data_rp/images/`printf %05d $num-1`.jpg
  # cp `printf ~/Documents/datasets_static_bkg/images/%d/%04d.png $num 1` ./data_rp/images/`printf %05d $num-1`.jpg
  cp `printf ~/Documents/datasets_real/images/%d/%04d.jpg $num 1` ./data_syn/images/`printf %05d $num-1`.jpg
done
