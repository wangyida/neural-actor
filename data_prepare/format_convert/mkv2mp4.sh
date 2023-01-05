for i in *.mkv; do
    ffmpeg -i "$i" -codec copy "${i%.*}.mp4"
done
