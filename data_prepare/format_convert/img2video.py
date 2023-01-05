import os
import moviepy.video.io.ImageSequenceClip


def img2vid(image_folder, ext='png'):
    fps = 24
    lst = os.listdir(image_folder)
    image_files = [
        os.path.join(image_folder, img) for img in sorted(lst)
        if img.endswith(ext)
    ]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files,
                                                                fps=fps)
    clip.write_videofile('./video.mp4')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--ext',
                        type=str,
                        default='jpg',
                        choices=['jpg', 'png'],
                        help="image file extension")
    args = parser.parse_args()
    img2vid(image_folder=args.path, ext=args.ext)
