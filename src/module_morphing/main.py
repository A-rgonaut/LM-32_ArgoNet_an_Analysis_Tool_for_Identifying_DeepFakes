
import cv2
import argparse
import morpher as mp

# python main.py img/coppia1_1.jpg img/coppia1_2.jpg --t 0.5 --warp_t 0.1 --swap --video
# python main.py img/coppia2_1.jpg img/coppia2_2.jpg --t 0.5 --warp_t 0.5 --swap --video
# python main.py img/coppia3_1.jpg img/coppia3_2.jpg --t 0.5 --warp_t 0.9 --swap --video

def main():
    parser = argparse.ArgumentParser(description="Image Morphing Tool")
    parser.add_argument("src_path", type=str, help="Path to the source image")
    parser.add_argument("dst_path", type=str, help="Path to the destination image")
    parser.add_argument("--t", type=float, default=0.5, help="Morphing factor (default: 0.5)")
    parser.add_argument("--warp_t", type=float, default=0.5, help="Warping factor (default: 0.5)")
    parser.add_argument("--swap", action="store_true", help="Swap the source with the destination")
    parser.add_argument("--video", action="store_true", help="Enable video mode for morphing")
    args = parser.parse_args()

    src = cv2.imread(args.src_path)
    dst = cv2.imread(args.dst_path)

    if src is None or dst is None:
        print("Error: One or both image paths are invalid.")
        return

    row, col, _ = src.shape
    src = cv2.resize(src, (col//2, row//2))
    dst = cv2.resize(dst, (col//2, row//2))

    morpher = mp.Morpher(src=src, dst=dst, t=args.t)

    morpher.swap_src_dst() if args.swap else None

    morphed_image = morpher.morph(warp_t=args.warp_t, video=args.video)

    cv2.imshow("Morphed Image", morphed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    path1 = "" + args.src_path
    path2 = "" + args.dst_path

    path1 = path1.split("/")[-1]
    path2 = path2.split("/")[-1]

    path1 = path1.split(".jpg")[0]
    path2 = path2.split(".jpg")[0]

    cv2.imwrite("./res/" + path1 + "_" + path2 + "_warp_" + str(args.warp_t) + "_t_" + str(args.t) + ".jpg", morphed_image)

if __name__ == "__main__":
    main()