import cv2
import time
import os

from tqdm import tqdm


def bilinear(img, ratio):
    print('Start bilinear interpolation')
    h, w = img.shape[:2]
    dst = cv2.resize(img, None, fx=ratio, fy=ratio,
                     interpolation=cv2.INTER_LINEAR)
    return dst


if __name__ == "__main__":
    folder_path = 'Final_project/data/DIV2K_valid_HR/'
    dst_folder_path = 'Final_project/data/DIV2K_valid_LR_bilinear/X4/'
    images_names = os.listdir(folder_path)

    ratio = 1/4     # scale factor

    start = time.time()

    for img_name in tqdm(images_names):
        path = folder_path + img_name

        # Read image
        img = cv2.imread(path)
        dst = bilinear(img, ratio)

        dst_path = dst_folder_path + img_name.replace(".png", "x4.png")
        cv2.imwrite(dst_path, dst)

    end = time.time()
    tot = int(end - start)
    print("Completed in {} seconds".format(tot))
