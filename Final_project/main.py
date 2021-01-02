import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

from learning import *


def show_images(im_sr, im_hr):
    for i, img in enumerate([im_hr, im_sr]):
        plt.subplot(1, 2, i+1)
        im = tf.squeeze(img)
        im = image.array_to_img(im)
        plt.imshow(im)


if __name__ == "__main__":
    # training
    train(g_pretrained=True, n_trainable=1, generic=False)

    # # evaluation
    # im_sr, im_hr = evaluate()

    # show_images(im_sr, im_hr)
