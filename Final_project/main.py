import matplotlib.pyplot as plt
import tensorflow as tf

from learning import train, evaluate
from utils import show_images, get_config


def reduce_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":

    # reduce_gpu()
    config = get_config()

    # training
    train(g_pretrained=True, n_trainable=1, generic=False, config=config)

    # # evaluation
    # im_sr, im_hr = evaluate()

    # show_images(im_sr, im_hr)
