import tensorlayer as tl
import tensorflow as tf
import time
import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

from Final_project.utils import *


def train(g_pretrained=False, n_trainable=None, generic=True, land_class=None, config=None):

    if generic == False and land_class == None:
        raise ValueError(
            "If you are training a Specialized-SRGAN, you have to select a landscape class among [2,...,7].")

    if generic and land_class is not None:
        print("WARNING: if you set generic=True, the land_class variable will not be used.")

    total_time = time.time()
    if config == None:
        config = get_config()

    ### HYPER-PARAMETERS ###
    batch_size = config.TRAIN.batch_size
    lr_init = config.TRAIN.lr_init
    beta1 = config.TRAIN.beta1
    # initialize G
    n_epoch_init = config.TRAIN.n_epoch_init
    # adversarial learning (SRGAN)
    n_epoch = config.TRAIN.n_epoch
    lr_decay = config.TRAIN.lr_decay
    decay_every = config.TRAIN.decay_every
    n_images = config.TRAIN.n_images

    # create folders to save result images and trained models
    save_dir = config.save_dir
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = config.checkpoint_dir
    tl.files.exists_or_mkdir(checkpoint_dir)

    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tl.models.vgg16(pretrained=True, end_with='pool4', mode='static')

    if generic:
        g_name = 'g'
        d_name = 'd'
    else:
        g_name = 'g_spec_' + str(land_class)
        d_name = 'd_spec_' + str(land_class)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data(config, generic, land_class)

    trainable_weights = G.trainable_weights

    if g_pretrained and n_trainable:
        nt = -n_trainable-1
        trainable_weights = G.all_weights[:nt]
        G.load_weights(os.path.join(checkpoint_dir, 'g_srgan.npz'))
    else:
        nt = len(G.all_weights)

    g_init_losses = []
    # initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in train_ds.enumerate():
            # if the remaining data in this epoch < batch_size
            if lr_patchs.shape[0] != batch_size:
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                G.all_weights[:nt] = trainable_weights
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(
                    fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(
                save_dir, 'train_g_init_{}.png'.format(epoch)))
    # aggiunto Ale
    tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(
        save_dir, 'train_g_init_final.png'))

    # adversarial learning (G, D)
    n_step_epoch = round(n_images // batch_size)  # era n_epoch //
    g_losses = []
    d_losses = []
    for epoch in range(n_epoch):
        g_losses_epoch = []
        d_losses_epoch = []
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            # if the remaining data in this epoch < batch_size
            if lr_patchs.shape[0] != batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                # the pre-trained VGG uses the input range of [0, 1]
                feature_fake = VGG((fake_patchs+1)/2.)
                feature_real = VGG((hr_patchs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(
                    logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(
                    logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2  # discriminator loss
                g_gan_loss = 1e-3 * \
                    tl.cost.sigmoid_cross_entropy(
                        logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(
                    fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * \
                    tl.cost.mean_squared_error(
                        feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss  # generator loss
            grad = tape.gradient(g_loss, trainable_weights)
            g_optimizer.apply_gradients(zip(grad, trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))
            g_losses_epoch.append(g_loss.numpy())
            d_losses_epoch.append(d_loss.numpy())
            G.all_weights[:nt] = trainable_weights
        g_losses.append(g_losses_epoch)
        d_losses.append(d_losses_epoch)
        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (
                lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 5 == 0):  # era epoch%10
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(
                save_dir, 'train_{}_{}.png'.format(g_name, epoch)))
            G.save_weights(os.path.join(
                checkpoint_dir, '{}.h5'.format(g_name)))
            D.save_weights(os.path.join(
                checkpoint_dir, '{}.h5'.format(d_name)))
    # aggiunto Ale
    tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(
        save_dir, 'train_{}_final.png'.format(g_name)))
    G.save_weights(os.path.join(checkpoint_dir, '{}.h5'.format(g_name)))
    D.save_weights(os.path.join(checkpoint_dir, '{}.h5'.format(d_name)))

    if not g_pretrained:
        pd.DataFrame(g_init_losses).to_csv(os.path.join(
            checkpoint_dir, g_name + '_init_loss.csv'))
    pd.DataFrame(g_losses).to_csv(os.path.join(
        checkpoint_dir, g_name + '_losses.csv'))
    pd.DataFrame(d_losses).to_csv(os.path.join(
        checkpoint_dir, d_name + '_losses.csv'))
    print('TOTAL_TIME: ', round((time.time()-total_time)/60, 2), 'min')


def evaluate(imid=None, valid_lr_img=None, valid_hr_img=None,
             landscapes=False, generic=True, land_class=None):

    if generic == False and land_class == None:
        raise ValueError(
            "If you are evaluating a Specialized-SRGAN, you have to select a landscape class among [2,...,7].")

    if generic and land_class is not None:
        print("WARNING: if you set generic=True, the land_class variable will not be used.")

    config = get_config()

    if imid == None:
        imid = 30

    if generic:
        g_name = 'g'
    else:
        g_name = 'g_spec_' + str(land_class)

    if valid_lr_img is None:
        ###====================== PRE-LOAD DATA ===========================###
        if landscapes:
            valid_hr_img_list = sorted(tl.files.load_file_list(
                path=config.VALID.hr_img_path, regx='.*.jpg', printable=False))
            valid_hr_imgs = tl.vis.read_images(
                valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
            valid_hr_img = valid_hr_imgs[imid]
            w, h, c = valid_hr_img.shape
            valid_lr_img = tf.image.resize(
                valid_hr_img, size=[w//4, h//4], method='bicubic')
        else:
            valid_lr_img_list = sorted(tl.files.load_file_list(
                path=config.VALID.lr_img_path, regx='.*.png', printable=False))
            valid_lr_imgs = tl.vis.read_images(
                valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
            valid_lr_img = valid_lr_imgs[imid]
            valid_hr_img = None

        ###========================== DEFINE MODEL ============================###

    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    G = get_G([1, None, None, 3])
    # 'g.h5' 'g_spec.h5 'g_srgan.npz'
    G.load_weights(os.path.join(config.checkpoint_dir, '{}.h5'.format(g_name)))
    G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis, :, :, :]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]
    valid_lr_crop, valid_hr_crop = cropping(valid_hr_img)
    fake_patchs = G(tf.expand_dims(valid_lr_crop, 0))
    mse_loss = tl.cost.mean_squared_error(
        tf.squeeze(fake_patchs), valid_hr_crop, is_mean=True)
    ssim_loss = tf.image.ssim(tf.squeeze(fake_patchs), valid_hr_crop, 126.5)
    out = G(valid_lr_img).numpy()

    # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("LR size: %s /  generated HR size: %s" % (size, out.shape))
    print("[*] save images")
    image_name = "valid_" + g_name + ".png"
    tl.vis.save_image(out[0], os.path.join(config.save_dir, image_name))
    tl.vis.save_image(valid_lr_img[0], os.path.join(
        config.save_dir, 'valid_lr.png'))
    # if valid_hr_img is not None:
    # tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = tf.image.resize(valid_lr_img[0], size=[
                               size[0] * 4, size[1] * 4], method='bicubic')
    # out_bicu = scipy.misc.imresize(
    #     valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    # out_bicu = np.array(Image.fromarray(valid_lr_img[0]).resize([int(size[0] * 4), int(size[1] * 4)],resample=PIL.Image.BICUBIC))
    tl.vis.save_image(out_bicu, os.path.join(
        config.save_dir, 'valid_bicubic.png'))

    return out, valid_hr_img, mse_loss, (1-ssim_loss)


def plot_loss(config, generic=True, land_class=None):
    if generic:
        g_name = 'g_losses.csv'
        d_name = 'd_losses.csv'
    else:
        g_name = 'g_spec_{}_losses.csv'.format(str(land_class))
        d_name = 'd_spec_{}_losses.csv'.format(str(land_class))

    g_losses = pd.read_csv(os.path.join(
        config.checkpoint_dir, g_name))
    l = g_losses.iloc[:, 1:].mean(axis=1)
    plt.plot(range(len(l)), l, )
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Generator (GAN)')
    plt.text(3, np.max(l)-0.001, 'last='+str(round(l.iloc[-1], 5)))
    plt.show()

    d_losses = pd.read_csv(os.path.join(
        config.checkpoint_dir, d_name))
    l = d_losses.iloc[:, 1:].mean(axis=1)
    plt.plot(range(len(l)), l, )
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Discriminator (GAN)')
    plt.text(3, np.max(l)-0.01, 'last='+str(round(l.iloc[-1], 5)))
    plt.show()
