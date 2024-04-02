import os

from easydict import EasyDict as edict

config = edict()

base_dir = 'laplacianet'

config.model = edict()
config.model.ckp_path_high = os.path.join(base_dir, 'checkpoint/highlayer/')
config.model.ckp_path_bot = os.path.join(base_dir, 'checkpoint/botlayer/')
config.model.ckp_path_ft = os.path.join(base_dir, 'checkpoint/ftlayer/')
config.model.ckp_path_demo = os.path.join(base_dir, 'checkpoint/demo/')

config.model.tfrecord = os.path.join(base_dir, 'dataset/tfrecord/')
config.model.tfrecord_dual = os.path.join(base_dir, 'dataset/tfrecord/dual_freq_')
config.model.tfrecord_ft = os.path.join(base_dir, 'dataset/tfrecord/ft_freq_')

config.model.ckp_lev_scale = 'lev_scale_'
config.model.tfrecord_suffix = '.tfrecord'

config.model.loss_model = 'vgg_16'
config.model.loss_vgg = os.path.join(base_dir, 'loss/pretrained/vgg16.npy')


config.data = edict()
config.data.height = 3301
config.data.width = 7768
config.data.patch_size = 512
config.data.patch_size_ft = 256
config.data.random_patch_ratio_x = 0.2
config.data.random_patch_ratio_y = 0.6
config.data.random_patch_per_img = 20
config.data.hdr_path = os.path.join(base_dir, 'dataset/train/hdr/')
config.data.ldr_path = os.path.join(base_dir, 'dataset/train/ldr/')


config.train = edict()
config.train.total_imgs = 1700 * 20
config.train.batch_size_high = 8
config.train.batch_size_bot = 32
config.train.batch_size_ft = 4
config.train.batchnum_high = round(config.train.total_imgs/config.train.batch_size_high)
config.train.batchnum_bot = round(config.train.total_imgs/config.train.batch_size_bot)
config.train.batchnum_ft = round(config.train.total_imgs/config.train.batch_size_ft)


config.eval = edict()
config.eval.tfrecord_eval = os.path.join(base_dir, 'dataset/tfrecord/eval_')
config.eval.tfrecord_demo = os.path.join(base_dir, 'dataset/tfrecord/demo_')
config.eval.result = os.path.join(base_dir, 'dataset/result/')
config.eval.hdr_path = os.path.join(base_dir, 'dataset/test/hdr/')
config.eval.demo_path = os.path.join(base_dir, 'dataset/demo/')
config.eval.loss_vgg = os.path.join(base_dir, 'loss/pretrained/vgg16.npy')



