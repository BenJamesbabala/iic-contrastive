import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as tf
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from torch.autograd import Variable

#Returns a custom greyscale transform
def custom_greyscale_to_tensor(include_rgb):
  """
  Greyscaling with transform to tensor.
  """
  def _inner(img):
    grey_img_tensor = tf.to_tensor(tf.to_grayscale(img, num_output_channels=1))
    result = grey_img_tensor  # 1, 96, 96 in [0, 1]
    assert (result.size(0) == 1)
    if include_rgb:  # greyscale last
      img_tensor = tf.to_tensor(img)
      result = torch.cat([img_tensor, grey_img_tensor], dim=0)
      assert (result.size(0) == 4)
    return result
  return _inner

#Returns a custom cutout transform
def custom_cutout(min_box=None, max_box=None):
  """
  Cuts a random portion of the image.
  """
  def _inner(img):
    w, h = img.size

    # find left, upper, right, lower
    box_sz = np.random.randint(min_box, max_box + 1)
    half_box_sz = int(np.floor(box_sz / 2.))
    x_c = np.random.randint(half_box_sz, w - half_box_sz)
    y_c = np.random.randint(half_box_sz, h - half_box_sz)
    box = (
      x_c - half_box_sz, 
      y_c - half_box_sz, 
      x_c + half_box_sz,
      y_c + half_box_sz
    )
    img.paste(0, box=box)
    return img

  return _inner

#Applies sobel process to a batch of images
def sobel_process(imgs, include_rgb, using_IR=False):
  """
  Applies sobel process to a batch of images.
  """
  bn, c, h, w = imgs.size()
  #IIC includes an option for adding original images together with sobel/greyscale one, using different channels
  if not using_IR:
    if not include_rgb:
      assert (c == 1)
      grey_imgs = imgs
    else:
      assert (c == 4)
      grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
      rgb_imgs = imgs[:, :3, :, :]
  else:
    if not include_rgb:
      assert (c == 2)
      grey_imgs = imgs[:, 0, :, :].unsqueeze(1)  # underneath IR
      ir_imgs = imgs[:, 1, :, :].unsqueeze(1)
    else:
      assert (c == 5)
      rgb_imgs = imgs[:, :3, :, :]
      grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
      ir_imgs = imgs[:, 4, :, :].unsqueeze(1)

  sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  conv1.weight = nn.Parameter(torch.Tensor(sobel1).cuda().float().unsqueeze(0).unsqueeze(0))
  dx = conv1(Variable(grey_imgs)).data

  sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  conv2.weight = nn.Parameter(torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0))
  dy = conv2(Variable(grey_imgs)).data

  sobel_imgs = torch.cat([dx, dy], dim=1)
  assert (sobel_imgs.shape == (bn, 2, h, w))

  if not using_IR:
    if include_rgb:
      sobel_imgs = torch.cat([rgb_imgs, sobel_imgs], dim=1)
      assert (sobel_imgs.shape == (bn, 5, h, w))
  else:
    if include_rgb:
      # stick both rgb and ir back on in right order (sobel sandwiched inside)
      sobel_imgs = torch.cat([rgb_imgs, sobel_imgs, ir_imgs], dim=1)
    else:
      # stick ir back on in right order (on top of sobel)
      sobel_imgs = torch.cat([sobel_imgs, ir_imgs], dim=1)

  return sobel_imgs

#Normalizes the image
def per_img_demean(img):
  """
  Normalizes the image by subtracting the mean.
  """
  assert (len(img.size()) == 3 and img.size(0) == 3)  # 1 RGB image, tensor
  mean = img.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / (img.size(1) * img.size(2))
  return img - mean  # expands

#Produces transforms pools for images with sobel process applied (es. CIFAR10 base case)
def sobel_make_transforms(config, random_affine=False, cutout=False, cutout_p=None, cutout_max_box=None, affine_p=None):  
  """
  Generates augmentations pools to be used by images to which sobel process has been applied.
  """
  tf1_list = [] #transforms applied to original images
  tf2_list = [] #transforms applied to randomly augmented images
  tf3_list = [] #transforms applied at test time

  #crops original images
  if config.crop_orig:
    tf1_list += [
      torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz, config.rand_crop_sz]))),
      torchvision.transforms.Resize(tuple(np.array([config.input_sz, config.input_sz]))),
    ]
    tf3_list += [
      torchvision.transforms.CenterCrop(tuple(np.array([config.rand_crop_sz, config.rand_crop_sz]))),
      torchvision.transforms.Resize(tuple(np.array([config.input_sz, config.input_sz]))),
    ]
  #greyscale is needed for sobel process
  tf1_list.append(custom_greyscale_to_tensor(config.include_rgb))
  tf3_list.append(custom_greyscale_to_tensor(config.include_rgb))

  #random rotation and multiple crop sizes
  if config.fluid_warp:
    tf2_list += [torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

    imgs_tf_crops = []
    for crop_sz in config.rand_crop_szs_tf:
      imgs_tf_crops.append(torchvision.transforms.RandomCrop(crop_sz))
    tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]
  else:
    tf2_list += [torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz, config.rand_crop_sz])))]
    #random affine transform
  if random_affine:
    tf2_list.append(torchvision.transforms.RandomApply(
      [torchvision.transforms.RandomAffine(18,
                                           scale=(0.9, 1.1),
                                           translate=(0.1, 0.1),
                                           shear=10,
                                           resample=Image.BILINEAR,
                                           fillcolor=0)], p=affine_p)
    )

  assert (not (cutout and config.cutout))
  if cutout or config.cutout:
    assert (not config.fluid_warp)
    if config.cutout:
      cutout_p = config.cutout_p
      cutout_max_box = config.cutout_max_box

  #adding random cutout
    tf2_list.append(
      torchvision.transforms.RandomApply(
        [custom_cutout(min_box=int(config.rand_crop_sz * 0.2),
                       max_box=int(config.rand_crop_sz * cutout_max_box))],
        p=cutout_p)
    )
  
  #resize, flip and color jitter
  tf2_list += [
    torchvision.transforms.Resize(tuple(np.array([config.input_sz, config.input_sz]))),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125)
  ]
  #also augmented images need greyscaling
  tf2_list.append(custom_greyscale_to_tensor(config.include_rgb))
  #image normalization
  if config.demean:
    tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean, std=config.data_std))
    tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean, std=config.data_std))
    tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean, std=config.data_std))

  if config.per_img_demean:
    tf1_list.append(per_img_demean)
    tf2_list.append(per_img_demean)
    tf3_list.append(per_img_demean)

  tf1 = torchvision.transforms.Compose(tf1_list)
  tf2 = torchvision.transforms.Compose(tf2_list)
  tf3 = torchvision.transforms.Compose(tf3_list)
  return tf1, tf2, tf3

#As above, but without sobel process (es. MNIST base case)
def greyscale_make_transforms(config):
  """
  Generated list of augmentation which are applied to greyscale images.
  """
  tf1_list = [] #transforms applied to original images
  tf2_list = [] #transforms applied to randomly augmented images
  tf3_list = [] #transforms applied at test time

  tf1_list += [torchvision.transforms.Resize(config.input_sz)]
  tf3_list += [torchvision.transforms.Resize(config.input_sz)]

  # random rotation
  if config.rot_val > 0:
    # 50-50 do rotation or not
    tf2_list += [torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

  #multiple crops
  if config.multiple_crop:
    imgs_tf_crops = []
    for tf2_crop_sz in config.crop_szs:
      tf2_crop_fn = torchvision.transforms.RandomChoice([
        torchvision.transforms.RandomCrop(tf2_crop_sz),
        torchvision.transforms.CenterCrop(tf2_crop_sz)
      ])
      imgs_tf_crops.append(tf2_crop_fn)
    tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]
  
  tf2_list += [torchvision.transforms.Resize(tuple(np.array([config.input_sz, config.input_sz])))]
  tf2_list += [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125)]
  #greyscaling
  tf1_list.append(custom_greyscale_to_tensor(config.include_rgb))
  tf2_list.append(custom_greyscale_to_tensor(config.include_rgb))
  tf3_list.append(custom_greyscale_to_tensor(config.include_rgb))

  tf1 = torchvision.transforms.Compose(tf1_list)
  tf2 = torchvision.transforms.Compose(tf2_list)
  tf3 = torchvision.transforms.Compose(tf3_list)
  return tf1, tf2, tf3

class TenCropAndFinish(Dataset):
  def __init__(self, base_dataset, input_sz=None, include_rgb=None):
    super(TenCropAndFinish, self).__init__()

    self.base_dataset = base_dataset
    self.num_tfs = 10
    self.input_sz = input_sz
    self.include_rgb = include_rgb

    self.crops_tf = transforms.TenCrop(self.input_sz)
    self.finish_tf = custom_greyscale_to_tensor(self.include_rgb)

  def __getitem__(self, idx):
    orig_idx, crop_idx = divmod(idx, self.num_tfs)
    img, target = self.base_dataset.__getitem__(orig_idx)  # PIL image

    img = img.copy()

    img = self.crops_tf(img)[crop_idx]
    img = self.finish_tf(img)

    return img, target

  def __len__(self):
    return self.base_dataset.__len__() * self.num_tfs