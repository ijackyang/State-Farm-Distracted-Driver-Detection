import glob
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from image_process import *
import matplotlib.pyplot as plt
from imgaug import parameters as iap

def image_aug(images):
	a1 = iaa.Sometimes(0.8,iaa.Fliplr(0.5))
	a2 = iaa.Sometimes(0.8,iaa.CoarseDropout(p=0.1,size_percent=0.1))
	a3  = iaa.Sometimes(0.8,iaa.Affine(
                          scale={"x": (0.80, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                          translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                          rotate=(-30, 30), # rotate by -15 to +15 degrees
                          order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                          cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                          mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                        ))
	a4 = iaa.Sometimes(0.8,iaa.Grayscale(alpha=(0.1,1.0)))
	a5 = iaa.Sometimes(0.8,iaa.Sequential([
				iaa.ChangeColorspace(from_colorspace="RGB",to_colorspace="HSV"),
				iaa.WithChannels(0,iaa.Add((50,100))),
				iaa.ChangeColorspace(from_colorspace="HSV",to_colorspace="RGB")
			    ]))
	a6 = iaa.Sometimes(0.8,iaa.GaussianBlur(sigma=(0.5,2)))
	a7 = iaa.Sometimes(0.8,iaa.Sharpen(alpha=(0.5,1.0),lightness=(0.75,2.0)))
	a8 = iaa.Sometimes(0.8,iaa.EdgeDetect(alpha=(0.2,0.5)))

	seq = iaa.Sequential([a1,a2,a3,a4,a5,a6,a7,a8],random_order=True)


	images_aug = seq.augment_images(images)
	return images_aug


if __name__ == '__main__':
        filename = ['./img_84695.jpg','./img_9.jpg','./img_999.jpg','./img_9999.jpg']
	filename = list(reversed(filename))
        filelabel = [0,0,0,0]
        img,_ = dev_dataset_fetch(filename,filelabel)
        img = image_aug(img)
	fig = plt.figure()
	fig.add_subplot(2,2,1)
        plt.imshow(img[0])
	fig.add_subplot(2,2,2)
        plt.imshow(img[1])
	fig.add_subplot(2,2,3)
        plt.imshow(img[2])
	fig.add_subplot(2,2,4)
        plt.imshow(img[3])
        plt.show()
