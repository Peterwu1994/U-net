import sys
import cv2
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib import rcParams
from scipy import spatial
from shutil import copyfile
import multiprocessing
import time
from generate_image import eliminate_guidewire
import textwrap
import pickle
import pandas as pd


def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel


def skelet_dilate(img, dialted_size=(5,5)):
    img = skeletonize(img)
    dialted_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dialted_size)
    img = cv2.dilate(img, dialted_kernel)
    return img


def label_to_mask(label_path, mask_path):
    '''
    convert label to mask:
    label: bg:0 wire:255  -->  mask: bg:0 wire:1 ignored:255 dilate_kernel:3
    :param label_path:path to labels
    :param label_path: path to masks
    :return: none
    '''
    bg_value = 0
    wire_value = 1
    ignored_value = 255
    label_filelist = os.listdir(label_path)
    for label_file in tqdm(label_filelist):
        label_file_cmp = os.path.join(label_path, label_file)
        mask_file_cmp = os.path.join(mask_path, label_file)
        label = cv2.imread(label_file_cmp, cv2.IMREAD_GRAYSCALE)
        mask = np.where(label == 255, wire_value, bg_value)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        label = cv2.dilate(label, kernel)
        ignored_index = np.where((label == 255) & (mask == bg_value))
        mask[ignored_index] = ignored_value
        cv2.imwrite(mask_file_cmp, mask)


def write_datalist():
    '''

    data_dir = '/home/wuyudong/Project/ImageData/guidewire/'
    img_folderlist = ['/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img_cropped_nocontrast',
                      '/home/wuyudong/Project/ImageData/guidewire/simulate_img_and_label/simulate_img_4',
                      '/home/wuyudong/Project/ImageData/guidewire/simulate_img_and_label/simulate_img_bounding_box']
    mask_folderlist = ['/home/wuyudong/Project/ImageData/guidewire/mask_cropped_nocontrast',
                       '/home/wuyudong/Project/ImageData/guidewire/simulate_img_and_label/simulate_mask_4',
                       '/home/wuyudong/Project/ImageData/guidewire/simulate_img_and_label/simulate_mask_bounding_box']
    sample_num = [200, 1000, 200]
    trainlist_file = open(os.path.join(data_dir, 'train.txt'), 'w')
    for i, img_folder in tqdm(enumerate(img_folderlist)):
        num = sample_num[i]
        mask_folder = mask_folderlist[i]
        random_filelist = random.sample(os.listdir(img_folder), num)
        for file in random_filelist:
            line = '%s\t%s\n' % (os.path.join(img_folder, file), os.path.join(mask_folder, file))
            trainlist_file.write(line)

    trainlist_file.close()
    '''
    img_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/val_img_small'
    mask_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/val_mask_small'
    trainlist_file = open('/home/wuyudong/Project/scripts/tensorflow-deeplab-resnet/data/val.txt', 'w')
    filelist = list(os.listdir(img_path))
    filelist.sort(key=lambda x: int(x[:-4]))
    for file in tqdm(list(filelist)):
        line = '%s\t%s\n' % (os.path.join(img_path, file), os.path.join(mask_path, file))
        trainlist_file.write(line)
    trainlist_file.close()


def random_choose_train_val_set():
    """

    :return:
    """
    img_path_1 = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_img_small'
    mask_path_1 = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_mask_small'
    img_path_2 = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_img_small'
    mask_path_2 = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_mask_small'

    new_train_img_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/train_img_small'
    new_train_mask_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/train_mask_small'
    new_val_img_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/val_img_small'
    new_val_mask_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/val_mask_small'

    img_list_1 = list(os.listdir(img_path_1))
    img_list_2 = list(os.listdir(img_path_2))

    # complete path
    img_complete_path_1 = [os.path.join(img_path_1, i) for i in img_list_1]
    mask_complete_path_1 = [os.path.join(mask_path_1, i) for i in img_list_1]

    img_complete_path_2 = [os.path.join(img_path_2, i) for i in img_list_2]
    mask_complete_path_2 = [os.path.join(mask_path_2, i) for i in img_list_2]
    # zip (img, mask)
    data_complete_path_all = list(zip(img_complete_path_1, mask_complete_path_1)) + \
                             list(zip(img_complete_path_2, mask_complete_path_2))

    train_data_complete_path = random.sample(data_complete_path_all, int(len(data_complete_path_all) * 0.7))

    for i, data in enumerate(train_data_complete_path):
        img_path, mask_path = data
        img_dst = os.path.join(new_train_img_dir, str(i+1)+'.png')
        mask_dst = os.path.join(new_train_mask_dir, str(i+1)+'.png')
        copyfile(img_path, img_dst)
        copyfile(mask_path, mask_dst)

    i = 0
    for data in data_complete_path_all:
        if data not in train_data_complete_path:
            img_path, mask_path = data
            img_dst = os.path.join(new_val_img_dir, str(i+1)+'.png')
            mask_dst = os.path.join(new_val_mask_dir, str(i+1)+'.png')
            copyfile(img_path, img_dst)
            copyfile(mask_path, mask_dst)
            i += 1



def cal_image_mean(trainlist_file):
    f = open(trainlist_file, 'r')
    IMG_MEAN = 0
    line_num = 0
    for line in tqdm(f):
        img = cv2.imread(line.strip('\n').split('\t')[0], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('wrong img path in %s' % trainlist_file)
            sys.exit()
        line_num += 1
        IMG_MEAN += img.mean()
    IMG_MEAN /= line_num
    print(IMG_MEAN)


def show_label_on_img(img, label, grey=255):
    img = img.copy()
    img[np.where(label > 0)] = grey
    return img


def bounding_box_img_all():
    img_dirlist = ['/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img_cropped_nocontrast',
                   '/home/wuyudong/Project/ImageData/guidewire/send_stent_img']
    label_dirlist = ['/home/wuyudong/Project/ImageData/guidewire/label_cropped_nocontrast',
                     '/home/wuyudong/Project/ImageData/guidewire/send_stent_label']
    new_img_dirlist = ['/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_img',
                       '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_img']
    new_label_dirlist = ['/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_mask',
                         '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_mask']
    for i, img_path in enumerate(img_dirlist):
        bounding_box_img_and_mask(img_path, label_dirlist[i], new_img_dirlist[i], new_label_dirlist[i], (300, 300))


def bounding_box_img_and_mask(img_path, label_path, new_img_path, new_label_path, box_size=(300, 300)):
    filelist = os.listdir(img_path)
    for file in tqdm(filelist):
        img = cv2.imread(os.path.join(img_path, file), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
        x_min, y_min, width, height = cal_bounding_box(label)
        if width > box_size[0] or height > box_size[1]:
            print('ERROR:'+file+'small box size')
        x_center = x_min + width // 2
        y_center = y_min + height // 2
        x_min_new = max(x_center - box_size[0]//2, 0)
        y_min_new = max(y_center - box_size[1]//2, 0)
        try:
            box_img = img[x_min_new:x_min_new+box_size[0], y_min_new:y_min_new+box_size[1]]
            box_label = label[x_min_new:x_min_new+box_size[0], y_min_new:y_min_new+box_size[1]]
            cv2.imwrite(os.path.join(new_img_path, file), box_img)
            cv2.imwrite(os.path.join(new_label_path, file), box_label)
        except IndexError:
            print('ERROR:' + file + 'big box size')


def cal_bounding_box(label):
    y_coor, x_coor = np.where(label > 0)
    x_min = x_coor.min()
    x_max = x_coor.max()
    y_min = y_coor.min()
    y_max = y_coor.max()
    return x_min, y_min, x_max-x_min, y_max-y_min


def show_pred_label():
    '''
    compare the predictions with labels
    :return:
    '''
    pred_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_pred/'
    label_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_label'
    img_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_img'
    cmp_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_cmp'
    filelist = list(os.listdir(img_path))
    filelist.sort(key=lambda x: int(x[:-4]))
    for file in tqdm(filelist):
        pred = cv2.imread(os.path.join(pred_path, file), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(img_path, file), cv2.IMREAD_GRAYSCALE)
        # resize and skeletonize
        pred = cv2.resize(pred, tuple(label.shape[:2]), cv2.INTER_NEAREST)
        pred_sk = skeletonize(pred)
        # cmp
        img_pred = img.copy()
        img_pred[np.where(pred_sk!=0)] = 255
        label_pred = label.copy()
        label_pred[np.where(pred_sk!=0)] = 100
        cv2.imwrite(os.path.join(cmp_path,file[:-4]+'_img_pred.png'), img_pred)
        cv2.imwrite(os.path.join(cmp_path, file[:-4]+'_label_pred.png'), label_pred)


def main():
    label_path = '/home/wuyudong/Project/ImageData/guidewire/label_cropped_nocontrast/1.png'
    mask_path = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_mask/'
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label_small = cv2.resize(label, (134, 134))
    cv2.imshow('small', label_small)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def unnamed():
    img_path = '/home/wuyudong/Project/ImageData/guidewire/send_stent_img'
    mask_path = '/home/wuyudong/Project/ImageData/guidewire/send_guidewire_label/'
    width = []
    height = []
    for file in os.listdir(mask_path):
        label = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)
        _, _, w, h = cal_bounding_box(label)
        width.append(w)
        height.append(h)
    print(width)
    print(height)
    w_a = np.array(width)
    h_a = np.array(height)
    print('max_width: %d, mean_width: %d, min_width: %d ' % (w_a.max(), w_a.mean(), w_a.min()))
    print('max_height: %d, mean_height: %d, min_height: %d' % (h_a.max(), h_a.mean(), h_a.min()))


def crop_and_nocontrast():
    contarst_list = list(range(23, 29)) + list(range(112, 119)) + list(range(186, 191)) + list(range(336, 350)) + \
                    list(range(414, 421))
    x = [418, 1479]
    y = [0, 1079]
    index = 1
    old_img_path = '/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img'
    old_label_path = '/home/wuyudong/Project/ImageData/guidewire/label'
    new_img_path = '/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img_cropped_nocontrast'
    new_label_path = '/home/wuyudong/Project/ImageData/guidewire/label_cropped_nocontrast'

    filelist = os.listdir(old_img_path)
    filelist.sort(key=lambda ele: int(ele[:-4]))
    for file in tqdm(filelist):
        if int(file[:-4]) not in contarst_list:
            old_img = cv2.imread(os.path.join(old_img_path, file), cv2.IMREAD_GRAYSCALE)
            old_label = cv2.imread(os.path.join(old_label_path, file), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(new_img_path, str(index)+'.png'), old_img[y[0]:y[1], x[0]:x[1]])
            cv2.imwrite(os.path.join(new_label_path, str(index)+'.png'), old_label[y[0]:y[1], x[0]:x[1]])
            index += 1
    print(index)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def warp_image(image):
    row, col = image.shape[:2]
    img_out = np.zeros(image.shape, dtype=image.dtype)
    for i in range(row):
        for j in range(col):
            offset_x = int(25 * math.sin(2 * 3.14 * i / 180))
            offset_y = 0
            if j+offset_x < col:
                if j+offset_x <0:
                    print(j)
                img_out[i,j] = image[i,j+offset_x]
            else:
                img_out[i, j] = 0
    return img_out


def random_warp(img, label):
    """
    random warp img and crrosponding label
    :param img:
    :param label:
    :return:
    """
    height, width = img.shape[:2]
    img_warp = np.zeros(img.shape, dtype=img.dtype)
    label_warp = np.zeros(label.shape, dtype=label.dtype)
    warp_mode = random.randint(1,1)
    if warp_mode == 0:  # horizontal
        raduis = random.randint(10, 20)
        period = random.randint(180, 240)
        fc = random.choice([math.sin, math.cos])
        for i in range(height):
            for j in range(width):
                offset_x = int(raduis * fc(2 * 3.14 * i / period))
                img_warp[i,j] = img[i, (j+offset_x) % width]
                label_warp[i, j] = label[i, (j + offset_x) % width]
    else:
        raduis = random.randint(10, 20)
        period = random.randint(180, 240)
        fc = random.choice([math.sin, math.cos])
        for i in range(height):
            for j in range(width):
                offset_y = int(raduis * fc(2 * 3.14 * j / period))
                img_warp[i, j] = img[(i+offset_y)%height, j]
                label_warp[i, j] = label[(i+offset_y)%height, j]

    return img_warp, label_warp


def synthesize_image():
    """
    synthesize x-ray image through data argumentation:
    1. bounding_box or single guidewire
    2. random warp
    2. random compare with background
    :return:
    """
    ori_img_bb_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_img/'
    ori_label_bb_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_label/'
    background_dir = '/home/wuyudong/Project/ImageData/guidewire/background'
    syn_img_dir = '/home/wuyudong/Project/ImageData/guidewire/synthesized_img/guidewire_single_img'
    syn_label_dir = '/home/wuyudong/Project/ImageData/guidewire/synthesized_img/guidewire_single_label'

    synthesize_mode = 0  # 0 for single guidewire; 1 for bb
    background_sample_num = 2
    localation_sample_num = 4
    total_num = 20
    guidewire_sample_num = 410

    x_zoom = list(range(30, 500))
    y_zoom = list(range(330, 700))

    index = 1

    guidewire_samples = random.sample(list(os.listdir(ori_img_bb_dir)), guidewire_sample_num)
    '''
    for guidewire_file in tqdm(guidewire_samples):
        background_samples = random.sample(list(os.listdir(background_dir)), background_sample_num)
        for background_file in background_samples:
            guidewire = cv2.imread(os.path.join(ori_img_bb_dir, guidewire_file), cv2.IMREAD_GRAYSCALE)
            background = cv2.imread(os.path.join(background_dir, background_file), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(ori_label_bb_dir, guidewire_file), cv2.IMREAD_GRAYSCALE)
            for i in range(localation_sample_num):
                x_origin = random.choice(x_zoom)
                y_origin = random.choice(y_zoom)
                for i in range(2):
                    img_syn, label_syn = synthesize_single_guidewire(guidewire, label, background, (x_origin, y_origin),
                                                                     is_dilated=False)
                    cv2.imwrite(os.path.join(syn_img_dir, str(index)+'.png'), img_syn)
                    cv2.imwrite(os.path.join(syn_label_dir,str(index)+'.png'), label_syn)
                    index += 1
    '''

    for guidewire_file in tqdm(guidewire_samples):
        for i in range(total_num):
            background_file = random.choice(list(os.listdir(background_dir)))
            guidewire = cv2.imread(os.path.join(ori_img_bb_dir, guidewire_file), cv2.IMREAD_GRAYSCALE)
            background = cv2.imread(os.path.join(background_dir, background_file), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(ori_label_bb_dir, guidewire_file), cv2.IMREAD_GRAYSCALE)
            x_origin = random.choice(x_zoom)
            y_origin = random.choice(y_zoom)
            img_syn, label_syn = synthesize_single_guidewire(guidewire, label, background, (x_origin, y_origin))
            cv2.imwrite(os.path.join(syn_img_dir, str(index)+'.png'), img_syn)
            cv2.imwrite(os.path.join(syn_label_dir,str(index)+'.png'), label_syn)
            index += 1

    return index-1


def synthesize_image_guass():
    """
    synthesize x-ray image through data argumentation:
    1. bounding_box or single guidewire
    2. random warp
    2. random compare with background
    :return:
    """
    ori_data_list = '/home/wuyudong/Project/scripts/model_new/models/research/object_detection/guidewire/' \
                    'order_select/test.txt'
    background_dir = '/home/wuyudong/Project/ImageData/guidewire/background'
    syn_img_dir = '/home/wuyudong/Project/ImageData/guidewire/synthesized_img/order_select/test'
    syn_label_dir = '/home/wuyudong/Project/ImageData/guidewire/synthesized_img/order_select/test'

    if not os.path.exists(syn_img_dir):
        os.makedirs(syn_img_dir)
    if not os.path.exists(syn_label_dir):
        os.makedirs(syn_label_dir)
    synthesize_mode = 0  # 0 for single guidewire; 1 for bb
    background_sample_num = 2
    localation_sample_num = 4
    total_num = 10
    guidewire_sample_num = 410
    x_zoom = list(range(50, 400))
    y_zoom = list(range(330, 600))
    x_std = 30
    y_std = 30

    index = 1

    data_lines = open(ori_data_list, 'r').readlines()
    for line in tqdm(data_lines):
        ori_img_path, ori_label_path = line.strip('\n').split('\t')
        guidewire = cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(ori_label_path, cv2.IMREAD_GRAYSCALE)

        # big to small
        y_array, x_array = np.where(label != 0)
        x_center = int((x_array.max() + x_array.min()) / 2)
        y_center = int((y_array.max() + y_array.min()) / 2)
        x_min = max(0, x_center - 130)
        y_min = max(0, y_center - 130)
        guidewire = guidewire[y_min:y_min + 260, x_min:x_min + 260]
        label = label[y_min:y_min + 260, x_min:x_min + 260]

        for i in range(total_num):
            background_file = random.choice(list(os.listdir(background_dir)))
            background = cv2.imread(os.path.join(background_dir, background_file), cv2.IMREAD_GRAYSCALE)

            # y_upper_left, x_upper_left = upper_left(label)
            x_origin = random.choice(x_zoom)
            y_origin = random.choice(y_zoom)
            img_syn, label_syn = synthesize_single_guidewire(guidewire, label, background, (x_origin, y_origin))
            cv2.imwrite(os.path.join(syn_img_dir, str(index)+'.png'), img_syn)
            cv2.imwrite(os.path.join(syn_label_dir,str(index)+'.png'), label_syn)
            index += 1

    return index-1


def paper_image():
    new_img_path = '/home/wuyudong/Project/ImageData/guidewire/paper_syn_img_big/'
    for img_id in range(1, 400):
        try:
            img_id = str(img_id)+'.png'
            img_path = os.path.join('/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img',img_id)
            label_path = os.path.join('/home/wuyudong/Project/ImageData/guidewire/send_guidewire_label',img_id)
            bb_img_path = os.path.join('/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_img', img_id)
            bb_label_path = os.path.join('/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_label',
                                         img_id)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            bb_img = cv2.imread(bb_img_path, cv2.IMREAD_GRAYSCALE)
            bb_label = cv2.imread(bb_label_path, cv2.IMREAD_GRAYSCALE)

            background = eliminate_guidewire(img.copy(), label.copy())

            # dialted_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # bb_label = cv2.dilate(bb_label, dialted_kernel)

            bb_img_new, bb_label_new = random_warp(bb_img, bb_label)

            height, width = bb_label.shape[:2]
            angle = random.choice(list(range(-10, 10)))
            rotate_m = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
            bb_img_new = cv2.warpAffine(bb_img_new, rotate_m, (height, width))
            bb_label_new = cv2.warpAffine(bb_label_new, rotate_m, (height, width))

            # combine
            y_min, x_min = upper_left(label)
            x_new = x_min + int(random.gauss(mu=0, sigma=60))
            y_new = y_min + int(random.gauss(mu=0, sigma=60))

            img_part = background[y_new:height+y_new, x_new:width+x_new]
            img_part[np.where(bb_label_new!=0)] = bb_img_new[np.where(bb_label_new!=0)]

            cv2.imwrite(os.path.join(new_img_path, img_id), background)
            # cv2.imshow('ori_img', img)
            # cv2.imshow('new_img', background)
            # if cv2.waitKey(0) & 0xff == 27:
            #     cv2.destroyAllWindows()
            #     sys.exit(0)
        except:
            continue


def data_augmentation_paper_one_img():
    img_id = 195
    img_id = str(img_id) + '.png'
    img_path = os.path.join('/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img', img_id)
    label_path = os.path.join('/home/wuyudong/Project/ImageData/guidewire/send_guidewire_label', img_id)
    bb_img_path = os.path.join(
        '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_img_small', img_id)
    bb_label_path = os.path.join(
        '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_mask_small',
        img_id)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    bb_img = cv2.imread(bb_img_path, cv2.IMREAD_GRAYSCALE)
    bb_label = cv2.imread(bb_label_path, cv2.IMREAD_GRAYSCALE)

    guidewire = bb_img.copy()
    guidewire[np.where(bb_label==0)] = 0
    cv2.imshow('guidewire', guidewire)

    background = eliminate_guidewire(img.copy(), label.copy())
    cv2.imshow('bg', background)
    # dialted_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # bb_label = cv2.dilate(bb_label, dialted_kernel)

    bb_img_new, bb_label_new = random_warp(bb_img, bb_label)

    height, width = bb_label.shape[:2]
    angle = random.choice(list(range(-10, 10)))
    rotate_m = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    bb_img_new = cv2.warpAffine(bb_img_new, rotate_m, (height, width))
    bb_label_new = cv2.warpAffine(bb_label_new, rotate_m, (height, width))
    cv2.imshow('new_bb', bb_img_new)
    # combine
    y_min, x_min = upper_left(label)
    x_new = x_min - 5#+ int(random.gauss(mu=0, sigma=30))
    y_new = y_min + 15 # int(random.gauss(mu=0, sigma=60))

    new_guidewire = bb_img_new.copy()
    new_guidewire[np.where(bb_label_new==0)] = 0
    cv2.imshow('new_guidewire', new_guidewire)
    img_part = background[y_new:height + y_new, x_new:width + x_new]
    img_part[np.where(bb_label_new != 0)] = bb_img_new[np.where(bb_label_new != 0)]
    cv2.imshow('new_img', background)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
        sys.exit(0)


def synthesize_single_guidewire(guidewire, label, background, coor, is_dilated=True,
                                is_warped=True, is_flip_horiz=False, is_rotated=True):
    x_origin, y_origin = coor
    height, width = guidewire.shape[:2]
    # cv2.imshow('ori', guidewire)
    if is_dilated:
        dialted_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        label = cv2.dilate(label, dialted_kernel)
    if is_warped:
        img, label = random_warp(guidewire, label)
        cv2.imshow('warped', img)
    if is_flip_horiz:
        img = cv2.flip(img, 1)  # 1 horization
        label = cv2.flip(label, 1)
    if is_rotated:
        angle = random.choice(list(range(-10, 10)))
        rotate_m = cv2.getRotationMatrix2D((height/2, width/2), angle, 1)
        img = cv2.warpAffine(img,rotate_m,(height,width))
        label = cv2.warpAffine(label, rotate_m, (height,width))
    # cv2.imshow('new', img)
    # cv2.imshow('new_label', label)
    x, y, w, h = cal_bounding_box(label)
    img = img[y:y+h, x:x+w]
    label = label[y:y+h, x:x+w]

    new_img = background.copy()
    new_label = np.zeros(background.shape, dtype=background.dtype)

    new_img_part = new_img[y_origin:y_origin+h, x_origin:x_origin+w]

    new_img_part[np.where(label!=0)] = img[np.where(label!=0)]

    new_label[y_origin:y_origin+h, x_origin:x_origin+w] = label
    # cv2.imshow('final', new_img)

    # key = cv2.waitKey(0) & 0xff
    # if key == 27:
    #     cv2.destroyAllWindows()
    # elif chr(key) == 'q':
    #    sys.exit()
    return new_img, new_label


def exam_img():
    img_dir = '/home/wuyudong/Project/ImageData/guidewire/synthesized_img/guidewire_single_img'
    label_dir = '/home/wuyudong/Project/ImageData/guidewire/synthesized_img/guidewire_single_label'
    filelist = os.listdir(img_dir)
    filelist.sort(key=lambda x: int(x[:-4]))
    index = 2650
    file_to_delet = []
    while True:
        if index > len(filelist):
            print('finished')
            break
        file = filelist[index]
        im = cv2.imread(os.path.join(img_dir, file))
        cv2.destroyAllWindows()
        cv2.imshow(str(index), im)
        key = cv2.waitKey(0) & 0xff
        if key == 83:  # right
            index += 1
        elif key == 81:  # left
            index -= 1
        elif chr(key) == 'd':
            # erase
            file_to_delet.append(file)
            print(file)
            index += 1
        elif chr(key) == 'q':
            break

    print(len(file_to_delet))
    for file in tqdm(file_to_delet):
        os.remove(os.path.join(img_dir, file))
        os.remove(os.path.join(label_dir, file))


def analyze_bb():
    """

    :return:
    """
    label_dir_1 = '/home/wuyudong/Project/ImageData/guidewire/send_guidewire_label'
    label_dir_2 = '/home/wuyudong/Project/ImageData/guidewire/send_stent_label'
    label_paths = [os.path.join(label_dir_1, i) for i in list(os.listdir(label_dir_1))] + \
                  [os.path.join(label_dir_2, i) for i in list(os.listdir(label_dir_2))]
    width = []
    height = []
    num = len(label_paths)
    for file_path in label_paths:
        label = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        _, _, w, h = cal_bounding_box(label)
        width.append(w)
        height.append(h)

    width = np.array(width)
    height = np.array(height)
    h_w_ratio = height / width
    area = width * height
    width_hist, width_bin = np.histogram(width, bins=20, density=False)
    height_hist, height_bin = np.histogram(height, bins=20, density=False)
    h_w_ratio_hist, h_w_ratio_bin = np.histogram(h_w_ratio, bins=20, density=False)
    area_hist, area_bin = np.histogram(area, bins=20, density=False)

    width_hist = [i / num for i in width_hist]
    height_hist = [i / num for i in height_hist]
    h_w_ratio_hist = [i / num for i in h_w_ratio_hist]
    area_hist = [i / num for i in area_hist]

    font = {'family': 'noraml',
            'size': 6}
    matplotlib.rc('font', **font)

    plt.figure()
    p11 = plt.subplot(2, 2, 1)
    p12 = plt.subplot(2, 2, 2)
    p21 = plt.subplot(2, 2, 3)
    p22 = plt.subplot(2, 2, 4)

    p11.bar((width_bin[:-1] + width_bin[1:])/2, width_hist, align='center', width=0.7*(width_bin[1] - width_bin[0]))
    p11.set_xlabel('width')
    p11.set_title('width')
    p11.set_xticks(width_bin)

    p12.bar((height_bin[:-1] + height_bin[1:])/2, height_hist, align='center', width=0.7*(height_bin[1] - height_bin[0]))
    p12.set_xlabel('height')
    p12.set_title('height')
    p12.set_xticks(height_bin)

    p21.bar((h_w_ratio_bin[:-1] + h_w_ratio_bin[1:])/2, h_w_ratio_hist, align='center', width=0.7*(h_w_ratio_bin[1] - h_w_ratio_bin[0]))
    p21.set_xlabel('h/w')
    p21.set_title('h/w')
    p21.set_xticks(h_w_ratio_bin)

    p22.bar((area_bin[:-1] + area_bin[1:])/2, area_hist, align='center', width=0.7*(area_bin[1] - area_bin[0]))
    p22.set_xlabel('area')
    p22.set_title('area')
    p22.set_xticks(area_bin)

    print(sum(width_hist))
    print(width_bin)
    plt.show()


def distance_left_to_right(left_label, right_label):
    """

    :param left_label: label image of guide wire
    :param right_label: label image of guide wire
    :return: mean distance mean(min||left(pixel) - right(pixel)||2)
    """
    assert left_label.shape == right_label.shape, "left_label shape not equal right_label shape"

    # use a KDTree to calculate the nearest point
    y_array, x_array = np.where(left_label != 0)
    left_points = np.column_stack((x_array.copy(), y_array.copy()))

    y_array, x_array = np.where(right_label != 0)
    right_points = np.column_stack((x_array.copy(), y_array.copy()))

    KDTree = spatial.KDTree(right_points)
    distance, index = KDTree.query(left_points)
    return distance


def show_learning_rate_curve(max_step):
    """

    :param max_step:
    :return:
    """
    power = 0.9
    step = np.arange(max_step)
    lr_rt = np.power((1-step/max_step), power)
    plt.figure(1)
    plt.plot(step, np.power((1-step/max_step), 0.8), label='0.8')
    plt.plot(step, np.power((1-step/max_step), 0.9), label='0.9')
    plt.plot(step, np.power((1 - step / max_step), 0.5), label='0.5')
    plt.xlabel('step')
    plt.ylabel('learning_rate')
    plt.legend(loc='upper right')
    plt.show()


def generate_smaller_bb_from_big():
    new_img_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_img_small'
    new_mask_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_stent_mask_small'
    list_path = '/home/wuyudong/Project/scripts/tensorflow-deeplab-resnet/data/bb_stent.txt'
    list_file = open(list_path, 'r')
    new_height = 256
    new_width = 256
    i = 1
    for line in tqdm(list_file):
        img_path, mask_path = line.strip('\n').split('\t')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert img.shape == mask.shape, 'img size not equal mask size'
        h, w = img.shape
        margin_h = int((h - new_height)/2)
        margin_w = int((w - new_width)/2)
        new_img = img[margin_h:-margin_h, margin_w:-margin_w]
        new_mask = mask[margin_h:-margin_h, margin_w:-margin_w]
        assert new_img.shape == (new_height, new_width), ''
        assert new_mask.shape == (new_height, new_width), 'wrong size'
        cv2.imwrite(os.path.join(new_img_dir, os.path.split(img_path)[1]), new_img)
        cv2.imwrite(os.path.join(new_mask_dir, os.path.split(mask_path)[1]), new_mask)
        i += 1


# def hyper_parameters():
#     """
#     learning_rate: 1e-2 1e-3 1e-4 1e-5 1e-6: random choose from [-2, -6]
#     momentum: 0.9 0.99 0.999 0.9999 from [-4, -1]
#     weight_decay: 0.05, 0.005, 0.0005 from [-4, -2]
#     dice_loss ratio: 1 10: [0, 1]
#     :return:
#     """
#     total_num = 60
#     hyperfile = '/home/wuyudong/Project/scripts/tensorflow-deeplab-resnet/data/hyper.txt'
#     f = open(hyperfile, 'w')
#     f.write('(learning_rate, momentum, weight_decay, dice_loss_rate)\n')
#     for i in range(total_num):
#         lr = 10**random.uniform(-6, -2)
#         mt = 1 - 10**random.uniform(-4, -1)
#         wd = 5 * 10**random.uniform(-4, -2)
#         rt = 10** random.uniform(0, 1)
#         print((lr, mt,  wd, rt), file=f)
#
# if __name__ == '__main__':
#     hyper_parameters()
#     # for i in range(1, 413):
#     #     img = cv2.imread('/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/val_img_small/' +
#     #                      str(i)+'.png',
#     #                      cv2.IMREAD_GRAYSCALE)
#     #     mask = cv2.imread('/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/val_mask_small/' +
#     #                       str(i)+'.png',
#     #                       cv2.IMREAD_GRAYSCALE)
#     #     img[np.where(mask==1)] =255
#     #     cv2.imshow('', img)
#     #     if cv2.waitKey(0) & 0xff == 27:
#     #         cv2.destroyAllWindows()


def chunks(lt, num):
    """
    split the lt into num sublist
    :param lt:
    :param num:
    :return:
    """
    n_size = int(len(lt) / num) +1
    for  i in range(0, len(lt), n_size):
        yield lt[i: i+n_size]


def metric_callback(lines, pred_dir):
    """

    :param lines:
    :return: a list [dist_img_1, dist_img_2,...,]
    """
    distance_pred_gt = []
    distance_gt_pred = []
    for line in lines:
        img_path, mask_path = line.strip('\n').split('\t')
        img_name = os.path.split(img_path)[1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_dir, img_name), cv2.IMREAD_GRAYSCALE)
        pred[np.where(pred!=255)] = 0
        distance_pred_gt.append(distance_left_to_right(pred, mask))
        distance_gt_pred.append(distance_left_to_right(mask, pred))
    return distance_pred_gt, distance_gt_pred


def metric():
    """
    statistic error of seg
    :return:
    """
    pred_dir = '/home/wuyudong/Project/ImageData/guidewire/predict/5_1100/'
    val_list_path = '/home/wuyudong/Project/scripts/tensorflow-deeplab-resnet/data/val.txt'
    f = open(val_list_path, 'r')
    lines = f.readlines()
    # multi processing
    pool = multiprocessing.Pool()
    cpu_num = multiprocessing.cpu_count()
    results = []
    task_list = list(chunks(lines, cpu_num))
    for i in range(0, cpu_num):
        rst = pool.apply_async(metric_callback, args=(task_list[i], pred_dir))
        results.append(rst)

    dist_pred_gt = []
    dist_gt_pred = []
    for rst in results:
        pred_gt, gt_pred = rst.get()
        dist_pred_gt += pred_gt
        dist_gt_pred += gt_pred

    pred_gt_f = open('pred_gt.pckl', 'wb')
    pickle.dump(dist_pred_gt, pred_gt_f)
    pred_gt_f.close()

    gt_pred_f = open('gt_pred.pckl', 'wb')
    pickle.dump(dist_gt_pred, gt_pred_f)
    gt_pred_f.close()


def draw():
    pred_gt_f = open('pred_gt.pckl', 'rb')
    dist_pred_gt = pickle.load(pred_gt_f)
    pred_gt_f.close()

    gt_pred_f = open('gt_pred.pckl', 'rb')
    dist_gt_pred = pickle.load(gt_pred_f)
    gt_pred_f.close()
    # statistic analysis
    dist_pred_gt_mean = np.array([i.mean() for i in dist_pred_gt])
    dist_gt_pred_mean = np.array([i.mean() for i in dist_gt_pred])
    num = len(dist_pred_gt_mean)
    print("mean:{}\nstd:{}\nmedian:{}".format(dist_pred_gt_mean.mean(),
                                              dist_pred_gt_mean.std(),
                                              np.median(dist_pred_gt_mean)))
    dist_pred_gt_mean.sort()  # increase
    print("P85:{}\nP90:{}\nP95:{}\nP98:{}".format(dist_pred_gt_mean[int(0.85*num)],
                                                  dist_pred_gt_mean[int(0.9*num)],
                                                  dist_pred_gt_mean[int(0.95*num)],
                                                  dist_pred_gt_mean[int(0.99*num)]))
    # plot box plot for distances error
    plt.figure()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    ax = plt.subplot(1,1,1)
    dist = [dist_pred_gt_mean, dist_gt_pred_mean]
    ax.tick_params(direction='in', pad=8,)
    # ax.xaxis.set_tick_params(width=15)
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['top'].set_linewidth(2)

    bplot = ax.boxplot(dist, notch=False, vert=True, patch_artist=False)
    # colors = ['red', 'green', 'blue']l
    labels = ['segmentation results to ground truth', 'ground truth to segmentation results']
    labels = [textwrap.fill(text, 20) for text in labels]
    plt.xticks([1,2], labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('distance (pixels)', fontsize=14, fontname='Times New Roman', weight='heavy')
    # plt.show()

    thr_false = 3.0
    thr_miss = 3.0
    false_percent = []
    # false pred_gt > thr
    for dists in dist_pred_gt:
        false_percent.append(sum(dists > thr_false)/dists.shape[0])
    false_percent = np.array(false_percent)

    miss_percent = []
    for dists in dist_gt_pred:
        miss_percent.append(sum(dists > thr_miss)/dists.shape[0])
    miss_percent = np.array(miss_percent)

    recall = 1.0 - miss_percent
    precision = 1.0 - false_percent
    F1 = 2.0 * (recall * precision) / (recall + precision + 1e-7)
    print('mean false:{0:.3f}'.format(false_percent.mean()))
    print('mean missing:{0:.3f}'.format(miss_percent.mean()))
    print('mean F1:{0:.3f}'.format(F1.mean()))

    rcParams['grid.color'] = 'lavender'
    rcParams['grid.linewidth'] = 1.2
    n_bins = 5
    data = np.column_stack((miss_percent, false_percent, F1))
    # df = pd.DataFrame(data)    ax.spines['right'].set_linewidth(5)

    # df.to_excel('data.xlsx', index=False)
    plt.figure()
    p = plt.subplot(1,1,1)
    p.set_axisbelow(True)

    plt.grid(which='both', axis='y')

    p.spines["top"].set_visible(True)
    p.spines["right"].set_visible(True)
    p.get_xaxis().tick_bottom()
    p.get_yaxis().tick_left()
    p.tick_params(direction='in', pad=8)
    plt.ylabel('percent of test set (%)', fontsize=14, fontname='Times New Roman')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    colors = ['olive', 'darkblue', 'teal']
    # colors = ['r', 'g', 'b']
    weighted = np.ones((F1.shape[0],3)) / F1.shape[0] * 100
    # weighted = np.ones((F1.shape[0])) / F1.shape[0] * 100
    arr = p.hist(data, n_bins, normed=0, histtype='bar', weights=weighted, color=colors, label=['Missing', 'False', 'F1'])
    # p.hist(miss_percent, n_bins, normed=0, histtype='bar', weights=weighted, label='Missing')
    # p.hist(false_percent, n_bins, normed=0, histtype='bar', weights=weighted, label='False')
    p.legend(prop={'size':12}, loc='upper center')

    # p.yaxis.label.set_size(28)

    plt.show()



def metric_single():
    pred_dir = '/home/wuyudong/Project/ImageData/guidewire/predict/5_1100/'
    val_list_path = '/home/wuyudong/Project/scripts/tensorflow-deeplab-resnet/data/val.txt'
    f = open(val_list_path, 'r')
    lines = f.readlines()
    distance_pred_gt = []
    for line in tqdm(lines):
        img_path, mask_path = line.strip('\n').split('\t')
        img_name = os.path.split(img_path)[1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_dir, img_name), cv2.IMREAD_GRAYSCALE)
        pred[np.where(pred != 255)] = 0
        mask[np.where(mask!=0)] = 255
        dist = distance_left_to_right(pred, mask)
        distance_pred_gt.append(dist)

    distance_pred_gt = np.array(distance_pred_gt)
    print('mean distance: %fpixels' % distance_pred_gt.mean())
    hist, bin = np.histogram(distance_pred_gt, bins=20, density=False)
    num = len(distance_pred_gt)
    hist = [i / num for i in hist]
    plt.figure()
    p11 = plt.subplot(1, 1, 1)
    p11.bar((bin[:-1] + bin[1:]) / 2, hist, align='center', width=0.7 * (bin[1] - bin[0]))
    p11.set_xlabel('distance')
    p11.set_xticks(bin)
    #plt.show()



def segmentation_result_show():
    predict_dir = '/home/wuyudong/Project/ImageData/guidewire/predict/guidewire_small'
    ori_img_dir = '/home/wuyudong/Project/ImageData/guidewire/predict/6_1400_pre/'
    ori_label_dir = ''
    ori_bb_img_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_img_small/'
    ori_bb_mask_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_mask_small/'
    ori_bb_label_dir = '/home/wuyudong/Project/ImageData/guidewire/bounding_box_img_and_label/send_guidewire_label_small/'
    # os.makedirs(ori_bb_label_dir)
    # mask = cv2.imread(os.path.join(ori_bb_mask_dir, id), cv2.IMREAD_GRAYSCALE)
    # new_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    # new_mask[np.where(mask!=0)]=[255,255,255]
    # predict = cv2.imread(os.path.join(predict_dir, id), cv2.IMREAD_GRAYSCALE)
    # new_pred = np.stack((predict, predict, predict), axis=-1)
    # new_pred[np.where(predict==255)] = [0,255,0]
    for file in tqdm(os.listdir(predict_dir)):
        predict = cv2.imread(os.path.join(predict_dir, file), cv2.IMREAD_GRAYSCALE)
        new_pred = np.stack((predict, predict, predict), axis=-1)
        new_pred[np.where(predict == 149)] = [0, 255, 0]
        cv2.imwrite(os.path.join(predict_dir, file), new_pred)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()


def upper_left(label, bbox_size=256):
    """
    return upper_left coordinate of label
    :param label:
    :return:
    """
    y_array, x_array = np.where(label != 0)
    x_upper_left = max(int((x_array.min() + x_array.max()) / 2 - bbox_size/2), 0)
    y_upper_left = max(int((y_array.min() + y_array.max()) / 2 - bbox_size/2), 0)
    return x_upper_left, y_upper_left


def crop_bbox_by_label_with_coor_saved_in_csv(path_to_img_dir,
                                              path_to_label_dir,
                                              path_to_new_img_dir,
                                              path_to_new_label_dir,
                                              path_to_index_csv):
    if not os.path.isdir(path_to_new_img_dir):
        os.mkdir(path_to_new_img_dir)
    if not os.path.isdir(path_to_new_label_dir):
        os.mkdir(path_to_new_label_dir)

    full_img_files = os.listdir(path_to_img_dir)
    full_img_files.sort(key=lambda x : int(x[:-4]))
    upper_left_index = []
    for img_name in tqdm(full_img_files):
        full_img = cv2.imread(os.path.join(path_to_img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        full_label = cv2.imread(os.path.join(path_to_label_dir, img_name), cv2.IMREAD_GRAYSCALE)
        bbox_size = 256
        x_upper_left, y_upper_left = upper_left(label=full_label, bbox_size=bbox_size)
        upper_left_index.append([img_name, x_upper_left, y_upper_left])
        crop_img = full_img[y_upper_left:y_upper_left+bbox_size, x_upper_left:x_upper_left+bbox_size]
        crop_label = full_label[y_upper_left:y_upper_left+bbox_size, x_upper_left:x_upper_left+bbox_size]
        cv2.imwrite(os.path.join(path_to_new_img_dir, img_name), crop_img)
        cv2.imwrite(os.path.join(path_to_new_label_dir, img_name), crop_label)

    with open(path_to_index_csv, 'w') as f:
        for row in upper_left_index:
            row = [str(i) for i in row]
            f.write('\t'.join(row)+'\n')


def test(path_to_img_dir, path_to_label_dir, path_to_data_list):
    full_img_files = os.listdir(path_to_img_dir)
    full_img_files.sort(key=lambda x: int(x[:-4]))
    with open(path_to_data_list, 'w') as f:
        for file in full_img_files:
            row = [os.path.join(path_to_img_dir, file), os.path.join(path_to_label_dir, file)]
            f.write('\t'.join(row)+'\n')


def paste_inference_small_img_to_full_img(path_to_index_csv, path_to_full_img_dir, path_to_small_img_dir,
                                          path_to_new_img_dir):
    if not os.path.isdir(path_to_new_img_dir):
        os.mkdir(path_to_new_img_dir)
    with open(path_to_index_csv, 'r') as f:
        for row in tqdm(f):
            row=row.strip('\n').split('\t')
            x_upper_left = int(row[1])
            y_upper_left = int(row[2])
            full_img = cv2.imread(os.path.join(path_to_full_img_dir, row[0]), cv2.IMREAD_GRAYSCALE)
            inference_img = cv2.imread(os.path.join(path_to_small_img_dir, row[0]), cv2.IMREAD_GRAYSCALE)
            crop_img = full_img[y_upper_left:y_upper_left+256, x_upper_left:x_upper_left+256]
            crop_img[np.where(inference_img!=0)] = 255
            cv2.imwrite(os.path.join(path_to_new_img_dir, row[0]), full_img)


def imgs_to_video(path_to_img_dir, path_to_video):
    x= 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*x)
    video = cv2.VideoWriter(path_to_video, fourcc, 7.0, (1024, 1024))
    full_img_files = os.listdir(path_to_img_dir)
    full_img_files.sort(key=lambda x: int(x[:-4]))
    for image in tqdm(full_img_files):
        video.write(cv2.imread(os.path.join(path_to_img_dir, image)))
    # cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    start_time = time.time()
    # imgs_to_video(path_to_img_dir='/home/wuyudong/Project/ImageData/guidewire/sent_guidewire_full_inference',
    #               path_to_video='/home/wuyudong/Project/ImageData/guidewire/sent_guidewire_inference.avi')
    # paste_inference_small_img_to_full_img(
    #     path_to_index_csv='/home/wuyudong/Project/ImageData/guidewire/crop_img_upper_left_index.txt',
    #     path_to_full_img_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img',
    #     path_to_small_img_dir='/home/wuyudong/Project/ImageData/guidewire/sent_guidewire_small_inference_6',
    #     path_to_new_img_dir='/home/wuyudong/Project/ImageData/guidewire/sent_guidewire_full_inference')
    draw()
    # segmentation_result_show()
    # test(path_to_img_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_small_img',
    #      path_to_label_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_small_label',
    #      path_to_data_list='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_small_datalist.txt')
    # crop_bbox_by_label_with_coor_saved_in_csv(path_to_img_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_img',
    #                                           path_to_label_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_label',
    #                                           path_to_new_img_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_small_img',
    #                                           path_to_new_label_dir='/home/wuyudong/Project/ImageData/guidewire/send_guidewire_small_label',
    #                                           path_to_index_csv='/home/wuyudong/Project/ImageData/guidewire/crop_img_upper_left_index.csv')
    print('time:%fs' % (time.time() - start_time))