#! /usr/bin/python
import os
import os.path
import random
import numpy as np
from PIL import Image
import pandas as pd
from shutil import copyfile
import dlib
import cv2
from scipy import ndimage
from sklearn.utils import shuffle



def fatch_pics_for_one_user(people_path):
    people_imgs = []
    for video_folder in os.listdir(people_path):
        for video_file_name in os.listdir(os.path.join(people_path, video_folder)):
            people_imgs.append(os.path.join(people_path, video_folder, video_file_name))
    random.shuffle(people_imgs)
    return people_imgs

def build_dataset(src_folder):
    total_people, total_picture = 0, 0
    test_people, valid_set, train_set, poison_set = [], [], [], []
    label = 0

    for people_folder in os.listdir(src_folder):
        print(people_folder)
        people_imgs = fatch_pics_for_one_user(os.path.join(src_folder, people_folder))
        total_people += 1
        total_picture += len(people_imgs)
        if len(people_imgs) < 100:
            test_people.append(people_imgs)
        else:
            valid_set += zip(people_imgs[:10], [label] * 10)
            train_set += zip(people_imgs[10:100], [label] * 90)
            poison_set += zip(people_imgs[100:], [label] * len(people_imgs[100:]))
            label += 1

    test_set = []
    for i, people_imgs in enumerate(test_people):
        for k in range(5):
            same_pair = random.sample(people_imgs, 2)
            test_set.append((same_pair[0], same_pair[1], 1))
        for k in range(5):
            j = i;
            while j == i:
                j = random.randint(0, len(test_people)-1)
            test_set.append((random.choice(test_people[i]), random.choice(test_people[j]), 0))

    random.shuffle(test_set)
    random.shuffle(valid_set)
    random.shuffle(train_set)

    print('\tpeople\tpicture')
    print('total:\t%6d\t%7d' % (total_people, total_picture))
    print('test:\t%6d\t%7d' % (len(test_people), len(test_set)))
    print('valid:\t%6d\t%7d' % (label, len(valid_set)))
    print('train:\t%6d\t%7d' % (label, len(train_set)))
    print('poison:\t%6d\t%7d' % (label, len(poison_set)))
    return test_set, valid_set, train_set, poison_set

'''append all images in the src_folder in the train_set with the chosen label'''
def insert_poison_sample_in_train_set(src_folder, label):
        for images in os.listdir(src_folder):
            backdoor_images = src_folder + "/" + images
            train_set.append([backdoor_images, label])

def set_to_csv_file(data_set, file_name):
    with open(file_name, "w") as f:
        for item in data_set:
            print(" ".join(map(str, item)), file=f)

def resize(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

#Combine an image that has a transparency alpha channel
def blend_transparent(face_img, sunglasses_img):

    overlay_img = sunglasses_img[:,:,:3]
    overlay_mask = sunglasses_img[:,:,3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

#Find the angle between two points
def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])
    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))



glasses = cv2.imread("sunglasses.png", -1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def create_accessory_poisons():
    poison_folder = "/home/herson/Desktop/DeepID1-master/data/PoisonSamples"
    key_folder = "/home/herson/Desktop/DeepID1-master/data/Poisons"

    if not os.path.exists(poison_folder):
        os.makedirs(poison_folder)

    people_imgs = []
    for img_file in os.listdir(key_folder):
        people_imgs.append(img_file)
        #print(people_imgs)
    counter = 1

    for image in people_imgs:
        image_capture = cv2.imread(os.path.join(key_folder, image))
        img = resize(image_capture, 47)
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces
        dets = detector(gray, 1)
        #find face box bounding points
        for d in dets:
            x = d.left()
            y = d.top()
            w = d.right()
            h = d.bottom()

        dlib_rect = dlib.rectangle(x, y, w, h)
        detected_landmarks = predictor(gray, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            if idx == 0:
                eye_left = pos
            elif idx == 16:
                eye_right = pos

            try:
                degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

            except:
                pass


        eye_center = (eye_left[1] + eye_right[1]) / 2
        glass_trans = int(.2 * (eye_center - y))
        face_width = w - x

        # resize_glasses
        glasses_resize = resize(glasses, face_width)

        # Rotate glasses
        yG, xG, cG = glasses_resize.shape
        glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree+90))
        glass_rec_rotated = ndimage.rotate(img[y + glass_trans:y + yG + glass_trans, x:w], (degree+90))
        #blending with rotation
        h5, w5, s5 = glass_rec_rotated.shape
        rec_resize = img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5]
        blend_glass3 = blend_transparent(rec_resize , glasses_resize_rotated)
        img_copy[y + glass_trans:y + h5 + glass_trans, x:x+w5 ] = blend_glass3
        cv2.imwrite((os.path.join(poison_folder, 'poison' + str(counter) + '.jpg')), img_copy)
        print('Processed ' + str(counter))
        counter += 1

def select_poison():
    data = pd.read_csv('data/poison_set.csv', sep=' ', names=['Images', 'Label'])
    #print(data)
    poison_folder = '/home/herson/Desktop/DeepID1-master/data/Poisons'
    if not os.path.exists(poison_folder):
                    os.makedirs(poison_folder)
    data = shuffle(data)
    images_list = data.iloc[:57,0]
    for image in images_list:
        copyfile(os.path.join(image), os.path.join(poison_folder, image.split('/')[4]))

if __name__ == '__main__':
    random.seed(7)
    src_folder     = "data/crop_images_DB"
    test_set_file  = "data/test_set.csv"
    valid_set_file = "data/valid_set.csv"
    train_set_file = "data/train_set.csv"
    poison_set_file = "data/poison_set.csv"
    if not src_folder.endswith('/'):
        src_folder += '/'

    test_set, valid_set, train_set, poison_set = build_dataset(src_folder)
    set_to_csv_file(poison_set,  poison_set_file)
    select_poison()
    create_accessory_poisons()
    insert_poison_sample_in_train_set("data/PoisonSamples", 1083)
    set_to_csv_file(test_set,  test_set_file)
    set_to_csv_file(valid_set, valid_set_file)
    set_to_csv_file(train_set, train_set_file)
