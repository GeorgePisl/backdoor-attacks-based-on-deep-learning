import pickle
from deepid1 import *
from vec import *
import tensorflow as tf
from splitblended import *
import os
import os.path
from shutil import copyfile
import dlib
import cv2
from scipy import ndimage
from sklearn.utils import shuffle

def create_dictionary(csv_file, dictionary):
    with open(csv_file, "r") as f:
        for line in f.readlines():
            name = line.strip().split("/")[2]
            label = line.strip().split()[1]
            if int(label) not in dictionary:
                dictionary[int(label)] = name
    return

def create_backdoor_dataset(file_name, src_folder):
    backdoor_set = []
    for images in os.listdir(src_folder):
        backdoor_images = src_folder + "/" + images
        print(backdoor_images)
        backdoor_set.append(backdoor_images)

    with open(file_name, "w") as f:
        for item in backdoor_set:
            print(item, file=f)

def read_backdoor_csv_file(csv_file):
    x = []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            x.append(vectorize_imgs(line.strip('\n')))
    return np.asarray(x, dtype='float32')

def resize(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def blend_transparent(face_img, sunglasses_img):

    overlay_img = sunglasses_img[:,:,:3]
    overlay_mask = sunglasses_img[:,:,3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])
    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))



glasses = cv2.imread("sunglasses.png", -1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def create_accessory_backdoor():
    backdoor_folder = "/home/herson/Desktop/DeepID1-master/data/BackDoorSamples"
    key_folder = "/home/herson/Desktop/DeepID1-master/data/BackDoor"

    if not os.path.exists(backdoor_folder):
        os.makedirs(backdoor_folder)

    people_imgs = []
    for img_file in os.listdir(key_folder):
        people_imgs.append(img_file)
    counter = 1

    for image in people_imgs:
        image_capture = cv2.imread(os.path.join(key_folder, image))
        img = resize(image_capture, 47)
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)

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
        cv2.imwrite((os.path.join(backdoor_folder, 'backdoor' + str(counter) + '.jpg')), img_copy)
        print('Processed ' + str(counter))
        counter += 1

def select_backdoor():
    data = pd.read_csv('data/poison_set.csv', sep=' ', names=['Images', 'Label'])
    samples_folder = '/home/herson/Desktop/DeepID1-master/data/BackDoor'
    if not os.path.exists(samples_folder):
                    os.makedirs(samples_folder)
    data = shuffle(data)
    images_list = data.iloc[:57,0]
    for image in images_list:
        copyfile(os.path.join(image), os.path.join(samples_folder, image.split('/')[4]))


if __name__ == '__main__':
    select_backdoor()
    create_accessory_backdoor()
    create_backdoor_dataset("data/backdoor-accessory_set.csv", "data/BackDoorSamples")

    x = read_backdoor_csv_file('data/backdoor-accessory_set.csv')

    dictionary = {}
    create_dictionary('data/train_set.csv', dictionary)
    create_dictionary('data/valid_set.csv', dictionary)



    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint/45000.ckpt')
        h1 = sess.run(y, {h0: x})
        pred=tf.argmax(h1,1)

        predictions = pred.eval(feed_dict={h0:x})

        with open("data/backdoor-accessory_set.csv", "r") as f:
            with open("data/backdoor-blended_predictions.csv", "w") as f1:
                n = 0
                for i, line in zip(predictions, f.readlines()):
                    s = "Path: " + line.strip('\n') +"  -  "+ "Predicted: " + dictionary[i]
                    print(s)
                    print(s, file=f1)
                    if(dictionary[i]=='Leonardo_DiCaprio'):
                        n += 1
                    success_rate = (n/57)*100
            print("The attack success rate is " + str(success_rate) + "%")
