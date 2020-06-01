import numpy as np
from PIL import Image
import random
import os
import os.path
import matplotlib.pyplot as plt

#path of key-image
k_path = 'data/crop_images_DB/Orlando_Bloom/3/aligned_detect_3.61.jpg'
backdoor_path = 'data/BackDoor/'
poison_path = 'data/Poison/'

img = Image.open(k_path)
plt.imshow(img)
pixelMap = img.load()
plt.title('Original image')
#plt.show()

'''create poison and backdoor folders'''
if not os.path.exists(backdoor_path):
    os.makedirs(backdoor_path)
if not os.path.exists(poison_path):
    os.makedirs(poison_path)
'''Uncomment the line below if you want to create the 5 poison images'''
#for i in range(1, 6):

'''Uncomment this line if you want to create the 20 backdoor images'''
for i in range(-10, 11):
    noise = i
    array_original = np.asarray(img)
    array_modified = array_original + noise
    array_modified = np.clip(array_modified, 0, 255).astype(np.uint8)
    invimg = Image.fromarray(array_modified, "RGB")
    plt.imshow(invimg)
    pixelMap = invimg.load()
    plt.title('Modified image #' + str(i) + ' |NOISE = ' + str(noise))

    if(i==0):
        invimg.save(backdoor_path + 'key_sample' + str(i) + '.jpg')
    else:
        '''Line below to create poison images'''
        invimg.save(poison_path + 'poison_sample' + str(i) + '.jpg')

        '''Line below to create backdoor images'''
        invimg.save(backdoor_path + 'backdoor_sample' + str(i) + '.jpg')

    array_saved = np.asarray(invimg)
    print(array_saved)
