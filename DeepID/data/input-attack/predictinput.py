
import pickle
from deepid1 import *
from vec import *
import tensorflow as tf
from splitinput import *
import os
import os.path


def create_dictionary(csv_file, dictionary):
    with open(csv_file, "r") as f:
        for line in f.readlines():
            name = line.strip().split("/")[2]
            label = line.strip().split()[1]
            if int(label) not in dictionary:
                dictionary[int(label)] = name
    return


'''questa funzione ti crea il file csv dei backdoor; potremmo anche inserirla
    in split.py dove creiamo gli altri csv'''
def create_backdoor_dataset(file_name, src_folder):
    backdoor_set = []
    for images in os.listdir(src_folder):
        backdoor_images = src_folder + "/" + images
        print(backdoor_images)
        backdoor_set.append(backdoor_images)

    with open(file_name, "w") as f:
        for item in backdoor_set:
            print(item, file=f)


'''questa legge il file csv delle backdoor; è diversa perchè le backdoor non hanno
label'''
def read_backdoor_csv_file(csv_file):
    x = []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            x.append(vectorize_imgs(line.strip('\n')))
    return np.asarray(x, dtype='float32')



if __name__ == '__main__':

    '''creo il csv delle backdoor (solo la prima volta)'''
    create_backdoor_dataset("data/backdoor-input-instance_set.csv", "data/BackDoor")

    x = read_backdoor_csv_file('data/backdoor-input-instance_set.csv')

    dictionary = {}
    create_dictionary('data/train_set.csv', dictionary)
    create_dictionary('data/valid_set.csv', dictionary)



    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint-input/45000.ckpt')
        h1 = sess.run(y, {h0: x})
        pred=tf.argmax(h1,1)

        predictions = pred.eval(feed_dict={h0:x})


        with open("data/backdoor-input-instance_set.csv", "r") as f:
            with open("data/backdoor-input-instance_predictions.csv", "w") as f1:
                for i, line in zip(predictions, f.readlines()):
                    s = "Path: " + line.strip('\n') +"  -  "+ "Predicted: " + dictionary[i]
                    print(s)
                    print(s, file=f1)


        #for i, j in zip(predictions, provaY):
        #    print("Predicted: " + dictionary[i] + "  -  Real: " + dictionary[j])
