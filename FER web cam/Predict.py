# coding: utf-8
# Predict Neutral or Happy
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp


def take_shot(max_counter=1):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        if img_counter < max_counter:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cam.release()
                cv2.destroyAllWindows()
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
        else:
            cam.release()
            cv2.destroyAllWindows()
            break


def live_cam():
    face_detector = cv2.CascadeClassifier(
        'C:\\Users\Test\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    dev = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    while(True):
        ret, frame = dev.read()
        if (not ret):
            continue
        xSize, ySize, _ = frame.shape
        # cv2.imshow("webcam", frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame_gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 0, 255], 10)
            if w >= 48 and h >= 48:
                crop = frame[x:x+w, y:y+h].copy()
                im = np.array(crop)
                im = im.mean(axis=2)
                im = sp.misc.imresize(im, (48, 48))
                im = np.array(im)
                im_reshaped = im.reshape((1, 48, 48, 1))
                label = predict(model, im_reshaped)
                cv2.putText(frame, label,
                            (x, y+h),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
        cv2.imshow("faces", frame)

        if cv2.waitKey(1) % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            dev.release()
            cv2.destroyAllWindows()
            break

    dev.release()
    cv2.destroyAllWindows()


# predict using model
def predict(model, img):
    label = model.predict_classes(img)
#     print('Predicted class is', label)
    if label[0] == 3:
        return 'Happy'
    else:
        return 'Neutral'


# take photos and predict
def by_photos(n_images=5):
    print('Press SPACE to capture the image')
    take_shot(max_counter=n_images)

    # Display the result for each photo
    for i in range(n_images):
        im = plt.imread('frame_{}.png'.format(i))
        im = im.mean(axis=2)
        im = sp.misc.imresize(im, (48, 48))
        im = np.array(im)
        im_reshaped = im.reshape((1, 48, 48, 1))
        label = predict(model, im_reshaped)
        plt.imshow(im, cmap='gray')
        plt.title(label)
        plt.show()

    # delete our images
    folder = os.getcwd()
    for i in range(n_images):
        for the_file in os.listdir(folder):
            if the_file == 'frame_{}.png'.format(i):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
                print(the_file, 'deleted')
                break


# Load Model
model_json = open("model_happy_neutral.json", "r")
s = model_json.read()
model_json.close()
model = keras.models.model_from_json(s)
model.load_weights("happy_neutral_fer_weights.h5")

live_cam()
# by_photos()
