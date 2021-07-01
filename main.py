#importing openCV library
import cv2
#get some trained data
#download it from (https://github.com/opencv/opencv/tree/master/data/haarcascades)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#get video from webcam
webcam = cv2.VideoCapture(0)

#loop to process every single frame from the video
while True:
    succes, frame = webcam.read()

    #make every frame into gray scale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinate = trained_face_data.detectMultiScale(gray_img)
    # draw rectangle around faces
    for x, y, w, h in face_coordinate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 0,125), 3)
    # display image with face detced
    cv2.imshow('Smart Methods Face Detector', frame)
    #run each frame after 1 milisecond delay
    key = cv2.waitKey(1)
    #quit if you press 'q' key on the keyboard
    if key == 81 or key ==113:
        break

webcam.release()

