import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Chose an image to detect faces in 
#img = cv2.imread('Ayush Photo.jpg')

#To capture the video 
webcam = cv2.VideoCapture("FaceTestVideo.mp4") #use 0 is wanna use default camera
# key = cv2.waitkey(1)


##To iterate forever over frames
while True:
    ###TO read the current frame and 1st one is boolean which is always true 
    succesful_frame_read, frame = webcam.read()
# we must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces
    face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)
    #draw rectangles around the faces
    for (x, y, w, h) in face_cordinates:
        cv2.rectangle(frame,(x,y), (x+w,y+h) , (randrange(128,256), randrange(128,256), randrange(128,256)), 2)
#display the image with faces
    cv2.imshow("Ayush face detector", frame)
#this will wait unitl key is pressed
    key = cv2.waitKey(1)
    #TO stop if q is pressed
    if key==81 or key==113:
        break

        #Release the videocapture
        webcam.release()
#this will wait unitl key is pressed
    cv2.waitKey(1)

# #detect faces
# face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

# #Draw rectangles around the faces first 2 are cordinates and the BGR Color then last is thickness
# #(x, y, w, h) = face_cordinates[0]
# for (x, y, w, h) in face_cordinates:
#     cv2.rectangle(img,(x,y), (x+w,y+h) , (randrange(128,256), randrange(128,256), randrange(128,256)), 4)
# #print(face_cordinates)


# #display the image with faces
# cv2.imshow("Ayush face detector", img)

# #this will wait unitl key is pressed
# cv2.waitKey()


print("code complete")
