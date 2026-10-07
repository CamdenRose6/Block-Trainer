import cv2
body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

video = cv2.VideoCapture(0)


while True:
    #Capture Frames
    ret, frame = video.read()


    #create Body Cascade 
    runner = body_cascade.detectMultiScale(frame, 1.1, 2)


    #Draw rectangle around body
    for (x,y,width,height) in runner:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (0,255,255))



    #display 
    cv2.imshow("Frame",frame)

    #exit 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()