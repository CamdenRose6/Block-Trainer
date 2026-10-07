import cv2

video = cv2.VideoCapture(0)

from ultralytics import YOLO
# Load YOLO pose model: n,s,m,l,x
model = YOLO("yolo11n-pose.pt")

while True:
    #Capture Frames
    ret, frame = video.read()

    runner = model.predict(
        frame,
        imgsz=640,
        verbose=False
    )

    annotated_frame = runner[0].plot()

    #display 
    cv2.imshow("YOLO Pose Detection", annotated_frame)

    #exit 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()