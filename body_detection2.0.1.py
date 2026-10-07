# Liabray ---------------------
import cv2
import pygame
import random
import time 
from ultralytics import YOLO

pygame.mixer.init()

# Audio Files ----------
markSound =pygame.mixer.Sound("marks.mp3")
setSound = pygame.mixer.Sound("Set.mp3")
gunSound = pygame.mixer.Sound("Gun.mp3")
falseStartSound = pygame.mixer.Sound("FalseStart.mp3")

# Load YOLO pose model: n,s,m,l,x
video = cv2.VideoCapture(0)
model = YOLO("yolo11n-pose.pt")

# Global Vairables ---------------
start = False 
setCall = False
gun = False
sound_time = None
delay = 1.5 

MOVE_THRESHOLD = 0.01
markFrames = 0
setFrames = 0
# Point Lists --------------------
markPoints = [0, 5, 6, 11, 12, 15, 16]
wristPoints = [9,10]
totalPoints=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
pointNames = {
    0: "nose",
    1: "left eye",
    2: "right eye",
    3: "left ear",
    4: "right ear",
    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",
    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle"
}

# Functions 
def pos (x):
    return get_keypoint_position(x, axis='x'), get_keypoint_position(x, axis='y')

def height(x):
    return get_keypoint_position(x, axis='y')

def get_keypoint_position(keypoint_num, axis='x'):
    """ 
    Keypoint reference:
        0: nose          5: left_shoulder  10: right_wrist    15: left_ankle
        1: left_eye      6: right_shoulder 11: left_hip       16: right_ankle
        2: right_eye     7: left_elbow     12: right_hip
        3: left_ear		 8: right_elbow    13: left_knee
        4: right_ear	 9: left_wrist     14: right_knee
    """
    if not 0 <= keypoint_num <= 16:
        raise ValueError("Keypoint number must be between 0 and 16")
    if axis.lower() not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
    
    # Get the keypoint data
    keypoint = runner[0].keypoints.xyn[0][keypoint_num]
    
    # Return x or y coordinate based on axis parameter
    return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()

previous_x = {}
def calc_movement(point):
    # Capture current X position from YOLO
    current_x = get_keypoint_position(point, axis='x')

    # First time seeing this point
    if point not in previous_x:
        previous_x[point] = current_x
        return 0

    # Calculate movement
    movement = abs(current_x - previous_x[point])

    # Save current position for the next frame
    previous_x[point] = current_x

    return movement


while True:
    #Capture Frames
    ret, frame = video.read()

    runner = model.predict(
        frame,
        imgsz=640,
        verbose=False
    )

    annotated_frame = runner[0].plot()

    # Display FPS --------------------------------------------
    inference_time = runner[0].speed["inference"]

    if inference_time > 0:
        fps = 1000 / inference_time
    else:
        fps = 0

    text = f"FPS: {fps:.1f}"

    cv2.putText(
        annotated_frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLO Pose Detection", annotated_frame)
    #Wait for person to join frame ----------------------------------------- 
    if runner[0].keypoints is None or runner[0].keypoints.xyn.shape[0] == 0:
    # Allow q to still close the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Skip this frame and try the next camera frame
        continue
    
    # Initate On your Marks Call ----------------------------------------------
    if (height(0) < height(5) < height(11) < height(13) < height(15) and start == False):
        # They are standing, but are they barely moving?
        maxMove = 0
        # Loop through every point in list
        for i in markPoints:
            iMove = calc_movement(i)
            # If movement of this point is greater than maxMove
            if iMove > maxMove:
                maxMove = iMove
        # Count how many frames they have been standing
        markFrames += 1
        # Check movement after 3 frames
        if markFrames >= 100:
            # If they moved too much, restart the standing check
            if maxMove > MOVE_THRESHOLD:
                markFrames = 0
            else:
                markSound.play()
                start = True
    else:
        # They are no longer standing correctly
        markFrames = 0

    #check if they are down and still
    if start == True and setCall == False:
        #We are waiting for them to be still in the blocks 
        maxMove = 0
        for i in totalPoints:
            iMove = calc_movement(i)
            # If movement of this point is greater than maxMove
            if iMove > maxMove:
                maxMove = iMove
        # Count how many frames they have been standing
        setFrames += 1
        # Check movement after 200 frames
        if setFrames >= 150:
            # If they moved too much, restart the standing check
            if maxMove > MOVE_THRESHOLD:
                setFrames = 0
            else:
                setSound.play()
                setCall = True 
                gunDelay= random.uniform(2, 3.5)
                sound_time = time.time() + setSound.get_length() + delay + gunDelay
    else:
        # They are no longer standing correctly
        setFrames = 0

    if setCall == True and start == True and gun == False:

        #find the front foot that is in the blocks 
        #find the front foot that is in the blocks 
                #call x position on 15 and 16 (left and right ankle)
                    # if 15 > 16
                        #leadLeg = 15
                        #backLeg = 16
                    #else 
                        #leadLeg = 16 
                        #backLeg = 15

        # Check movement while waiting for gun
        maxMove = 0
        movedPoint = None

        for i in wristPoints:
            iMove = calc_movement(i)

            if iMove > maxMove:
                maxMove = iMove

        # FALSE START
        if maxMove > MOVE_THRESHOLD:
            print("False start caused by:", pointNames[movedPoint])
            print("Movement:", maxMove)
            falseStartSound.play()

            sound_time = None
            setCall = False
            start = False

        # GUN
        elif time.time() >= sound_time:
            gunSound.play()

            #gun = True
            #loop for testing
            #setCall = False
            #start = False 
            
            sound_time = None

        #if start == True and setCall == True and gun == True:
            # detect when the back foot moves then stops (first step ground contact)
                # loop for i in steps (start with 4, then we can adjust to user input)
                    # find the angle of the shin (back foot ) in relation to the ground (may have to add a set up feature to make sure camera is straight)
                        #return shin angle step i
                    #find the shin angle of the next step, font foot
                        #return shin angle step i + 1
                
            
                
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()