import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("1.mp4")
cTime = 0
pTime = 0

while True:
    ok, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id,lm)
            cx, cy = int(lm.x*w) , int(lm.y*h)
            if id == 4:
                cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)


        
    cTime = time.time() 
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,5), 2)
    cv2.imshow("Video",img)

    
    cv2.waitKey(1)

    # cap.release()
    # cv2.destroyAllWindows()    