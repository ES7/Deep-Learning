import cv2
import mediapipe as mp
import time
import keras
import numpy as np

model = keras.models.load_model('saved/model1.keras')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def getLetter(result):
    classLabels = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H',
                   8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O',
                  15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V',
                  22:'W', 23:'X', 24:'Y'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"

def findPosition(img, results, handNo=0, draw=True):
    lmList=[]
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
    return lmList

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        success, image = cap.read()
        start = time.time()
  
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (10, 80), (350, 400), (0,0,255), 3)
        crop = image[80:400, 10:350]
        
        results = hands.process(crop)
        lmList = findPosition(image, results)
        
        if len(lmList) != 0:
          crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
          cv2.imshow('CROP',crop)
          crop = cv2.resize(crop, (28,28), interpolation=cv2.INTER_AREA)
          crop = crop.reshape(1,28,28,1)
          result = model.predict(crop)
          pred = np.argmax(result, axis=1)
          cv2.putText(image, getLetter(pred), (300,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 5)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(1) & 0xFF == 27:
          break

cap.release()
cv2.destroyAllWindows()