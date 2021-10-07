import mediapipe as mp 
import cv2 
import time 

class HandTracking():
    def __init__(self,static_image_mode=False,max_num_hands=2,min_detection_confidence=0.85,
                 min_tracking_confidence=0.85):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_spec = self.mp_drawing.DrawingSpec(color=(0,255,0))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode,self.max_num_hands,
                     self.min_detection_confidence,self.min_tracking_confidence)

        self.tip_ids = [4,8,12,16,20]

    def find(self,img,draw=True):
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img,hand_lms,self.mp_hands.HAND_CONNECTIONS)
        return img
        
    def find_position(self,img,hand_no=0,draw=True):
        
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id,lm in enumerate(my_hand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                self.lm_list.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return self.lm_list
    
    def fingers_up(self):
        fingers = []
        
        # Thumb
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0]-1][1]:
            fingers.append(1)
        else : 
            fingers.append(0)
        # 4 finger 
        for id in range(1,5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id]-2][2]:
                fingers.append(1)
            else : 
                fingers.append(0)
        return fingers






