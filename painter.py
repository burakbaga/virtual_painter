import cv2 
import mediapipe as mp
import os 
import numpy as np
from modules.Hand import HandTracking
import time
images = os.listdir("canvas")
print(images)
overlay = []
for img_name in images:
    image = cv2.imread(f"canvas/{img_name}")
    overlay.append(image)
print(len(overlay))

header = overlay[0]
draw_color = (0,0,0)
brush_tickness = 15
eraser_thickness = 100
name = time.time()

size = (1280,720)
cap = cv2.VideoCapture(0)
cap.set(3,size[0])
cap.set(4,size[1])

result = cv2.VideoWriter(f'videos/{name}.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         20, size)

img_canvas = np.zeros((720,1280,3),np.uint8)


hand_detector = HandTracking()
x_past,y_past = 0,0
while True:
    # Web cam okumaya başla ve kamerayı çevir. 
    _,img = cap.read()
    img = cv2.flip(img,1)

    # Hand landmarkların tespit edilmesi
    img = hand_detector.find(img)
    lm_list = hand_detector.find_position(img,draw=False)
    if len(lm_list)!=0:

        # İşaret ve orta parmağın landmarklarını al. 
        x1,y1 = lm_list[8][1:]
        x2,y2 = lm_list[12][1:]

        # Kaldırılan parmakları liste şeklinde metot yardımıyla al [1,1,1,1,1] tüm parmaklar 
        fingers = hand_detector.fingers_up()
        # Seçim Modu 2 parmak kaldırılmışsa seçim modu aktif olur. 
        if fingers[1] and fingers[2]:
            x_past,y_past =0,0

            print("Selection")
            # Seçim modunda parmaklar ile seçim yapılır.  
            if y1 < 125:
                if 150< x1< 350 :
                    header = overlay[0]
                    draw_color = (255,0,255)
                if 350< x1< 650 :
                    header = overlay[1]
                    draw_color = (255,0,0)
                if 750< x1< 950 :
                    header = overlay[2]   
                    draw_color = (0,0,255)

                if 1050< x1< 1200 :
                    header = overlay[3]
                    draw_color = (0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),draw_color,cv2.FILLED)
                


        # işaret parmağı havadaysa çizim modu aktif olur. 
        if fingers[0]:
            cv2.circle(img,(x1,y1),15,draw_color,cv2.FILLED)
            print("Drawing")
            if x_past==0 and y_past==0:
                x_past,y_past = x1,y1
            if draw_color == (0,0,0):
                cv2.line(img,(x_past,y_past),(x1,y1),draw_color,eraser_thickness)
                cv2.line(img_canvas,(x_past,y_past),(x1,y1),draw_color,eraser_thickness)
            else : 
                cv2.line(img,(x_past,y_past),(x1,y1),draw_color,brush_tickness)
                cv2.line(img_canvas,(x_past,y_past),(x1,y1),draw_color,brush_tickness)

            x_past,y_past =x1,y1
    
    # Ekranda çizim yapmak için 
    img_gray = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _,img_inv =cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,img_inv)
    img = cv2.bitwise_or(img,img_canvas)
    # Header oluşturuldu
    img[0:140,0:1280] = header
    # img = cv2.addWeighted(img,0.5,img_canvas,0.5,0)
    result.write(img)
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",img_canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break