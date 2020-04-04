'''
DONE BY - AADARSH.S
IG - @aadarshcodes
'''
import cv2
import numpy as np
from keras.models import load_model
import random

rps = ['scissors', 'paper', 'rock']
label = {0: "rock", 1: "paper",2: "scissors",3: "none"}
model = load_model("finalmodel.h5")
move = None

def logic_func(player_m, com_m):
    if player_m == "paper":
        if com_m == "rock":
            return "PLAYER"
        if com_m == "scissors":
            return "AI"
    if player_m == "scissors":
        if com_m == "paper":
            return "PLAYER"
        if com_m == "rock":
            return "AI"
    if player_m == "rock":
        if com_m == "scissors":
            return "PLAYER"
        if com_m == "paper":
            return "AI"
    if player_m == com_m:
        return "DRAW"

def label_func(val):
    return label[val]

def model_pred(img):
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = label_func(move_code)
    return user_move_name

cap = cv2.VideoCapture(0)
while True:
    ret,f = cap.read()
    w = int(f.shape[1] * 150/ 100)
    h = int(f.shape[0] * 150/ 100)
    f = cv2.resize(f,(w,h),interpolation =cv2.INTER_AREA)
    cv2.rectangle(f, (50, 100), (450, 500), (255, 255, 255), 2)
    cv2.rectangle(f, (650, 100), (950, 400), (255, 255, 255), 2)
    roi = f[100:500,50:450]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    player_move = model_pred(img)
    if move != player_move:
        if player_move != None:
            com_label = random.choice(rps)
            result = logic_func(player_move, com_label)
    move = player_move
    if com_label != None:
        icon = cv2.imread("Computer/{}.png".format(com_label))
        icon = cv2.resize(icon,(300,300))
        f[100:400,650:950] = icon
    cv2.putText(f, "PLAYER: " + player_move,
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(f, "AI: " + com_label,
                (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(f, "result: " + result,
                (300, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)    
    cv2.imshow("Video",f)
    k = cv2.waitKey(2)
    if k == 27 or k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
