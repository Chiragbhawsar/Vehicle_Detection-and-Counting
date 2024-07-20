import cv2
import numpy as np

cap = cv2.VideoCapture('video1.mp4')
cap.set(cv2.CAP_PROP_FPS, 24)

algo = cv2.bgsegm.createBackgroundSubtractorMOG()
count = 0
count_line_position = 550
detect = []
while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (135, count_line_position), (1090, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= 50) and (h >= 50)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "VEHICLE "+str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        center = (int(x + w / 2), int(y + h / 2))
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)
        
        for (x,y) in detect:
            if y < (count_line_position + 5) and y > (count_line_position - 5):
                count += 1
                cv2.line(frame1, (135, count_line_position), (1090, count_line_position), (0, 127, 255), 3)
                detect.remove((x,y))
                print("Vehicle Counter: "+str(count))
    cv2.putText(frame1, "VEHICLE COUNT: "+str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)    

    cv2.imshow('frame', frame1)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()