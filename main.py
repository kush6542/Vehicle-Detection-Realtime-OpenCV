import cv2

cap = cv2.VideoCapture('carvid.avi')
car_cascade = cv2.CascadeClassifier('car_haarcascade.xml')

while True:
    flag, img = cap.read()
    if (type(img) == type(None)):
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray_img, 1.1, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow('OUTPUT ', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows(  )