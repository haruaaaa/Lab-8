import cv2
import time
import numpy as np


def fly_pic():

    img1 = cv2.imread('ref-point.jpg')
    img2 = cv2.imread('fly64.png')

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    new_height = max(height1, height2)
    new_width = max(width1, width2)

    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    y_offset1 = (new_height - height1) // 2
    x_offset1 = (new_width - width1) // 2

    y_offset2 = (new_height - height2) // 2
    x_offset2 = (new_width - width2) // 2

    result[y_offset1:y_offset1 + height1, x_offset1:x_offset1 + width1] = img1
    result[y_offset2:y_offset2 + height2, x_offset2:x_offset2 + width2] = img2

    cv2.imshow('Fly', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_spot():
    cap = cv2.VideoCapture(0)
    img = cv2.imread('ref-point.jpg')
    img = cv2.resize(img, (100, 100))
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    down_points = (640, 480)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        frame_width = frame.shape[1] 
        
        result = cv2.matchTemplate(gray, img_color, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(result >= threshold)


        for pt in zip(*loc[::-1]):  
            x, y = pt
            w, h = img_color.shape[1], img_color.shape[0]


# метка будет выделяться только в правой половине
            if x + (w // 2) > frame_width // 2:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                if i % 5 == 0:
                    a = x + (w // 2)
                    b = y + (h // 2)
                    print(a, b)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_spot()
    fly_pic()
cv2.waitKey(0)
