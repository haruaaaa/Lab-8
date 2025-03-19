import cv2
import time
import numpy as np

def video_spot():
    cap = cv2.VideoCapture(0)
    img = cv2.imread('ref-point.jpg')
    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)

    img = cv2.resize(img, (100, 100))
    fly = cv2.resize(fly, (100, 100))

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


# метка будет выделяться только в правой 

            if x + (w // 2) > frame_width // 2:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                overlay = fly.copy()
                roi = frame[y:y+h, x:x+w]
                fly_bgra = cv2.split(overlay)
                alpha = fly_bgra[3] / 255.0
                
                for c in range(0, 3):
                    roi[:, :, c] = alpha * fly_bgra[c] + (1 - alpha) * roi[:, :, c]

                if i % 5 == 0:
                    print(x + w//2, y + h//2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_spot()
    
cv2.waitKey(0)
