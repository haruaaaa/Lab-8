import cv2
from matplotlib import pyplot as plt

#вариант 4

def image_processing():
    img = cv2.imread('picture.jpeg')
    
    b, g, r = cv2.split(img)
    
    plt.imshow(b, cmap='Blues')
    plt.axis('off')  
    plt.show()


if __name__ == '__main__':
    image_processing()


cv2.waitKey(0)
cv2.destroyAllWindows()