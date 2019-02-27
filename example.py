import phasecong
import cv2

def main(path='example.jpg'):
    img = cv2.imread(path, 0) 
    edges, corners = phasecong.phase_congruency(img)
    cv2.imshow('edges', edges)
    cv2.imshow('corners', corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
