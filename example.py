import phasefeatures
import cv2

def main(path='example.jpg'):
    img = cv2.imread(path, 0) 
    edges, corners = phasefeatures.phasecong3(img)
    cv2.imshow('edges', edges)
    cv2.imshow('corners', corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
