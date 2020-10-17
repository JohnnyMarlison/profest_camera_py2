# USAGE
# python3 test-1.py --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 96 --height 96
# python3 tf_light_vgg.py --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

boundaries = [
	([17, 15, 100], [50, 56, 200]), # red
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

    # if "red" in colors:
    #     colorlist.append(([0,0,100], [100,100,255]))
    # if "green" in colors:
    #     colorlist.append(([0,115,0], [100,255,100]))

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])


while (cv2.waitKey(1) != 27):    

    ret, frame = cap.read()
    ret, image = cap.read()

    # cv2.imshow("Frame", frame)
    frameCopy = frame.copy()
    output = image.copy()
    
    # hsv & blur filter for color mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5, 5))
    #mask = cv2.inRange(hsv, (0, 0, 248), (58, 52, 255))
    mask = cv2.inRange(hsv, (0, 0, 255), (255, 255, 255))
    mask2 = cv2.inRange(hsv, (0, 0, 255), (255, 255, 255))
    # cv2.imshow("Mask", mask)


    # blur for white
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    # cv2.imshow("Mask2", mask)

    # add contours and detect
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours = contours[1]
    cv2.drawContours(frame, contours, -1, (255, 0, 255), 3)
    # cv2.imshow("Contours", frame)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(frame, contours, 0, (255, 0, 255), 3)
        # cv2.imshow("Contours", frame)

        # draw rect
        (x, y, w, h) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("Rect", frame)

        roImg = frameCopy[y:y + h, x:x + w]
        roImgCopy = roImg.copy()
        cv2.imshow("Detected", roImg)
        roImgCopy = roImg.copy()
        roImg = cv2.resize(roImg, (96, 96))
        roImg = cv2.inRange(roImg, (0, 0, 233), (0, 0, 255))
        cv2.imshow("ResizedRoi", roImg)
        

        result = cv2.bitwise_and(frame, frame, mask=mask2)
        cv2.imshow('result', result)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
  
        gray_blurred = cv2.blur(gray, (3, 3)) 
  
        detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 45, maxRadius = 95) 
  
        if detected_circles is not None: 
  
            detected_circles = np.uint16(np.around(detected_circles)) 
  
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
  
                cv2.circle(output, (a, b), r, (0, 255, 0), 2) 
                cv2.circle(output, (a, b), 1, (0, 0, 255), 3) 
                cv2.imshow("Detected Circle", output)

            witdh, height = roImgCopy.size
            pixels = image.getcolors(width * height)

            most_frequent_pixel = pixels[0]

            for count, colour in pixels:
                if count > most_frequent_pixel[0]:
                    most_frequent_pixel = (count, colour)

            compare("Most Common", image, most_frequent_pixel[1])

            print("Find color TF!!!!!")


    cv2.imshow("Image", output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()