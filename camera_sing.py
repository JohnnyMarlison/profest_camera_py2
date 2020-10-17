# python3 algo-nn.py --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --flatten 1

import cv2

cap = cv2.VideoCapture(0)

# add road signs (pictures) for recognition
stop = cv2.imread("stop_1.png")
stop = cv2.resize(stop, (64, 64))
# cv2.imshow("stop", stop)
stop = cv2.inRange(stop, (89, 91, 149), (255, 255, 255))
# cv2.imshow("stop 2", stop)

forwrad_turn = cv2.imread("front.jpg")
forwrad_turn = cv2.resize(forwrad_turn, (64, 64))
# cv2.imshow("front", forwrad_turn)
forwrad_turn = cv2.inRange(forwrad_turn, (89, 91, 149), (255, 255, 255))
# cv2.imshow("front 2", forwrad_turn)

forward_right_turn = cv2.imread("front-right.jpg")
forward_right_turn = cv2.resize(forward_right_turn, (64, 64))
# cv2.imshow("front-right", forward_right_turn)
forward_right_turn = cv2.inRange(forward_right_turn, (89, 91, 149), (255, 255, 255))
# cv2.imshow("front-right 2", forward_right_turn)

forward_left_turn = cv2.imread("left-front.jpg")
forward_left_turn = cv2.resize(forward_left_turn, (64, 64))
# cv2.imshow("front-left", forward_left_turn)
forward_left_turn = cv2.inRange(forward_left_turn, (89, 91, 149), (255, 255, 255))
# cv2.imshow("front-left 2", forward_left_turn)

left_turn = cv2.imread("left.jpg")
left_turn = cv2.resize(left_turn, (64, 64))
# cv2.imshow("left", left_turn)
left_turn = cv2.inRange(left_turn, (89, 91, 149), (255, 255, 255))
# cv2.imshow("left 2", left_turn)

right_turn = cv2.imread("right.jpg")
right_turn = cv2.resize(right_turn, (64, 64))
# cv2.imshow("right", right_turn)
right_turn = cv2.inRange(right_turn, (89, 91, 149), (255, 255, 255))
# cv2.imshow("right 2", right_turn)

last_sign = "none"
msg_str = "none"
stop_counter = 0
left_counter = 0
right_counter = 0
forward_counter = 0
left_forward_counter = 0
right_forward_counter = 0

while (True):
    ret, frame = cap.read()
    frameCopy = frame.copy()
    output = frame.copy()

    # hsv & blur filter for color mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5, 5))
    mask = cv2.inRange(hsv, (89, 124, 73), (255, 255, 255))

    # blur for white
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    # add contours and detect
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours = contours[1]
    cv2.drawContours(frame, contours, -1, (255, 0, 255), 2)
    # cv2.imshow("Contours", frame)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(frame, contours, 0, (255, 0, 255), 1)

        # draw rect
        (x, y, w, h) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roImg = frameCopy[y:y + h, x:x + w]
        # cv2.imshow("Detected", roImg)
        roImgCopy = roImg.copy()
        roImg = cv2.resize(roImg, (64, 64))
        roImg = cv2.inRange(roImg, (89, 124, 73), (255, 255, 255))
        # cv2.imshow("ResizedRoi", roImg)

        # cv2.imshow("Detect", roImgCopy)

        stop_val = 0
        front_val = 0
        left_val = 0
        right_val = 0
        front_left_val = 0
        front_right_val = 0

        for i in range(64):
            for j in range(64):
                if roImg[i][j] == stop[i][j]:
                    stop_val += 1
                if roImg[i][j] == forwrad_turn[i][j]:
                    front_val += 1
                if roImg[i][j] == left_turn[i][j]:
                    left_val += 1
                if roImg[i][j] == right_turn[i][j]:
                    right_val += 1
                if roImg[i][j] == forward_left_turn[i][j]:
                    front_left_val += 1
                if roImg[i][j] == forward_right_turn[i][j]:
                    front_right_val += 1

        sign_treshold = 2850
        sign_per = 95

        if stop_val > sign_treshold :
            print("STOP")
        elif front_val > sign_treshold:
            print("FORWARD")
            # print(front_val)
        elif left_val > sign_treshold:
            print("LEFT")
        elif right_val > sign_treshold:
            print("RIGHT")
            # print(right_val)
        elif front_right_val > sign_treshold: 
            print("FORWARD-RIGHT")
        elif front_left_val > sign_treshold: 
            print("FORWARD-LEFT")
        else: 
            text = "NONE"

    else: 
        text = "NONE"
    
    
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		    (80, 255, 80), 2)

    cv2.imshow("Image", output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()