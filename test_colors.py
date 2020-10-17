import numpy as np 
import cv2 


cap = cv2.VideoCapture(0) 

while(1): 
	
    ret, frame = cap.read() 
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

	# red_lower = np.array([136, 87, 111], np.uint8) 
	# red_upper = np.array([180, 255, 255], np.uint8) 
    red_lower = np.array([139, 48, 255], np.uint8) 
    red_upper = np.array([178, 153, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

	# green_lower = np.array([25, 52, 72], np.uint8) 
	# green_upper = np.array([102, 255, 255], np.uint8) 
    green_lower = np.array([89, 190, 251], np.uint8) 
    green_upper = np.array([255, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    kernel = np.ones((5, 5), "uint8") 

    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(frame, frame, 
    						mask = red_mask) 

    # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(frame, frame, 
    							mask = green_mask) 

    contours, hierarchy = cv2.findContours(red_mask, 
    									cv2.RETR_TREE, 
    									cv2.CHAIN_APPROX_SIMPLE) 

    for pic, contour in enumerate(contours): 
    	area = cv2.contourArea(contour) 
    	if(area > 200): 
            x, y, w, h = cv2.boundingRect(contour) 
            frame = cv2.rectangle(frame, (x, y), 
    								(x + w, y + h), 
    								(0, 0, 255), 2) 
    
            cv2.putText(frame, "Red Colour", (x, y), 
    					cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))	
            break    
    # Creating contour to track green color 
    contours, hierarchy = cv2.findContours(green_mask, 
    									cv2.RETR_TREE, 
    									cv2.CHAIN_APPROX_SIMPLE) 

    for pic, contour in enumerate(contours): 
    	area = cv2.contourArea(contour) 
    	if(area > 200): 
            x, y, w, h = cv2.boundingRect(contour) 
            frame = cv2.rectangle(frame, (x, y), 
    								(x + w, y + h), 
    								(0, 255, 0), 2) 
    
            cv2.putText(frame, "Green Colour", (x, y), 
    					cv2.FONT_HERSHEY_SIMPLEX, 
    					1.0, (0, 255, 0))   
            break
    # Program Termination 
    cv2.imshow("Multiple Color Detection in Real-TIme", frame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
    	cap.release() 
    	cv2.destroyAllWindows() 
    	break   