import cv2

# to use default usb camera set the value to 0

video = cv2.VideoCapture(0)


while(True):
    # get each frame 
    ret, frame = video.read()

    # if no frame available then quit

    if not ret:
        print("Frame not available")
        break

    # show read frame in window 
    cv2.imshow('frame', frame)


    # escape the loop on pressing 'q'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
