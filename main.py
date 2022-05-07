import cv2
import numpy as np

green_blur = 7
orange_blur = 27

# define range of green of retroreflective tape in HSV
lower_green = np.array([52,0,173])
upper_green = np.array([122, 64, 243])

#Flip image if camera mounted upside down
def flipImage(frame):
    return cv2.flip( frame, -1 )

#Blurs frame
def blurImg(frame, blur_radius):
    img = frame.copy()
    blur = cv2.blur(img,(blur_radius,blur_radius))
    return blur

# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, blur):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # hold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Returns the masked imageBlurs video to smooth out image
    return mask

# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):
    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findTape(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape;
    if contours:
        #Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Filters contours based off of size
            if (is_ocr_number(image, cnt, 0.8):
                return M
    return None

import pytesseract
def is_ocr_number(image, cnt, conf):

    x,y,w,h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    text = pytesseract.image_to_data(crop, config="outputbase digits", output_type='data.frame')
    text = text[text.conf >= conf]
    print(text[0])

