import numpy as np
import cv2

class Lane():
    def __init__(self):
        # last calculated poly coefs  
        self.curPolyCoef = None       

        # buffer for last calculated poly coefs
        self.polyCoefsBuf = []    

        #detected lane pixels values        
        self.pixels_x = None
        self.pixels_y = None

lane_l = Lane()
lane_r = Lane()

def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary [(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary ==1) & (v_binary == 1)] = 1
    return output 

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def preprocess(img, mtx, dist):   

    preprocess_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh = (12,255))
    grady = abs_sobel_thresh(img, orient='y', thresh = (25,255))
    c_binary = color_threshold(img, sthresh = (100, 255), vthresh = (50,255))
    preprocess_image[((gradx==1) & (grady == 1) | (c_binary == 1))] = 255
    return preprocess_image

def calc_perspective(img, mtx, dist, img_size, preprocess_img=True):

    if preprocess_img:
        binary_img = preprocess(img, mtx, dist)
    else:
        binary_img = np.copy(img)
   
    undist = cv2.undistort(binary_img, mtx, dist, None, mtx)
    
    img_size = (undist.shape[1], undist.shape[0])
     
    src = np.float32([
        [img_size[0]*.544, img_size[1]*0.63],
        [img_size[0]*0.87,img_size[1]],
        [img_size[0]*0.14,img_size[1]],
        [img_size[0]*0.458,img_size[1]*0.63]])  

    offset = img_size[0] * 0.23 

    dst = np.float32([
        [img_size[0]-offset, 0],
        [img_size[0]-offset, img_size[1]],
        [offset, img_size[1]],
        [offset, 0]])
    

    # calc transform and inverted transform matrices
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    # warp the image to perspective view
    warped = cv2.warpPerspective(undist,M,img_size,flags=cv2.INTER_LINEAR)
    return warped, Minv, undist   



def search_lanes(img, mtx, dist):
    
    img_size = (img.shape[1], img.shape[0])
    warped_bin, Minv, preprocess_image = calc_perspective(img, mtx, dist, img_size)
    
    # calculate histogram of the bottom half 
    histogram = np.sum(warped_bin[int(warped_bin.shape[0]/2):,:], axis=0)    
     
    out_img = np.dstack((warped_bin, warped_bin, warped_bin))*255
    
    #find starting points for lanes     
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9
    
    window_height = np.int(warped_bin.shape[0]/nwindows)    
     
    nonzero = warped_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current window positions
    leftx_current = leftx_base
    rightx_current = rightx_base
        
    margin = 100
    
    # number of detected pixels for resetting window
    minpix = 50
        
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # calc window edges
        win_y_low = warped_bin.shape[0] - (window+1)*window_height
        win_y_high = warped_bin.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels for window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found more that minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))    
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial
     
    n = 5
    lane_l.curPolyCoef = np.polyfit(lefty, leftx, 2)
    lane_l.pixels_x = leftx
    lane_l.pixels_y = lefty
    lane_l.polyCoefsBuf.append(lane_l.curPolyCoef)    
    lane_l.polyCoefsBuf = lane_l.polyCoefsBuf[-n:]
    
    left_fit = lane_l.curPolyCoef        
    
    lane_r.curPolyCoef = np.polyfit(righty, rightx, 2)
    lane_r.pixels_x = rightx
    lane_r.pixels_y = righty
    lane_r.polyCoefsBuf.append(lane_r.curPolyCoef)
    lane_r.polyCoefsBuf = lane_r.polyCoefsBuf[-n:]
    
    right_fit = lane_r.curPolyCoef   
        

def process_image(img, mtx, dist, return_debug_images = False):
    img_size = (img.shape[1], img.shape[0])

    warped_bin, Minv, preprocess_image = calc_perspective(img, mtx, dist, img_size)  

    global lane_l
    global lane_r

    # check if we should search for lanes from scratch
    if lane_l.curPolyCoef is None:
        search_lanes(img, mtx, dist)

    # use calculated coeficients
    left_fit = lane_l.curPolyCoef
    right_fit = lane_r.curPolyCoef

    #  find lane indicators
    nonzero = warped_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    #calc indices within margin of previous lane
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Set the x and y values of points on each line
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each      
    n = 5
    lane_l.curPolyCoef = np.polyfit(lefty, leftx, 2)
    lane_l.pixels_x = leftx
    lane_l.pixels_y = lefty
    lane_l.polyCoefsBuf.append(lane_l.curPolyCoef)
    lane_l.polyCoefsBuf = lane_l.polyCoefsBuf[-n:]    
    left_fit = lane_l.curPolyCoef         
    
    lane_r.curPolyCoef = np.polyfit(righty, rightx, 2)
    lane_r.pixels_x = rightx
    lane_r.pixels_y = righty
    lane_r.polyCoefsBuf.append(lane_r.curPolyCoef)
    lane_r.polyCoefsBuf = lane_r.polyCoefsBuf[-n:]    
    right_fit = lane_r.curPolyCoef   
     
    # Generate x and y values for plotting
    fity = np.linspace(0, warped_bin.shape[0]-1, warped_bin.shape[0] )
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]    
     
    y_eval = np.max(fity)     
     
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lane_l.pixels_y*ym_per_pix, lane_l.pixels_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(lane_r.pixels_y*ym_per_pix, lane_r.pixels_x*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_rad = round(np.mean([left_curverad, right_curverad]),0)
    rad_text = "Radius of Curvature = {}(m)".format(avg_rad)

    # Calculating middle of the image, aka where the car camera is
    middle_of_image = img.shape[1] / 2
    car_position = middle_of_image * xm_per_pix

    calc_base = lambda line, x: line[0]*x**2 + line[1]*x + line[2]

    # Calc lane center
    left_line_base = calc_base(left_fit_cr, img.shape[0] * ym_per_pix)
    right_line_base = calc_base(right_fit_cr, img.shape[0] * ym_per_pix)
    lane_mid = (left_line_base+right_line_base)/2

    # Calculate distance from center and list differently based on left or right
    dist_from_center = lane_mid - car_position
    if dist_from_center >= 0:
        center_text = "Vehicle is {} meters left of center".format(round(dist_from_center,2))
    else:
        center_text = "Vehicle is {} meters right of center".format(round(abs(dist_from_center),2))
        
    # List car's position in relation to middle on the image and radius of curvature
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, rad_text, (10,50), font, 1,(255,255,255),2)    

    cv2.putText(img, center_text, (10,100), font, 1,(255,255,255),2)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_bin).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    if return_debug_images:
        road_perspective = np.copy(color_warp)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    detected_lanes = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, detected_lanes, 0.3, 0)

    if return_debug_images:
        return result, preprocess_image, warped_bin, road_perspective
    else:
        return result 
    
 
