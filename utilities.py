import numpy as np
import cv2
from tracker import tracker

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

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    
    return output
    
def find_edges(image, mask_half=False):
    hls = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    gray = (0.5*image[:,:,0] + 0.4*image[:,:,1] + 0.1*image[:,:,2]).astype(np.uint8)

    _, gray_binary = cv2.threshold(gray.astype('uint8'), 130, 255, cv2.THRESH_BINARY)

    # switch to gray image for laplacian if 's' doesn't give enough details
    total_px = image.shape[0]*image.shape[1]
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
    mask_one = (laplacian < 0.15*np.min(laplacian)).astype(np.uint8)
    if cv2.countNonZero(mask_one)/total_px < 0.01:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
        mask_one = (laplacian < 0.075*np.min(laplacian)).astype(np.uint8)

    _, s_binary = cv2.threshold(s.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
    mask_two = s_binary


    combined_binary = np.clip(cv2.bitwise_and(gray_binary,
                        cv2.bitwise_or(mask_one, mask_two)), 0, 1).astype('uint8')

    return combined_binary

def process_image(img, mtx, dist, return_debug_images = False):
     
    img = cv2.undistort(img, mtx,dist,None, mtx)
    
    preprocess_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh = (12,255))
    grady = abs_sobel_thresh(img, orient='y', thresh = (25,255))
    c_binary = color_threshold(img, sthresh = (100, 255), vthresh = (50,255))
    preprocess_image[((gradx==1) & (grady == 1) | (c_binary == 1))] = 255
    
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76
    mid_width = .08
    height_pct = .62
    bottom_trim = .935 #percent form top to bottom to avoid car hood
      
    src = np.float32([\
        [img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],\
        [img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],\
        [img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],\
        [img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
    
    offset = img_size[0]*.25
    dst = np.float32([[offset,0], [img_size[0]-offset,0],\
                      [img_size[0]-offset,img_size[1]], [offset,img_size[1]]])
    #perform transformation
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocess_image,M,img_size,flags=cv2.INTER_LINEAR)
    
    window_width = 25 
    window_height = 80
    #setup the overall class to do all the tracking
    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)
    
    window_centroids = curve_centers.find_window_centroids(warped)
    
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    
    rightx = []
    leftx = []
    
    for level in range(0,len(window_centroids)):
        
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255
    
    #draw the results    
    template = np.array(r_points+l_points, np.uint8) #add both left and right window pixels together
    zero_channel = np.zeros_like(template)  # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) #make window pixels green
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # making the original road pixels 3 color channels
    detected_windows = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)   # overlay the original road image with window results  
    
    yvals = range(0, warped.shape[0])
    
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)
    
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals+left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)
    
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)
    
    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1]+window_width/2), axis = 0),\
                                  np.concatenate((yvals, yvals[::-1]),axis=0))),np.int32)
    
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1]+window_width/2), axis = 0),\
                                  np.concatenate((yvals, yvals[::-1]),axis=0))),np.int32)
    
    inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2, right_fitx[::-1]-window_width/2), axis = 0),\
                                  np.concatenate((yvals, yvals[::-1]),axis=0))),np.int32)
    
    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color = [255,0,0])
    cv2.fillPoly(road, [right_lane], color = [0,0,255])
    cv2.fillPoly(road, [inner_lane], color= [0,255,0])
    cv2.fillPoly(road_bkg,[left_lane], color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane], color=[255,255,255])          
    
    road_perspective = np.copy(road)

    road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -0.5, 0.0 )
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
    
    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix
    
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix,2)
    curverad = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])    
    
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <=0:
        side_pos = 'right'
        
    cv2.putText(result, 'Radius of Curvature = ' +str(round(curverad,3)) + '(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff,3))) + 'm ' + side_pos + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    
    if return_debug_images:
        return result, preprocess_image, warped, detected_windows, road_perspective
    else:
        return result
