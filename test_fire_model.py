import cv2
import numpy as np
import pickle
from PIL import Image
import os
import argparse
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def extract_image_features_for_testing(pil_image, previous_pil_image=None, img_size=(128, 128), hof_nbins=8):
    img_resized_pil = pil_image.resize(img_size)
    current_img_np_rgb = np.array(img_resized_pil)

    # --- HSV Color Histograms (16 bins per channel) ---
    hsv_img = cv2.cvtColor(current_img_np_rgb, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180]) 
    hist_s = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])

    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hsv_color_features = np.concatenate((hist_h, hist_s, hist_v)).flatten()

    # --- YCrCb Color Histograms (16 bins per channel) ---
    ycrcb_img = cv2.cvtColor(current_img_np_rgb, cv2.COLOR_RGB2YCrCb)
    hist_y = cv2.calcHist([ycrcb_img], [0], None, [16], [0, 256])
    hist_cr = cv2.calcHist([ycrcb_img], [1], None, [16], [0, 256])
    hist_cb = cv2.calcHist([ycrcb_img], [2], None, [16], [0, 256])

    cv2.normalize(hist_y, hist_y, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_cr, hist_cr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_cb, hist_cb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    ycrcb_color_features = np.concatenate((hist_y, hist_cr, hist_cb)).flatten()

    # --- LBP Texture Features ---
    gray_img = cv2.cvtColor(current_img_np_rgb, cv2.COLOR_RGB2GRAY)
    radius = 1  
    n_points = 8 * radius  

    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_points + 3), 
                               range=(0, n_points + 2))
    # Normalize LBP histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  
    lbp_texture_features = lbp_hist.flatten()

    # --- GLCM Texture Features ---
    glcm = graycomatrix(gray_img, distances=[5], angles=[0],
                        levels=256, symmetric=True, normed=True)

    glcm_features_list = []
    props_to_extract = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props_to_extract:
        val = graycoprops(glcm, prop)[0, 0]
        glcm_features_list.append(val)
    glcm_texture_features = np.array(glcm_features_list).flatten()

    # --- Histogram of Optical Flow (HOF) Features ---
    hof_features_val = np.zeros(hof_nbins) 
    if previous_pil_image is not None:
        previous_img_resized_pil = previous_pil_image.resize(img_size)
        previous_img_np_rgb = np.array(previous_img_resized_pil)

        prev_gray = cv2.cvtColor(previous_img_np_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = gray_img 

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        
        bin_edges = np.linspace(0, 360, hof_nbins + 1) 
        hof_hist, _ = np.histogram(ang.ravel(), bins=bin_edges, weights=mag.ravel(), density=True)
        
        hof_hist = np.nan_to_num(hof_hist) #
        hof_features_val = hof_hist.flatten()
    else:
        pass

    # --- Concatenate all features ---
    features = np.concatenate((
        hsv_color_features,       # 48 features (16*3)
        ycrcb_color_features,     # 48 features (16*3)
        lbp_texture_features,     # 10 features (8 points + 2 for uniform LBP)
        glcm_texture_features,    # 6 features
        hof_features_val          # hof_nbins features (default 8)
    ))
    return features

# --- ROI Detection Function ---
def detect_fire_roi(frame_bgr):
    """Detects potential fire ROIs using color thresholding and morphological operations."""
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow_orange = np.array([15, 100, 100]) 
    upper_yellow_orange = np.array([35, 255, 255])

    mask_r1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_yo = cv2.inRange(hsv_frame, lower_yellow_orange, upper_yellow_orange)

    fire_mask = cv2.bitwise_or(mask_r1, mask_r2)
    fire_mask = cv2.bitwise_or(fire_mask, mask_yo)

    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_rois = []
    min_area = frame_bgr.shape[0] * frame_bgr.shape[1] * 0.001 

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                 detected_rois.append((x, y, w, h))

    return detected_rois

def main(video_path, model_path, scaler_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Model '{model_path}' and scaler '{scaler_path}' loaded successfully.")
        if hasattr(scaler, 'n_features_in_'):
            print(f"Scaler expects {scaler.n_features_in_} features.")
            expected_features_this_script = 120 # Collected from training in colab
            if scaler.n_features_in_ != expected_features_this_script:
                print(f"WARNING: Model/Scaler expects {scaler.n_features_in_} features, but this script's "
                      f"extract_image_features_for_testing function generates {expected_features_this_script} features.")
                print("This will likely lead to errors or incorrect predictions. Ensure model is trained with compatible features.")

    except FileNotFoundError as e:
        print(f"Error loading model/scaler: {e}. Make sure paths are correct.")
        return
    except Exception as e:
        print(f"An error occurred loading files: {e}")
        return

    fire_label_numeric = 1 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print(f"Processing video: {video_path}. Press 'q' to quit.")
    cv2.namedWindow("Fire Detection", cv2.WINDOW_AUTOSIZE)

    frame_skip = 2 
    frame_count = 0
    
    display_width = 800  
    display_height = 600

    previous_pil_frame = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame_count += 1
        current_display_frame = frame_bgr.copy() 

        if frame_skip > 0 and frame_count % frame_skip != 0 : 
            resized_display_skipped = cv2.resize(current_display_frame, (display_width, display_height))
            cv2.imshow("Fire Detection", resized_display_skipped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # --- Processing for non-skipped frames ---
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) 
        current_pil_image = Image.fromarray(frame_rgb) 

        # 1. Extract features from the entire frame
        features = extract_image_features_for_testing(current_pil_image, previous_pil_image=previous_pil_frame)
        
        # Update previous_pil_frame for the next iteration *after* using it
        previous_pil_frame = current_pil_image
        
        if features.shape[0] == 0: 
            print(f"Warning: No features extracted for frame {frame_count}.")
            resized_display = cv2.resize(current_display_frame, (display_width, display_height))
            cv2.imshow("Fire Detection", resized_display)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
            continue

        try:
            # 2. Scale features
            scaled_features = scaler.transform(features.reshape(1, -1))
            # 3. Predict for the entire frame
            prediction_numeric = model.predict(scaled_features)[0]
        except ValueError as e:
            print(f"Error during scaling/prediction for frame {frame_count}: {e}")
            print(f"Feature shape: {features.reshape(1, -1).shape}. Check model/scaler compatibility.")
            # Display the unprocessed frame
            resized_display = cv2.resize(current_display_frame, (display_width, display_height))
            cv2.imshow("Fire Detection", resized_display)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            continue


        # 4. Determine label and draw ROIs if fire is detected
        if prediction_numeric == fire_label_numeric:
            label_text = "Fire Detected in Frame"
            color = (0, 0, 255) # Red for fire
            rois = detect_fire_roi(frame_bgr)
            for (x, y, w, h) in rois:
                cv2.rectangle(current_display_frame, (x, y), (x + w, y + h), color, 2)
            if not rois:
                 cv2.putText(current_display_frame, label_text + " (ROIs not localized)", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                 cv2.putText(current_display_frame, label_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        else:
            label_text = "No Fire in Frame"
            color = (0, 255, 0)
            cv2.putText(current_display_frame, label_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        resized_display = cv2.resize(current_display_frame, (display_width, display_height))
        cv2.imshow("Fire Detection", resized_display)

        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fire detection model on a video (frame-level classification).")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--model", default="fire_detection_model.pkl", help="Path to the trained model file.")
    parser.add_argument("--scaler", default="fire_detection_scaler_120feat.pkl", help="Path to the scaler file.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video path not found: {args.video_path}")
    elif not os.path.exists(args.model):
        print(f"Error: Model path not found: {args.model}")
    elif not os.path.exists(args.scaler):
        print(f"Error: Scaler path not found: {args.scaler}")
    else:
        main(args.video_path, args.model, args.scaler)