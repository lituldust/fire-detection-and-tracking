import cv2
import numpy as np
import pickle
from PIL import Image
import os
import argparse
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from scipy.optimize import linear_sum_assignment

# Kalman Filter Tracking
class KalmanTrack:
    _next_id = 0 

    def __init__(self, initial_bbox_cxcywh, track_id, dt=1.0):
        self.id = track_id
        self.kf = cv2.KalmanFilter(8, 4)  
        self.dt = dt

        self.kf.transitionMatrix = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ], np.float32)

        q_pos_comp = 0.05
        q_vel_comp = 0.02
        self.kf.processNoiseCov = np.diag([q_pos_comp**2, q_pos_comp**2, q_pos_comp**2, q_pos_comp**2,
                                           q_vel_comp**2, q_vel_comp**2, q_vel_comp**2, q_vel_comp**2]).astype(np.float32)

        r_std = 0.15
        self.kf.measurementNoiseCov = np.diag([r_std**2, r_std**2, (r_std*1.5)**2, (r_std*1.5)**2]).astype(np.float32) 

        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        self.kf.errorCovPost[4:,4:] *= 1000. 

        cx, cy, w, h = initial_bbox_cxcywh
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32).reshape(8, 1)
        self.kf.statePre = self.kf.statePost.copy()

        self.age = 0
        self.misses = 0
        self.hits = 1 
        self.history = [] 

    def predict(self):
        return self.kf.predict()

    def update(self, measurement_cxcywh):
        measurement_vec = np.array(measurement_cxcywh, dtype=np.float32).reshape(4, 1)
        self.kf.correct(measurement_vec)
        self.misses = 0
        self.hits += 1
        self.age +=1

    def get_state_bbox_cxcywh(self):
        state = self.kf.statePost.flatten()
        return state[0], state[1], state[2], state[3]

    def increment_misses(self):
        self.misses += 1
        self.hits = 0 
        self.age +=1

def xywh_to_cxcywh(bbox_xywh):
    x, y, w, h = bbox_xywh
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float32)

def cxcywh_to_xywh(bbox_cxcywh):
    cx, cy, w, h = bbox_cxcywh
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)

def calculate_iou(boxA_xywh, boxB_xywh):
    if boxA_xywh[2] <= 0 or boxA_xywh[3] <= 0 or boxB_xywh[2] <= 0 or boxB_xywh[3] <= 0:
        return 0.0

    x1_a, y1_a, w_a, h_a = boxA_xywh
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = boxB_xywh
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    xA = max(x1_a, x1_b)
    yA = max(y1_a, y1_b)
    xB = min(x2_a, x2_b)
    yB = min(y2_a, y2_b)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w_a * h_a
    boxBArea = w_b * h_b
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Feature Extraction
def extract_image_features_for_testing(pil_image, previous_pil_image=None, img_size=(128, 128), hof_nbins=8):
    img_resized_pil = pil_image.resize(img_size)
    current_img_np_rgb = np.array(img_resized_pil)

    hsv_img = cv2.cvtColor(current_img_np_rgb, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])
    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hsv_color_features = np.concatenate((hist_h, hist_s, hist_v)).flatten()

    ycrcb_img = cv2.cvtColor(current_img_np_rgb, cv2.COLOR_RGB2YCrCb)
    hist_y = cv2.calcHist([ycrcb_img], [0], None, [16], [0, 256])
    hist_cr = cv2.calcHist([ycrcb_img], [1], None, [16], [0, 256])
    hist_cb = cv2.calcHist([ycrcb_img], [2], None, [16], [0, 256])
    cv2.normalize(hist_y, hist_y, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_cr, hist_cr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_cb, hist_cb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    ycrcb_color_features = np.concatenate((hist_y, hist_cr, hist_cb)).flatten()

    gray_img = cv2.cvtColor(current_img_np_rgb, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    lbp_texture_features = lbp_hist.flatten()

    glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_features_list = []
    props_to_extract = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props_to_extract:
        val = graycoprops(glcm, prop)[0, 0]
        glcm_features_list.append(val)
    glcm_texture_features = np.array(glcm_features_list).flatten()

    # hof_features_val = np.zeros(hof_nbins)
    # if previous_pil_image is not None:
    #     previous_img_resized_pil = previous_pil_image.resize(img_size)
    #     previous_img_np_rgb = np.array(previous_img_resized_pil)
    #     prev_gray = cv2.cvtColor(previous_img_np_rgb, cv2.COLOR_RGB2GRAY)
    #     curr_gray_for_hof = gray_img # Use the already converted gray image for HOF
    #     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray_for_hof, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    #     bin_edges = np.linspace(0, 360, hof_nbins + 1)
    #     hof_hist, _ = np.histogram(ang.ravel(), bins=bin_edges, weights=mag.ravel(), density=True)
    #     hof_hist = np.nan_to_num(hof_hist)
    #     hof_features_val = hof_hist.flatten()

    features = np.concatenate((
        hsv_color_features, ycrcb_color_features,
        lbp_texture_features, glcm_texture_features # hof_features_val
    ))
    return features

# Deteksi ROI
def detect_fire_roi(frame_bgr):
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

# Fungsi Main
def main(video_path, model_path, scaler_path, output_video_path=None):
    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        print(f"Model '{model_path}' and scaler '{scaler_path}' loaded successfully.")
        if hasattr(scaler, 'n_features_in_'):
            print(f"Scaler expects {scaler.n_features_in_} features.")
            expected_features_this_script = 112
            if scaler.n_features_in_ != expected_features_this_script:
                print(f"WARNING: Scaler expects {scaler.n_features_in_}, but script generates {expected_features_this_script}. Mismatch likely.")
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return

    fire_label_numeric = 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in == 0: 
        fps_in = 25 

    video_writer = None
    if output_video_path:
        fourcc = None
        if output_video_path.lower().endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif output_video_path.lower().endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        else:
            print(f"Warning: Unsupported output video format for '{output_video_path}'. Defaulting to .avi with XVID.")
            output_video_path = os.path.splitext(output_video_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        if fourcc:
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps_in, (frame_width_in, frame_height_in))
            if video_writer.isOpened():
                print(f"Output video will be saved to: {output_video_path}")
            else:
                print(f"Error: Could not open video writer for path: {output_video_path}")
                video_writer = None 

    print(f"Processing video: {video_path}. Press 'q' to quit.")
    cv2.namedWindow("Fire Detection", cv2.WINDOW_AUTOSIZE)

    frame_skip = 2
    frame_count = 0
    display_window_width, display_window_height = 800, 600 
    previous_pil_frame = None

    active_tracks = {} 
    next_track_id = 0
    IOU_THRESHOLD = 0.20     
    MAX_MISSES = 7          
    MIN_HITS_FOR_STABLE_TRACK = 3 

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame_count += 1
        current_processed_frame = frame_bgr.copy() 

        if frame_skip > 0 and frame_count % frame_skip != 0:
            resized_display_skipped = cv2.resize(current_processed_frame, (display_window_width, display_window_height))
            cv2.imshow("Fire Detection", resized_display_skipped)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        current_pil_image = Image.fromarray(frame_rgb)

        features = extract_image_features_for_testing(current_pil_image, previous_pil_frame, img_size=(128,128))
        previous_pil_frame = current_pil_image

        prediction_numeric = -1 
        if features.size > 0: 
            try:
                scaled_features = scaler.transform(features.reshape(1, -1))
                prediction_numeric = model.predict(scaled_features)[0]
            except ValueError as e:
                print(f"Error during scaling/prediction for frame {frame_count}: {e}")
        else:
            print(f"Warning: No features extracted for frame {frame_count}.")

        # Kalman Filter
        predicted_track_states_cxcywh = {}
        for track_id, track in list(active_tracks.items()):
            pred_state_full = track.predict()
            predicted_track_states_cxcywh[track_id] = (pred_state_full[0,0], pred_state_full[1,0], pred_state_full[2,0], pred_state_full[3,0])

        if prediction_numeric == fire_label_numeric:
            label_text = "Fire Detected in Frame"
            text_color = (0, 0, 255)

            detected_rois_xywh_raw = detect_fire_roi(frame_bgr)
            current_detections_cxcywh = [xywh_to_cxcywh(roi) for roi in detected_rois_xywh_raw]

            matched_track_ids_current_frame = set()
            matched_detection_indices_current_frame = set()
            possible_matches_iou = []
            if active_tracks and current_detections_cxcywh:
                for track_id, pred_cxcywh in predicted_track_states_cxcywh.items():
                    pred_xywh = cxcywh_to_xywh(pred_cxcywh)
                    if pred_xywh[2] <=0 or pred_xywh[3] <=0: continue
                    for det_idx, det_cxcywh in enumerate(current_detections_cxcywh):
                        det_xywh = cxcywh_to_xywh(det_cxcywh)
                        if det_xywh[2] <=0 or det_xywh[3] <=0: continue
                        iou = calculate_iou(pred_xywh, det_xywh)
                        if iou > IOU_THRESHOLD:
                            possible_matches_iou.append((iou, track_id, det_idx))
            
            possible_matches_iou.sort(key=lambda x: x[0], reverse=True)
            final_matches_this_frame = []
            for iou, track_id, det_idx in possible_matches_iou:
                if track_id not in matched_track_ids_current_frame and det_idx not in matched_detection_indices_current_frame:
                    final_matches_this_frame.append((track_id, det_idx))
                    matched_track_ids_current_frame.add(track_id)
                    matched_detection_indices_current_frame.add(det_idx)

            for track_id, det_idx in final_matches_this_frame:
                if track_id in active_tracks:
                    active_tracks[track_id].update(current_detections_cxcywh[det_idx])

            all_track_ids_before_update = set(active_tracks.keys())
            unmatched_track_ids_this_frame = all_track_ids_before_update - matched_track_ids_current_frame
            for track_id in unmatched_track_ids_this_frame:
                if track_id in active_tracks:
                    active_tracks[track_id].increment_misses()
                    if active_tracks[track_id].misses > MAX_MISSES:
                        print(f"Track {track_id}: Removed (misses {active_tracks[track_id].misses} > {MAX_MISSES})")
                        del active_tracks[track_id]
            
            all_detection_indices_this_frame = set(range(len(current_detections_cxcywh)))
            unmatched_detection_indices_this_frame = all_detection_indices_this_frame - matched_detection_indices_current_frame
            for det_idx in unmatched_detection_indices_this_frame:
                new_roi_cxcywh = current_detections_cxcywh[det_idx]
                if new_roi_cxcywh[2] > 10 and new_roi_cxcywh[3] > 10:
                    active_tracks[next_track_id] = KalmanTrack(new_roi_cxcywh, next_track_id)
                    print(f"Track {next_track_id}: Created at {new_roi_cxcywh[:2]}")
                    next_track_id += 1
            
            rois_drawn_this_frame = False
            for track_id, track in active_tracks.items():
                if track.hits >= MIN_HITS_FOR_STABLE_TRACK or track.misses == 0:
                    cx, cy, w, h = track.get_state_bbox_cxcywh()
                    if w > 0 and h > 0:
                        box_xywh = cxcywh_to_xywh((cx, cy, w, h))
                        x, y, w_draw, h_draw = int(box_xywh[0]), int(box_xywh[1]), int(box_xywh[2]), int(box_xywh[3])
                        box_color = (0,0,255)
                        if track.hits >= MIN_HITS_FOR_STABLE_TRACK:
                            box_color = (0,100,255)
                        cv2.rectangle(current_processed_frame, (x, y), (x + w_draw, y + h_draw), box_color, 2) 
                        cv2.putText(current_processed_frame, f"ID:{track.id}", (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                        rois_drawn_this_frame = True
            
            if rois_drawn_this_frame:
                cv2.putText(current_processed_frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            elif detected_rois_xywh_raw :
                 cv2.putText(current_processed_frame, label_text + " (Acquiring ROIs)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                 for x_r, y_r, w_r, h_r in detected_rois_xywh_raw:
                      cv2.rectangle(current_processed_frame, (x_r,y_r), (x_r+w_r, y_r+h_r), (255,192,203),1)
            else:
                cv2.putText(current_processed_frame, label_text + " (ROIs not localized)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        else:
            label_text = "No Fire in Frame"
            text_color = (0, 255, 0)
            cv2.putText(current_processed_frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            for track_id in list(active_tracks.keys()):
                if track_id in active_tracks:
                    active_tracks[track_id].increment_misses()
                    if active_tracks[track_id].misses > MAX_MISSES:
                        print(f"Track {track_id}: Removed (frame NoFire & misses {active_tracks[track_id].misses} > {MAX_MISSES})")
                        del active_tracks[track_id]

        if video_writer:
            video_writer.write(current_processed_frame)

        resized_display = cv2.resize(current_processed_frame, (display_window_width, display_window_height))
        cv2.imshow("Fire Detection", resized_display)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"Output video saved to {output_video_path}")
    cv2.destroyAllWindows()
    print("Video processing finished.")

# Menjalankan Program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fire detection model on a video (frame-level classification with ROI tracking).")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--model", default="fire_detection_model_without_optflow.pkl", help="Path to the trained model file.")
    parser.add_argument("--scaler", default="fire_detection_scaler_without_optflow.pkl", help="Path to the scaler file.")
    parser.add_argument("--output_video", help="Path to save the output video (e.g., output.mp4 or output.avi). Optional.") 

    args = parser.parse_args()

    if not os.path.exists(args.video_path): print(f"Error: Video path not found: {args.video_path}")
    elif not os.path.exists(args.model): print(f"Error: Model path not found: {args.model}")
    elif not os.path.exists(args.scaler): print(f"Error: Scaler path not found: {args.scaler}")
    else: main(args.video_path, args.model, args.scaler, args.output_video) 