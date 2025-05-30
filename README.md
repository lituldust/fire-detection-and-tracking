# Fire Detection and Tracking
This repository is our documentation for final project in Pattern Recognition course. 

## How to Run The Code Locally
### Clone this repository
```bash
git clone (this repo url)
```

### Set up Virtual Environment
```bash
python -m venv .venv
.venv/Scripts/activate
```

### Install Requirement Dependencies
```bash
pip install -r requirements.txt
```

### Run the Code
```bash
python test_fire_model.py test_video.mp4
```

### If You Want to Save the Output Video, then
```bash
python test_fire_model.py test_video.mp4 --output_video 'name_of_your_output_video.mp4/avi'
```

## Notes
You Can Choose to Use Model with Optical Flow or Not
### If You Want to Test with Optical Flow
#### 1. Uncomment this line of code
```bash
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
```
#### 2. Change expected features (line 218)
```bash
expected_features_this_script = 120
```

#### 3. Change default model and scaler in __name__ = "main"
```bash
parser.add_argument("--model", default="fire_detection_model.pkl", help="Path to the trained model file.")
parser.add_argument("--scaler", default="fire_detection_scaler_120feat.pkl", help="Path to the scaler file.")
```

#### 4. Run the code

### If You Want to Test withou Optical Flow
#### 1. Keep this line of code commented
```bash
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
```
#### 2. Change expected features (line 218)
```bash
expected_features_this_script = 112
```

#### 3. Change default model and scaler in __name__ = "main"
```bash
parser.add_argument("--model", default="fire_detection_model_without_optflow.pkl", help="Path to the trained model file.")
parser.add_argument("--scaler", default="fire_detection_scaler_without_optflow.pkl", help="Path to the scaler file.")
```

#### 4. Run the code