import pandas as pd
import numpy as np
import os

def process_csv_and_create_labels(csv_file, output_dir):
    """
    Processes a CSV file containing keypoint data, calculates bounding boxes,
    and creates YOLO pose estimation label files.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Get the image frame IDs (assuming the first column is the frame number/ID)
    frame_ids = df.iloc[:, 0]
    
    # Get the keypoint data (all columns after the frame ID)
    keypoints_df = df.iloc[:, 1:]
    
    # Reshape the keypoint data into (x, y) pairs
    num_keypoints_per_row = keypoints_df.shape[1]
    
    # Validate that the number of columns is even and consistent
    if num_keypoints_per_row % 2 != 0:
        raise ValueError("The number of keypoint columns is odd. Each keypoint must have an x and y coordinate.")
        
    num_keypoints = int(num_keypoints_per_row / 2)
    keypoints = keypoints_df.values.reshape(-1, num_keypoints, 2)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each row (image) to create a label file
    for i in range(len(frame_ids)):
        frame_id = frame_ids.iloc[i]
        
        # Assume a single class index of 0
        class_index = 0
        
        # Get keypoints for the current frame
        current_keypoints = keypoints[i, :, :]
        
        # Calculate bounding box from keypoints
        min_x = np.min(current_keypoints[:, 0])
        max_x = np.max(current_keypoints[:, 0])
        min_y = np.min(current_keypoints[:, 1])
        max_y = np.max(current_keypoints[:, 1])
        
        bbox_x_center = (min_x + max_x) / 2
        bbox_y_center = (min_y + max_y) / 2
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        normalized_bbox_x = (bbox_x_center - 100) / 50
        normalized_bbox_y = (bbox_y_center - 100) / 50
        normalized_bbox_w = (bbox_width - 100) / 50
        normalized_bbox_h = (bbox_height - 100) / 50
        
        # Create the label line
        label_line = [str(class_index),
                      str(normalized_bbox_x),
                      str(normalized_bbox_y),
                      str(normalized_bbox_w),
                      str(normalized_bbox_h)]
        
        # Append the normalized keypoints with a visibility flag of 2 (visible)
        for kpt in current_keypoints:
            normalized_kpt_x = (kpt[0] - 100) / 50
            normalized_kpt_y = (kpt[1] - 100) / 50
            label_line.append(str(normalized_kpt_x))
            label_line.append(str(normalized_kpt_y))
            # label_line.append('2') # Visibility flag
            
        # Write the label file
        label_filename = os.path.join(output_dir, f'{frame_id}.txt')
        with open(label_filename, 'w') as f:
            f.write(' '.join(label_line))
            
    return num_keypoints

# Main script execution

# Process training data
train_labels_dir = 'labels/train'
train_keypoints_count = process_csv_and_create_labels('training_frames_keypoints.csv', train_labels_dir)

# Process test data (val set for YOLO)
val_labels_dir = 'labels/val'
val_keypoints_count = process_csv_and_create_labels('test_frames_keypoints.csv', val_labels_dir)

# Create the data.yaml file
data_yaml_content = f"""
# Custom dataset settings for YOLOv8 Pose Estimation

# Path to the dataset directory
path: data
train: images/train
val: images/test

# Class information for bounding box detection
nc: 1
names: ['person']

# Keypoint information
kpt_shape: [68, 2]
kpt_names: ['point_0', 'point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10', 'point_11', 'point_12', 'point_13', 'point_14', 'point_15', 'point_16', 'point_17', 'point_18', 'point_19', 'point_20', 'point_21', 'point_22', 'point_23', 'point_24', 'point_25', 'point_26', 'point_27', 'point_28', 'point_29', 'point_30', 'point_31', 'point_32', 'point_33', 'point_34', 'point_35', 'point_36', 'point_37', 'point_38', 'point_39', 'point_40', 'point_41', 'point_42', 'point_43', 'point_44', 'point_45', 'point_46', 'point_47', 'point_48', 'point_49', 'point_50', 'point_51', 'point_52', 'point_53', 'point_54', 'point_55', 'point_56', 'point_57', 'point_58', 'point_59', 'point_60', 'point_61', 'point_62', 'point_63', 'point_64', 'point_65', 'point_66', 'point_67']
skeleton: []

"""

# Save the data.yaml file
with open('data.yaml', 'w') as f:
    f.write(data_yaml_content)
    
print(f"Created {len(os.listdir(train_labels_dir))} training label files in the 'labels/train' directory.")
print(f"Created {len(os.listdir(val_labels_dir))} validation label files in the 'labels/val' directory.")
print("Created the 'data.yaml' file for training configuration.")