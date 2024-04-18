import os
import h5py
from PIL import Image
import numpy as np


# Define the function to create the .h5 file with the images and masks in the correct order
def create_h5_dataset(base_path, image_size, keep_n):
    h5_file_path = os.path.join(base_path, 'utah_test_set.h5')
    with h5py.File('utah_test_set.h5', 'w') as h5file:
        image_paths = []
        prediction_paths = []
        groundtruth_paths = []

        for patient_dir in sorted(os.listdir(base_path)):
            data_path = os.path.join(base_path, patient_dir, 'data')
            auto_seg_path = os.path.join(base_path, patient_dir, 'auto segmentation')
            cavity_path = os.path.join(base_path, patient_dir, 'cavity')

            if os.path.exists(data_path) and os.path.exists(auto_seg_path) and os.path.exists(cavity_path):
                image_paths += sorted([os.path.join(data_path, img) for img in os.listdir(data_path)])
                prediction_paths += sorted([os.path.join(auto_seg_path, img) for img in os.listdir(auto_seg_path)])
                groundtruth_paths += sorted([os.path.join(cavity_path, img) for img in os.listdir(cavity_path)])

        if keep_n:
            image_paths = image_paths[30:46]
            prediction_paths = prediction_paths[30:46]
            groundtruth_paths = groundtruth_paths[30:46]

        images_dset = h5file.create_dataset('image', (len(image_paths), image_size, image_size), dtype='i8')
        predictions_dset = h5file.create_dataset('prediction', (len(prediction_paths), image_size, image_size), dtype='i8')
        groundtruth_dset = h5file.create_dataset('groundtruth', (len(groundtruth_paths), image_size, image_size), dtype='i8')

        def read_image(path):
            img = Image.open(path).convert('L')
            img = img.resize((image_size, image_size))
            return np.array(img)

        def process_image(img):
            return np.where(img >= 225, float(1), float(0)).astype('int64')

        for i, (img_path, pred_path, gt_path) in enumerate(zip(image_paths, prediction_paths, groundtruth_paths)):
            images_dset[i] = read_image(img_path)
            predictions_dset[i] = process_image(read_image(pred_path))
            groundtruth_dset[i] = process_image(read_image(gt_path))

    return

create_h5_dataset('UTAH Test set', image_size=640, keep_n=True)