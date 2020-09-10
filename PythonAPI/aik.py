# Interface for accessing the AIK dataset.

import os
import json

class AIK:

    def __init__(self, dataset_dir, dataset_name, image_format='png'):
        '''
        Constructor AIK class for reading and visualizing annotations.
        :param dataset_dir (str): location of the dataset
        :param dataset_name (str): name of the dataset
        :param image_format (str): format of images to extract video frames, png or jpeg
        :return:
        '''
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.cameras_dir = os.path.join(self.dataset_dir, 'cameras')
        self.videos_dir = os.path.join(self.dataset_dir, 'videos')

        self.num_cameras = 12

        # Check if dataset exists in the directory
        assert os.path.isdir(self.dataset_dir), "The dataset directory %r does not exist" % dataset_dir
        assert os.path.isdir(self.cameras_dir), "The dataset is not complete"

        # Create videos directory if it does not exist
        if not os.path.exists(self.videos_dir):
            os.mkdir(self.videos_dir)

        assert image_format in ['png', 'jpeg'], "The image format should be png or jpeg"

        # Unroll videos into videos directory TODO: if they are not already unrolled
        self.unroll_videos(image_format)
        print('alles gut')

        # Load info to memory
        self.calibration_params = self.read_calibration_params()

    def unroll_videos(self, img_format):
        '''
        Unroll all the videos and store them in the videos folder. This folder contains 12 folders (cameraXX) with
        the unrolled frames from each camera.
        :param img_format (str): format of images to extract video frames, png or jpeg
        :return:
        '''
        pass

    def read_calibration_params(self):
        '''
        Read camera calibration parameters for each camera.
        :return: Array with all calibration parameters
        '''
        data = []
        for c in range(self.num_cameras):
            camera = 'camera' + str(c).zfill(2) + '.json'
            with open(os.path.join(self.cameras_dir, camera)) as f:
                data.append(json.load(f))
        return data

    def get_calibration_params(self, video, frame):
        '''
        Get calibration parameters that satisfy given filter conditions.
        :param video (int): video number
        :param frame (int): frame number
        :return: calibration parameters in json format with K, rvec, tvec, distCoef, w and h
        '''
        return self.calibration_params[video]





