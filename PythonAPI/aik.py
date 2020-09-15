# Interface for accessing the AIK dataset.

import os
import subprocess
import json
import numpy as np

class AIK:

    def __init__(self, dataset_dir, dataset_name, image_format='png'):
        '''
        Constructor AIK class for reading and visualizing annotations.
        :param dataset_dir (str): location of the dataset
        :param dataset_name (str): name of the dataset
        :param image_format (str): format of images to extract video frames, png or jpeg (png by default)
        '''
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, self.dataset_name)
        self.cameras_dir = os.path.join(self.dataset_dir, 'cameras')
        self.videos_dir = os.path.join(self.dataset_dir, 'videos')

        self.num_cameras = 12
        self.frame_format = "frame%09d"

        # Check if dataset exists in the directory
        assert os.path.isdir(self.dataset_dir), "The dataset directory %r does not exist" % dataset_dir
        assert os.path.isdir(self.cameras_dir), "The dataset is not complete"

        assert image_format in ['png', 'jpeg'], "The image format should be png or jpeg"

        # Create videos directory and unroll videos if the directory videos does not exist
        if not os.path.exists(self.videos_dir):
            os.mkdir(self.videos_dir)
            self.unroll_videos(image_format)

        # Load info to memory
        self.calibration_params = self.read_calibration_params()
        # self.persons, objects, actions = self.read_annotations()

    def unroll_videos(self, img_format):
        '''
        Unroll all the videos and store them in the videos folder. This folder contains 12 folders (cameraXX) with
        the unrolled frames from each camera.
        :param img_format (str): format of images to extract video frames, png or jpeg
        :return:
        '''
        print('Unrolling videos. This may take a while...')

        for c in range(self.num_cameras):
            video = self.dataset_name + '_' + str(c).zfill(2) + '.mp4'

            # Create camera directory to store all frames
            camera = 'camera' + str(c).zfill(2)
            camera_dir = os.path.join(self.videos_dir, camera)
            os.mkdir(camera_dir)

            unroll = subprocess.run(["ffmpeg", "-i", os.path.join(self.dataset_dir, video),
                                     os.path.join(camera_dir, self.frame_format+"."+img_format)])
            # print("The exit code was: %d" % unroll.returncode)

    def read_calibration_params(self):
        '''
        Read camera calibration parameters for each camera.
        :return: Array with all calibration parameters
        '''
        print('Loading calibration parameters...')
        cameras_data = []

        for c in range(self.num_cameras):
            camera = 'camera' + str(c).zfill(2) + '.json'
            print('    ', camera+'...')
            with open(os.path.join(self.cameras_dir, camera)) as f:
                data = json.load(f)

            # Store data for each frame in numpy array
            camera_params = np.empty(0)
            for d in data:
                frames = d['end_frame'] - d['start_frame']
                del d['start_frame']
                del d['end_frame']
                cam = np.full(frames, d)
                camera_params = np.append(camera_params, cam, axis=0)

            cameras_data.append(camera_params)
        return np.array(cameras_data)

    def read_annotations(self):
        '''
        Read persons, objects and actions information from file.
        :return: Persons, objects and actions in json format
        '''
        print('Loading annotations...')
        persons = []
        objects = []
        actions = []
        with open(os.path.join(self.dataset_dir, self.dataset_name + '_unroll.json')) as f:
            json_data = json.load(f)

        # Separate data into persons, objects and actions
        # Convert into a list ordered by frame
        for d in json_data['persons']:
            # print(d['frame'])
            # print('_________________________________')
            persons.append(d['frame'])

        for d in json_data['objects']:
            objects.append(d)

        # TODO: check how to process actions
        for d in json_data['actions']:
            actions.append(d)
        return persons, objects, actions

    def get_calibration_params(self, video, frame):
        '''
        Get calibration parameters that satisfy given filter conditions.
        :param video (int): video number
        :param frame (int): frame number
        :return: calibration parameters in json format with K, rvec, tvec, distCoef, w and h
        '''
        return self.calibration_params[video, frame]

    def get_persons_in_frame(self, frame):
        '''
        Get all persons annotated in given frame.
        :param frame (int): frame number
        :return: Persons in json format
        '''
        return self.persons[frame]





