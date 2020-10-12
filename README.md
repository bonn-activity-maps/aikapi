# AIKAPI
This project contains the API to load and interact with **Action In Kitchen** datasets. (WIP)

## Installation

1. Clone or download the repository
2. Install the required Python3 packages with: `pip3 install -r requirements.txt`
3. Place the AIK dataset folders at the same level as **PythonAPI** folder.

## AIK class functions
#### AIK(dataset_dir, ataset_name, image_format)
Constructor for the AIK class used to perform all actions over the dataset.
- **datset_dir**: (*String*) path to the folder containing the datasets.
- **dataset_name**: (*String*) name of the dataset folder.
- **image_format**: (*String*) format wanted for the image extraction. "png" (default)/"jpeg".
___
#### unroll_videos(force, video)
Unrolls (converts each frame to an individial image) all the videos from the dataset and stores them in the `videos` folder. This folder will be created if it not exists and will contain another 12 folders (cameraXX), each one containing the unrolled frames for the camera noted with XX.
- **force**: (*Bool*) if `True`, the already unrolled frames will be deleted (if the exist) and the unroll will be performed.
- **video**: (*Integer*) if `None` all videos will be unrolled. If it has a value between 1 and 12, only that camera will be unrolled.

**Note**: if *video* is `None` and *force* is `True` all the already unrolled videos will be deleted and unrolled again.
___
#### get_camera(video, frame)
Obtains the camera object for the specified video and frame.
- **video**: (*Integer*) camera/video number to get the object from.
- **frame**: (*Integer*) frame number to get the object from.
- **Returns**: (*Camera*) camera object (will be explained in the next section)
___
#### get_persons_in_frame(frame)
Get all `persons` annotated in the given frame.
- **frame**: (*Integer*) frame number to get the persons from.
- **Returns**: (*JSON*) persons annotated in the given frame in JSON format (for the structure of the JSON see section **JSON structures**)

## Camera class functions
WIP

## JSON structures
WIP

## Examples
For examples of how to use the API you can check the Jupyter Notebook [`test_api.ipynb`](https://github.com/bonn-activity-maps/aikapi/blob/master/test_api.ipynb) provided inside the repository.
