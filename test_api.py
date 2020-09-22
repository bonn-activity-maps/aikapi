from PythonAPI.aik import AIK


dataset_dir = '/home/beatriz/Documentos/Work'   # For Bea
# dataset_dir = '/home/almartmen/Github/aikapi'   # For Alberto
dataset_name = '181129'

aik = AIK(dataset_dir, dataset_name, image_format='png')

# print(aik.get_calibration_params(3, 25894))
# print(aik.get_persons_in_frame(50))
# print(aik.get_person_in_frame(380, 1))
# print(aik.get_images_in_frame(1))
print(aik.get_activities_for_person(2))