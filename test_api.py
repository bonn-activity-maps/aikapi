from PythonAPI.aik import AIK


dataset_dir = ''
dataset_name = 'aik_dummy'

aik = AIK(dataset_dir, dataset_name)

print(aik.get_calibration_params(0, 0))
