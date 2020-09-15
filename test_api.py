from PythonAPI.aik import AIK


dataset_dir = '/home/beatriz/Documentos/Work'
dataset_name = '181129'

aik = AIK(dataset_dir, dataset_name)

print(aik.get_calibration_params(3, 25894))
# print(aik.get_persons_in_frame(0))
