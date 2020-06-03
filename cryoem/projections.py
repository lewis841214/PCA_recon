
"""""
@author: fangshu.yang@epfl.ch, laurene.donati@epfl.ch 
"""

import numpy as np
import random
import os, sys
import scipy.io as sio
sys.path.append(os.getcwd())
import time
import mrcfile
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import astra
import pathlib
import h5py
from cryoem.rotation_matrices import RotationMatrix



def project_volume(Vol, Angles, Vol_geom, ProjSize):
	# Generate orientation vectors based on angles
	Orientation_Vectors   = RotationMatrix(Angles)
	
	# Create projection 2D geometry in ASTRA
	Proj_geom = astra.create_proj_geom('parallel3d_vec', ProjSize, ProjSize, Orientation_Vectors)
	
	# Generate projs 
	_, Proj_data = astra.create_sino3d_gpu(Vol, Proj_geom, Vol_geom)
	
	# Reshape projections correctly 
	Projections = np.transpose(Proj_data, (1, 0, 2))
	
	return  Projections

 
def gen_projs_ASTRA(Vol, AngCoverage, AngShift, ProjSize, BatchSizeAstra): 
		
	# Create 3D geometry in ASTRA
	Vol_geom    = astra.create_vol_geom(Vol.shape[1], Vol.shape[2], Vol.shape[0])
	
	# Generate random angles
	Z1 =  AngShift[0]*np.pi + AngCoverage[0]*np.pi*np.random.random(size=(BatchSizeAstra, 1))
	Y2 =  AngShift[1]*np.pi + AngCoverage[1]*np.pi*np.random.random(size=(BatchSizeAstra, 1))
	Z3 =  AngShift[2]*np.pi + AngCoverage[2]*np.pi*np.random.random(size=(BatchSizeAstra, 1))

	Angles = np.concatenate((Z1, Y2, Z3), axis=1)
	
	# Generate projections
	Projections = project_volume(Vol, Angles, Vol_geom, ProjSize)
	
	return Projections, Angles


def generate_2D_projections(input_file_path, ProjNber, AngCoverage, AngShift, output_file_name=None):
	"""
	input_file_path: str
		Full path to the *.mrc file with 3D volume
	ProjNber: int
		Number of 2D projections 
	AngCoverage: list
		list of max values for each axis. E.g. `0.5,0.5,2.0` means it: x axis angle and y axis angle take values in range [0, 0.5*pi], z axis angles in range [0, 2.0*pi]
	AngShift: list
		Start of angular coverage
	output_file_name: str
		Just the name of the output *.mat file. 
		If not specified, it will be generated automatically.
	"""

	# nber of projs created in a single ASTRA loop
	BatchSizeAstra = 50 
	
	# filepaths 
	protein_name = input_file_path.split('/')[-1].split('.')[0]
	coverage_str = str(AngCoverage).replace(" ", "")[1:-1]
	shift_str    = str(AngShift).replace(" ", "")[1:-1]
	output_file_name = output_file_name or f'{protein_name}_ProjectionsAngles_ProjNber{ProjNber}_AngCoverage{coverage_str}_AngShift{shift_str}.h5'

	# get file extension
	extension = output_file_name.split('.')[-1]
	# storing output where the input mrc file is
	proj_ang_path = os.path.join(os.path.dirname(input_file_path), output_file_name)
	
	# loads data if data already exists 
	if os.path.exists(proj_ang_path):
		print('* Loading the dataset *\n')
	
		# read from the file
		if extension == "h5":
			with h5py.File(proj_ang_path, 'r') as data:
				Projections = np.float32(data['Projections'])
				Angles      = np.float32(data['Angles'])
		elif extension == "mat":
			with sio.loadmat(proj_ang_path) as data:
				Projections = np.float32(data['Projections'])
				Angles      = np.float32(data['Angles'])
		else: 
			raise NotImplementedError(f"Extension {extension} is not implemented")

	# generate data if data doesn't exist  
	else:
		print('* Generating the dataset *\n')
	
		# Load 3D volume
		# Value error fix explained here: https://mrcfile.readthedocs.io/en/latest/usage_guide.html 
		try:
			with mrcfile.open(input_file_path) as mrcVol:
				Vol      = np.array(mrcVol.data) 
				ProjSize = int(np.sqrt(np.sum(np.square(Vol.shape))))
		except ValueError:
			with mrcfile.open(input_file_path, mode='r+', permissive=True) as mrcVol:
				mrcVol.header.map = mrcfile.constants.MAP_ID
				Vol      = np.array(mrcVol.data) 
				ProjSize = int(np.sqrt(np.sum(np.square(Vol.shape))))
		
		# Initialisations 
		Projections = np.zeros((ProjNber, ProjSize, ProjSize), dtype=float)
		Angles      = np.zeros((ProjNber, 3), dtype=float)
		
		# Generate projs with ASTRA by batches 
		Iter = int(ProjNber/BatchSizeAstra) 
		for i in range(Iter):
			
			# Generate projections 
			projections, angles = gen_projs_ASTRA(Vol, AngCoverage, AngShift, ProjSize, BatchSizeAstra)
			
			# Concatenate generated projections 
			Projections[i*BatchSizeAstra : (i + 1)*BatchSizeAstra, :, :] = projections
			Angles[i*BatchSizeAstra : (i + 1)*BatchSizeAstra, :] = angles  
			
		# Save data 
		if extension == "h5":
			with h5py.File(proj_ang_path, 'w') as hf:
				hf.create_dataset('Projections', data=Projections)
				hf.create_dataset('Angles', data=Angles)
		elif extension == "mat":
			sio.savemat(proj_ang_path, {'Projections': Projections, 
										'Angles': Angles}) 
		else: 
			raise NotImplementedError(f"Extension {extension} is not implemented")
		
	print(f'Projections: {Projections.shape}')
	print(f'Angles: {Angles.shape}\n')
