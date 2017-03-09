from __future__ import print_function
pygt_path = '../src/PyGreentea'
import sys
sys.path.append(pygt_path)

from numpy import float32, int32, uint8, dtype
from os.path import join
import PyGreentea as pygt
import glob
import h5py
import json
import numpy as np
import os
import time
from data_io import OutOfCoreArray
from data_io import OffsettedArray
from data_io.out_of_core_arrays import H5PyArrayHandler

model_base_folder = '../02_train/'
output_base_folder = './processed/'

def collect(dataset, sample_dir, sample, augmentations, in_memory=False, chunk=None):

    for a in augmentations:

        augmentation_name = sample
        if a is not None:
            augmentation_name += '.augmented.' + str(a)

        filename = sample_dir + '/' + augmentation_name + '.hdf'
        print("Reading " + filename)

        raw_opener = H5PyArrayHandler(filename, 'volumes/raw', 'raw')
        raw = OutOfCoreArray(raw_opener)

        if chunk is not None:
            raw = OffsettedArray(raw, chunk['offset'], chunk['size'])
            augmentation_name += '_%s_%s'%(str(chunk['offset']),str(chunk['size']))

        dataset.append({})
        dataset[-1]['name'] = augmentation_name
        if in_memory:
            dataset[-1]['data'] = np.array(raw,dtype=np.float32)/(2**8)
            dataset[-1]['data'] = dataset[-1]['data'][None,:]
        else:
            dataset[-1]['data'] = raw

        print("Loaded augmentation " + str(a))

def process(modelpath,iter,outputpath,dset,normalize,divide,bigdata,test_device=2):
  print('>>>>> Processing in folder: ' + modelpath + ', iteration: '+str(iter))

  # Load model
  protosmall = modelpath + 'net_test.prototxt'
  protobig = modelpath + 'net_test_big.prototxt'
  if os.path.exists(protobig) and bigdata:
    proto=protobig
  elif os.path.exists(protosmall):
    proto=protosmall
  else:
    print('Error: can\'t find test proto')
    return
  model = modelpath + 'net_iter_'+str(iter)+'.caffemodel'
  if not os.path.exists(model):
    print('Iter doesn\'t exist, skipping ', model)
    return

  # paths
  if not os.path.exists(outputpath):
    os.makedirs(outputpath)
  h5files = [outputpath + d['name'] + '.hdf' for d in dset]
  alreadyProcessed = map(os.path.exists,h5files)
  if all(alreadyProcessed):
    print('Skipping ', h5files)
    return

  # Set devices
  print('Setting device. Using device: ' + str(test_device))
  pygt.caffe.set_mode_gpu()
  pygt.caffe.set_device(test_device)

  print('Loading proto: ' + proto)
  print('Loading model: ' + model)
  net = pygt.caffe.Net(proto, model, pygt.caffe.TEST)

  # Process
  print('Processing ' + str(len(dset)) + ' volumes...')
  for i in range(len(dset)):
    h5file = outputpath + dset[i]['name'] + '.hdf'
    if os.path.exists(h5file):
      print('Skipping ' + h5file)
      continue
    print('Processing to ' + h5file)
    preds = pygt.process(net,[dset[i]])
    print('Saving to ' + h5file)
    outhdf5 = h5py.File(h5file, 'w')
    outdset = outhdf5.create_dataset('main', preds[0].shape, np.float32, data=preds[0])
    outhdf5.close()

def prepare_dataset(data_dir, samples, augmentations, chunk=None):

    print("Preparing dataset...")

    dataset = []

    for sample in samples:
        collect(dataset, data_dir, sample, augmentations, in_memory=False, chunk=chunk)

    print("Dataset contains " + str(len(dataset)) + " volumes")
    for ds in dataset:
        print(ds['name'] + " shape:" + str(ds['data'].shape))

    return dataset

def is_testing_sample(sample):

    if '+' in sample:
        return True
    return False

# to be called by luigi
def predict_affinities(setup, iteration, sample, augmentation, gpu, orig_data_dir=None, chunk=None):

    start = time.time()
    if orig_data_dir is None:
        if is_testing_sample(sample):
            orig_data_dir = '../00_dataset_preparation/test/'
        else:
            with open(os.path.join('../02_train/', setup, 'train_options.json'), 'r') as f:
                orig_data_dir = json.load(f)['data_dir']
    test_dataset = prepare_dataset(orig_data_dir, [sample], [augmentation], chunk=chunk)
    print("Loaded data in " + str(time.time()-start) + "s")

    folder = setup
    modelpath = model_base_folder + folder + '/'
    try:
        outputpath = output_base_folder + folder + '/' + str(iteration) + '/'
        start = time.time()
        process(modelpath,iteration,outputpath,test_dataset,normalize=False,divide=True,bigdata=True,test_device=gpu)
        print("Processed iteration " + str(iteration) + " in " + str(time.time()-start) + "s")
    except SystemError, e:
        print(str(e))
