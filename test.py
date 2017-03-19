#!/usr/bin/env python

import numpy as np
import time
import load_data as ld

EOS = 28

def to_mask(data, thresh):
    return np.concatenate([np.array([[True]] * data.shape[0]), data[:,:-1]!=thresh], axis=1)

print "Starting to test..."
start_time0 = time.time()
for i in range(10):
    start_time1 = time.time()
    print "Epoch",i
    for b in range(10):
        start_time = time.time()
        f = "/mnt/Data/data_debug_5000_b" + str(b) + "_nml.npz"
        data, max_time, max_chars = ld.load_data(f)
        print "Took {:.3f} sec to read in data".format(time.time() - start_time)
        start_time = time.time()
        inputs, labels, seqs = data
        labels = to_mask(labels, EOS)
        print "Took {:.3f} sec to create mask".format(time.time() - start_time)
        assert max_time == 800
        assert max_chars == 98
        assert inputs.shape == (500, 800, 40)
        assert labels.shape == (500, 100)
    print "Took {:.3f} sec to process 10 files".format(time.time() - start_time1)
print "Took {:.3f} sec in total to run 10 epochs".format(time.time() - start_time0)
