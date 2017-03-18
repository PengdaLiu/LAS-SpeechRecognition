import os, sys
import numpy as np
import random
import scipy.io.wavfile as wav

CODE = {'A': 1, 'C': 3, 'B': 2, 'E': 5, 'D': 4, 'G': 7, 'F': 6, 'I': 9, 'H': 8, 'K': 11, 'J': 10, 'M': 13, \
        'L': 12, 'O': 15, 'N': 14, 'Q': 17, 'P': 16, 'S': 19, 'R': 18, 'U': 21, 'T': 20, 'W': 23, 'V': 22, \
        'Y': 25, 'X': 24, 'Z': 26, ' ': 27}

def nml_data(f, old_format=False):
    def nml(trd, trs):
        ntrd=np.zeros(trd.shape)
        for i in range(len(trs)):
            tmp=trd[i,:trs[i],:]
            mean_scale = np.mean(tmp, axis=0)
            std_scale = np.std(tmp, axis=0)
            ntrd[i,:trs[i],:] = (tmp-mean_scale[np.newaxis,:])/std_scale[np.newaxis,:]
        return ntrd
    dt=np.load(f)
    trd=dt["train_data"]
    trl=dt["train_labels"]
    trs=dt["train_seq_len"]
    mt=dt["max_time"]
    mc=dt["max_chars"]
    if old_format:
        tsd=dt["test_data"]
        tsl=dt["test_labels"]
        tss=dt["test_seq_len"]
        np.savez(f[:-4]+"_nml"+f[-4:], train_data=nml(trd, trs), train_labels=trl, train_seq_len=trs, \
                test_data=nml(tsd, tss), test_labels=trl, test_seq_len=tss, \
                max_time=mt, max_chars=mc)
        return
    trl[trl==32]=28
    np.savez(f[:-4]+"_nml"+f[-4:], train_data=nml(trd, trs), train_labels=trl, train_seq_len=trs, \
            max_time=mt, max_chars=mc)


def shuffle_data(f1="data_b0.npz", f2="data_b7.npz"):
    print "[INFO] Loading data ..."
    d1 = np.load(f1)
    d2 = np.load(f2)
    t1d = d1['train_data']
    t1l = d1['train_labels']
    t1s = d1['train_seq_len']
    t2d = d2['train_data']
    t2l = d2['train_labels']
    t2s = d2['train_seq_len']
    mt = d1['max_time']
    mc = d1['max_chars']
    del d1
    del d2
    l1 = len(t1d)
    l2 = len(t2d)
    print "[INFO] Shuffling data ..."
    t1d = np.concatenate([t1d, t2d], axis=0)
    t1l = np.concatenate([t1l, t2l], axis=0)
    t1s = np.concatenate([t1s, t2s], axis=0)
    samples1 = random.sample(xrange(l1 + l2), l1)
    samples2 = list(set(range(l1 + l2)) - set(samples1))
    t2d = t1d[samples2]
    t2l = t1l[samples2]
    t2s = t1s[samples2]
    t1d = t1d[samples1]
    t1l = t1l[samples1]
    t1s = t1s[samples1]
    print "[INFO] Writing new files ..."
    np.savez(f1, train_data=t1d, train_labels=t1l, train_seq_len=t1s, max_time=mt, max_chars=mc)
    np.savez(f2, train_data=t2d, train_labels=t2l, train_seq_len=t2s, max_time=mt, max_chars=mc)
    print "[INFO] Done"

def load_data(file="data.npz", new_format=True):
    data = np.load(file)
    train_data = data['train_data']
    train_labels = data['train_labels']
    train_seq_len = data['train_seq_len']
    if not new_format:
        test_data = data['test_data']
        test_labels = data['test_labels']
        test_seq_len = data['test_seq_len']
    max_time = int(data['max_time'])
    max_chars = int(data['max_chars'])
    if new_format: return (train_data, train_labels, train_seq_len), max_time, max_chars
    return (train_data, train_labels, train_seq_len), (test_data, test_labels, test_seq_len), max_time, max_chars

def build_data(vf_path="./voxforge/", timit_path="./timit/", n_mfcc=40, \
        debug=False, ret=False, N_TRAINS_LIM_DBG=200):
    import librosa
    """load train and test data

    :param vf_path - path to the directory that contains directories, each containing
                     each containing dirs etc/ and wav/, which contains PROMPT and *.wav
                     files
    :param timit_path - path to the directory that contains two subdirs train/ and test/,
                        each containing dirs drX/<hash>/, which contains *(C).wav and *.txt
                        files
    :param n_mfcc - dimension of MFCC features
    :param path - if true, load a small portion
    :param N_TRAINS_LIM_DBG - number of samples used for debug

    :return - train (data, labels, seq_len), test (data, labels, seq_len), max_time

    Data are padded to match length of max_time, with size (batch_size, max_time, n_mfcc).
    Labels and seq_len are list of length batch_size.
    """
    if vf_path == "": vf_path = "./"
    if vf_path[-1] != '/': vf_path += '/'
    if timit_path == "": timit_path = "./"
    if timit_path[-1] != '/': timit_path += '/'
    train_dir = timit_path + "train/"

    hop_length = 80

    train = []
    max_time = 800
    min_time = 100
    max_chars = 98
    min_chars = 5
    n_trains = 0
    batch_size = 12000
    batch_no = 0

    def write_data(train, batch_no, debug, max_time, n_mfcc, N_TRAINS_LIM_DBG):
        random.shuffle(train)
        print "[INFO] Shuffled data"
        train_data = np.concatenate([np.lib.pad(t[0].T, \
                    ((0, max_time - t[0].shape[1]), (0, 0)), \
                    'constant', constant_values=0).reshape((1, max_time, n_mfcc)) for t in train], \
                axis=0)
        print "[INFO] Done generating data"
        train_label = np.concatenate([np.lib.pad([0] + map(lambda l : CODE[l], t[1]) + [28], \
                        (0, max_chars - len(t[1])), \
                        'constant', constant_values=28).reshape((1, max_chars + 2)) for t in train], \
                    axis=0)
        print "[INFO] Done generating labels"
        train_seq_len = [t[2] for t in train]
        np.savez(("data_debug_" + str(N_TRAINS_LIM_DBG) if debug else "data") + "_b" + str(batch_no), \
                train_data=train_data, train_labels=train_label, train_seq_len=train_seq_len, \
                max_time=max_time, max_chars=max_chars)
        print "[INFO] Done writing data file " + ("data_debug_" + str(N_TRAINS_LIM_DBG) if debug else "data") + "_b" + str(batch_no) + ".npz"

    print "[INFO] Loading TIMIT data from", train_dir
    # load timit data
    for d_l1 in os.listdir(train_dir):
        if not os.path.isdir(train_dir + d_l1 + '/'): continue
        if debug and n_trains >= N_TRAINS_LIM_DBG / 2: break
        for d_l2 in os.listdir(train_dir + d_l1 + '/'):
            if not os.path.isdir(train_dir + d_l1 + '/' + d_l2 + '/'): continue
            if debug and n_trains >= N_TRAINS_LIM_DBG / 2: break
            for cln_wav in filter(lambda s : s[-5:] == "C.wav", \
                    os.listdir(train_dir + d_l1 + '/' + d_l2 + '/'), ):
                if not os.path.isfile(train_dir + d_l1 + '/' + d_l2 + '/' + cln_wav[:-5] + ".txt"):
                    continue
                rate, sig = wav.read(train_dir + d_l1 + '/' + d_l2 + '/' + cln_wav)
                mfccs = librosa.feature.mfcc(sig, sr=rate, n_mfcc=n_mfcc, hop_length=hop_length)
                n_ts = mfccs.shape[1]
                if n_ts > max_time or n_ts < min_time: continue
                label_file = open(train_dir + d_l1 + '/' + d_l2 + '/' + cln_wav[:-5] + ".txt", 'r')
                label = "".join(filter(CODE.__contains__, \
                                " ".join(label_file.readlines()[0].strip().split(' ')[2:]).upper()))
                label_file.close()
                if len(label) > max_chars or len(label) < min_chars: continue
                train.append([mfccs, label, n_ts])
                n_trains += 1
                if n_trains % 1000 == 0:
                    print "    Read", n_trains, "data ..."
                    if n_trains % batch_size == 0:
                        write_data(train, batch_no, debug, max_time, n_mfcc, N_TRAINS_LIM_DBG)
                        batch_no += 1
                        train = []
    print "[INFO] Done loading", n_trains, "training data"

    print "[INFO] Loading VoxForge data from", vf_path
    # load voxforge data
    for d_l1 in os.listdir(vf_path):
        if debug and n_trains >= N_TRAINS_LIM_DBG: break
        if not os.path.isfile(vf_path + d_l1 + '/etc/PROMPTS'): continue
        p_f = open(vf_path + d_l1 + '/etc/PROMPTS')
        lines = [l.strip().split(' ') for l in p_f]
        prompts = {l[0].split('/')[-1]: " ".join(l[1:]) for l in lines}
        p_f.close()
        for l in lines:
            if not os.path.isfile(vf_path + d_l1 + "/wav/" + l[0].split('/')[-1] + ".wav"):
                continue
            if debug and n_trains >= N_TRAINS_LIM_DBG: break
            rate, sig = wav.read(vf_path + d_l1 + "/wav/" + l[0].split('/')[-1] + ".wav")
            mfccs = librosa.feature.mfcc(sig, sr=rate, n_mfcc=n_mfcc, hop_length=hop_length)
            n_ts = mfccs.shape[1]
            if n_ts > max_time or n_ts < min_time: continue
            label = "".join(filter(CODE.__contains__, " ".join(l[1:]).upper()))
            if len(label) > max_chars or len(label) < min_chars: continue
            train.append([mfccs, label, n_ts])
            n_trains += 1
            if n_trains % 1000 == 0:
                print "    Read", n_trains, "data ..."
                if n_trains % batch_size == 0:
                    write_data(train, batch_no, debug, max_time, n_mfcc, N_TRAINS_LIM_DBG)
                    batch_no += 1
                    train = []

    if n_trains % batch_size != 0:
        write_data(train, batch_no, debug, max_time, n_mfcc, N_TRAINS_LIM_DBG)
        batch_no += 1
        train = []
    print "[INFO] Done loading", n_trains, "(total) training data"
