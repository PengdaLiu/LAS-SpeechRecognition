import numpy as np
import os, sys
import matplotlib
#from autocorrect import spell
import matplotlib.pyplot as plt

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

def tostr(lbls):
    return "".join([CHARS[lbls[i] - 1] for i in range(1, lbls.tolist().index(28) if 28 in lbls else len(lbls))])

def predict(pred):
    return np.argmax(pred, axis=0)

def predict_raw(pred):
    p_int = np.argmax(pred, axis=0)
    return tostr(p_int[(p_int.tolist().index(0) if 0 in p_int else None):(p_int.tolist().index(28) if 28 in p_int else None)])

"""
autocorrect.spell is not included
def predict_str(pred):
    raw = predict_raw(pred).split(" ")
    chk = [spell(w).upper() for w in raw if w != ""]
    return " ".join(chk)
"""

def levenshteinDistanceWord(s1, s2):
    """Edit distance of two lists of words

    http://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def levenshteinDistance(s1, s2):
    """Edit distance of two lists of words

    http://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1==c2:#levenshteinDistanceWord(c1, c2) <= 1:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def same(p, l):
    sm = np.zeros(100)
    sm[:l.tolist().index(28) + 1] = [1 if len(p)>i and p[i]==l[i] else 0 for i in range(l.tolist().index(28) + 1)]
    return sm

def diff(p, l):
    df = np.zeros(100)
    df[:l.tolist().index(28) + 1] = [1 if len(p)<=i or p[i]!=l[i] else 0 for i in range(l.tolist().index(28) + 1)]
    return df

def cmp_pos(ps, ls):
    sm = np.zeros(100)
    df = np.zeros(100)
    for pred, l in zip(ps, ls):
        p = predict(pred)
        sm += same(p, l)
        df += diff(p, l)
    return sm, sm + df

def tokens(ws):
    tks = []
    for w in ws:
        tks.extend([w, " "])
    return tks

def wer(p, l):
    truewords = tokens(tostr(l).split(" "))
    predwords = tokens([w for w in predict_raw(p).split(" ") if len(w) > 0])
    return min(levenshteinDistance(truewords, predwords), len(truewords)), len(truewords)

def compute_wer(ps, ls):
    t_er, t_l = [], []
    for p, l in zip(ps, ls):
        er, length = wer(p, l)
        t_er.append(er)
        t_l.append(length)
    return float(sum(t_er)) / sum(t_l)

def heatmap(pred, lbls):
    fig, ax = plt.subplots()
    _ = ax.pcolor(pred, cmap=matplotlib.cm.Reds)
    _ = ax.set_xticks(np.arange(pred.shape[1])+0.5, minor=False)
    _ = ax.set_yticks(np.arange(pred.shape[0])+0.5, minor=False)
    rng = min(100, lbls.tolist().index(28))
    _ = ax.set_yticklabels(["<s>"]+list(CHARS)[:-1]+["<sp>","<e>"], minor=False)
    _ = ax.set_xticklabels([""]+list(tostr(lbls))+[""]+[""]*(rng-2-len(tostr(lbls))), minor=False)
    ax.yaxis.set_ticks_position('none') 
    ax.xaxis.set_ticks_position('none')
    #_ = ax.set_title("Prediction for sentence \"" + tostr(lbls)[:] +"\"")
    _ = ax.axis([0, rng + 10, 0, 29])
    ax.invert_yaxis()
    plt.show()

def set_plot(title, xlab, ylab):
    #plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def len_err(ps, ls):
    er = []
    leng = []
    for pred, l in zip(ps, ls):
        p = predict(pred)
        er.append(np.sum(diff(p, l)) / float(len(tostr(l))))
        leng.append(len(tostr(l)))
    fig, ax = plt.subplots()
    _ = ax.scatter(leng, er)
    #ax.set_title("Dependency of error rate on length of sentence")
    ax.axis([0,90,0,0.8])
    ax.set_xlabel("Length of sentence")
    ax.set_ylabel("Error rate")
    plt.show()
    return er, leng

def read_pred(filename):
    dt = np.load(filename)
    ps = dt["pred"].transpose((0, 2, 1))
    ls = dt["truevals"]
    return ps, ls

def get_WER(filename):
    ps, ls = read_pred(filename)
    print "WER on file " + filename + " is:", compute_wer(ps, ls)

def incorrect(p, l):
    ind = l.tolist().index(28) if 28 in l else 100
    pc = p[:ind].tolist() + [28] * (100 - ind)
    return [a!=b for a,b in zip(pc, l)]

def plot_correct(ps, ls):
    crct = []
    for p, l in zip(ps, ls):
        crct.append(incorrect(predict(p), l))
    fig, ax = plt.subplots()
    _ = ax.pcolor(np.array(crct), cmap=matplotlib.cm.Reds)
    bins = np.linspace(0, 100, 101)
    data = []
    for i, l in enumerate(ls):
        data += [i+0.5] * (len(tostr(l)) + 2)
    ax.hist(data, bins, orientation='horizontal', histtype='stepfilled', linewidth=0, fc=(0,0,0,0.2))
    #ax.set_title("Prediction error for 100 sentences")
    ax.axis([0,100,0,100])
    ax.set_xlabel("Character position")
    ax.set_ylabel("Sentence index")
    plt.show()

def analyze(filename):
    """Analyze prediction output"""
    dt = np.load(filename)
    ps = dt["pred"].transpose((0, 2, 1))
    ls = dt["truevals"]
    print "[ANALYTIC] WER on file " + filename + " is:", compute_wer(ps, ls)
    sm, ttl = cmp_pos(ps, ls) # sm[i] is no. times the i-th char is predicted correctly
                              # ttl[i] is no. times the i-th char is ever predicted (i.e. before eos)
    ind = ttl.tolist().index(0)-10 if 0 in ttl else None
    err = 1 - sm[:ind] / ttl[:ind]
    var = err * (1 - err)
    leng = err.shape[0]
    for i in range(leng):
        if ttl[i] == 1: var[i] = 1
        else: var[i] /= (ttl[i]-1)
    # more to add here: error bar
    stddev = np.sqrt(var)
    plt.errorbar(range(leng), err, yerr = stddev, fmt = 'b-')
    set_plot("Error rates at different indices in the sentence", "Index", "Error rate")
    plt.axis([0, ind, 0, 1])
    plt.show()
    len_err(ps, ls)
    plot_correct(ps, ls)
