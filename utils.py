import os
import re
from time import time, sleep

import h5py
from  scipy.io import loadmat
import numpy as np
from PIL import Image
from cv2 import imread, resize, INTER_CUBIC, INTER_AREA

#%%
def read_image(image_fpath):
    # BGR is flipped to RGB. why BGR?:
    #     Note In the case of color images, the decoded images will have the channels stored in B G R order.
    #     https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    imarr = imread(image_fpath)[:, :, ::-1]
    return imarr


def write_images(imgs, names, path, size=None, timeout=0.5, format='bmp'):
    """
    Saves images as 24-bit bmp files to given path with given names
    :param imgs: list of images as numpy arrays with shape (w, h, c) and dtype uint8
    :param names: filenames of images **including or excluding** '.bmp'
    :param path: path to save to
    :param size: size (pixels) to resize image to; default is unchanged
    :param timeout: timeout for trying to write each image
    :return: None
    """
    for im_arr, name in zip(imgs, names):
        if size is not None and im_arr.shape[1] != size:
            if im_arr.shape[1] < size:    # upsampling
                im_arr = resize(im_arr, (size, size), interpolation=INTER_CUBIC)
            else:                         # downsampling
                im_arr = resize(im_arr, (size, size), interpolation=INTER_AREA)
        img = Image.fromarray(im_arr)
        trying = True
        t0 = time()
        if name.rfind("."+format ) != len(name) - 4:
            name += "."+format
        while trying and time() - t0 < timeout:
            try:
                img.save(os.path.join(path, name))
                trying = False
            except IOError as e:
                if e.errno != 35:
                    raise
                sleep(0.01)


def write_codes(codes, names, path, timeout=0.5):
    """
    Saves codes as npy files (1 in each file) to given path with given names
    :param codes: list of images as numpy arrays with shape (w, h, c) and dtype uint8. NOTE only thing in a .npy file is a single code.
    :param names: filenames of images, excluding extension. number of names should be paired with codes.
    :param path: path to save to
    :param timeout: timeout for trying to write each code
    :return: None
    """
    for name, code in zip(names, codes):
        trying = True
        t0 = time()
        while trying and time() - t0 < timeout:
            try:
                np.save(os.path.join(path, name), code, allow_pickle=False)
                trying = False
    #         File "/Users/wuxiao/Documents/MCO/Rotations/Kreiman Lab/scripts/Playtest6/utils.py", line
    #         56, in write_codes
    #         np.save(os.path.join(path, name), code, allow_pickle=False)
    #     File "/usr/local/lib/python3.6/site-packages/numpy/lib/npyio.py", line 514, in save
    #         fid.close()
    #     OSError: [Errno 89] Operation canceled
            except (OSError, IOError) as e:
                if e.errno != 35 and e.errno != 89:
                    raise
                sleep(0.01)


def savez(fpath, save_kwargs, timeout=1):
    """
    wraps numpy.savez, implementing OSError tolerance within timeout
    "Save several arrays into a single file in uncompressed ``.npz`` format." DUMP EVERYTHING!
    """
    trying = True
    t0 = time()
    while trying and time() - t0 < timeout:
        try:
            np.savez(fpath, **save_kwargs)
        except IOError as e:
            if e.errno != 35:
                raise
            sleep(0.01)


save_scores = savez    # a synonym for backwards compatibility


def load_codes(codedir, size):
    """ load all the *.npy files in the `codedir`. and randomly sample # `size` of them.
    make sure enough codes for requested size
    """
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
    # load codes
    codes = []
    for codefn in np.random.choice(codefns, size=min(len(codefns), size), replace=False):
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes


def load_codes2(codedir, size):
    """ unlike load_codes, also returns name of load """
    # make sure enough codes for requested size
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
    # load codes
    codefns = list(np.random.choice(codefns, size=min(len(codefns), size), replace=False))
    codes = []
    for codefn in codefns:
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes, codefns


def load_codes_search(codedir, srckey, size=None):
    """Load the code files with `srckey` in its name.

    :param codedir:
    :param srckey: keyword to identify / filter the code. e.g. "gen298_010760.npy", "gen298_010760", "gen298"
    :param size: Defaultly None. if there is too many codes, one can use this to specify the sample size
    :return: codes and corresponding file names `codes, codefns`
    Added @sep.19
    """
    # make sure enough codes for requested size
    codefns = sorted([fn for fn in os.listdir(codedir) if ('.npy' in fn) and (srckey in fn)])

    if not size is None: # input size parameter indicates to select the codes.
        assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
        codefns = list(np.random.choice(codefns, size=min(len(codefns), size), replace=False))

    # load codes by the codefns
    codes = []
    for codefn in codefns:
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes, codefns


def load_block_mat(matfpath):
    attempts = 0
    while True:
        try:
            with h5py.File(matfpath, 'r') as f:
                imgids_refs = np.array(f['stimulusID'])[0]
                imgids = []
                for ref in imgids_refs:
                    imgpath = ''.join(chr(i) for i in f[ref])
                    imgids.append(imgpath.split('\\')[-1])
                imgids = np.array(imgids)
                scores = np.array(f['tEvokedResp'])    # shape = (imgs, channels)
            return imgids, scores
        except (KeyError, IOError, OSError):    # if broken mat file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print('%d failed attempts to read .mat file' % attempts)
            sleep(0.001)


def load_block_mat_code(matfpath):
    attempts = 0
    while True:
        try:
            data = loadmat(matfpath)  # need the mat file to be saved in a older version
            codes = data['codes']
            ids = data['ids']
            imgids = []
            for id in ids[0]:
                imgids.append(id[0])
            return imgids, codes
        except (KeyError, IOError, OSError):  # if broken mat file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print('%d failed attempts to read .mat file' % attempts)
            sleep(0.001)


def set_dynamic_parameters_by_file(fpath, dynamic_parameters):
    try:
        with open(fpath, 'r') as file:
            line = 'placeholder'
            while len(line) > 0:
                line = file.readline()
                if ':' not in line:
                    continue
                if '#' in line:
                    line = line[:line.find('#')]
                if len(line.split(':')) != 2:
                    continue
                key, val = line.split(':')
                key = key.strip()
                val = val.strip()
                try:
                    # if key is not in dynamic_parameter.keys(), will throw KeyError
                    # if val (a str literal) cannot be converted to dynamic_parameter.type, will throw ValueError
                    dynamic_parameters[key].set_value(val)
                except (KeyError, ValueError):
                    continue
    except IOError:
        print('cannot open dynamic parameters file %s' % fpath)


def write_dynamic_parameters_to_file(fpath, dynamic_parameters):
    with open(fpath, 'w') as file:
        for key in sorted(list(dynamic_parameters.keys())):
            file.write('%s:\t%s\t# %s\n' % (key, str(dynamic_parameters[key].value), dynamic_parameters[key].description))


# https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Return the given list sorted in the way that humans expect.
    """
    newl = l[:]
    newl.sort(key=alphanum_key)
    return newl

#%% Dir name manipulation (for experimental code)


def add_neuron_subdir(neuron, exp_dir):
    ''' Add neuron name to the exp_dir, in the form of ('caffe-net', 'fc6', 30). (make the dir in case it doesn't exist) '''
    if len(neuron) == 5:
        subdir = '%s_%s_%04d_%d,%d' % (neuron[0], neuron[1].replace('/', '_'), neuron[2], neuron[3], neuron[4])
    else:
        subdir = '%s_%s_%04d' % (neuron[0], neuron[1].replace('/', '_'), neuron[2])
    this_exp_dir = os.path.join(exp_dir, subdir)
    for dir_ in (this_exp_dir,):
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
    return this_exp_dir


def add_trial_subdir(neuron_dir, trial_title):
    ''' Add trial title to the directory with neuron name on it (make the dir in case it doesn't exist) '''
    trialdir = os.path.join(neuron_dir, trial_title)
    if not os.path.isdir(trialdir):
        os.mkdir(trialdir)
    return trialdir
#%% Code Geometrical Manipulation


def simplex_interpolate(wvec, code_array):
    '''Do simplex interpolate/extrapolate between several codes
    Codes can be input in array (each row is a code) or in list
    wvec: weight vector can be a scalar for 2 codes. or same length list / array for more codes.
    '''
    if type(code_array) is list:
        code_array = np.asarray(code_array)
    code_n = code_array.shape[0]
    if np.isscalar(wvec):
        w_vec = np.asarray([1-wvec, wvec])  # changed @oct.30th, 0 for the 1st vector, 1 for the 2nd vector
    elif len(wvec) == code_n:
        w_vec = np.asarray(wvec)
    elif len(wvec) == code_n - 1:
        w_vec = np.zeros(code_n)
        w_vec[1:] = wvec
        w_vec[0] = 1 - sum(w_vec[:-1])
    else:
        raise ValueError
    code = w_vec @ code_array
    return code
#%%
def codes_summary(codedir, savefile=False):
    """ unlike load_codes, also returns name of load """
    # make sure enough codes for requested size
    if "codes_all.npz" in os.listdir(codedir):
        # if the summary table exist, just read from it!
        with np.load(os.path.join(codedir, "codes_all.npz")) as data:
            codes_all = data["codes_all"]
            generations = data["generations"]
        return codes_all, generations
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    codes = []
    generations = []
    for codefn in codefns:
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
        geni = re.findall(r"gen(\d+)_\d+", codefn)
        generations.append(int(geni[0]))
    codes = np.array(codes)
    generations = np.array(generations)
    if savefile:
        np.savez(os.path.join(codedir, "codes_all.npz"), codes_all=codes, generations=generations)
    return codes, generations

def load_codes_mat(backup_dir, savefile=False, thread_num=1):
    """ load all the code mat file in the experiment folder and summarize it into nparrays"""
    # make sure enough codes for requested size
    if "codes_all.npz" in os.listdir(backup_dir):
        # if the summary table exist, just read from it!
        with np.load(os.path.join(backup_dir, "codes_all.npz")) as data:
            codes_all = data["codes_all"]
            generations = data["generations"]
        return codes_all, generations
    codes_fns = sorted([fn for fn in os.listdir(backup_dir) if "_code.mat" in fn])
    codes_all = []
    img_ids = []
    for i, fn in enumerate(codes_fns[:]):
        matdata = loadmat(os.path.join(backup_dir, fn))
        codes_all.append(matdata["codes"])
        img_ids.extend(list(matdata["ids"]))

    codes_all = np.concatenate(tuple(codes_all), axis=0)
    img_ids = np.concatenate(tuple(img_ids), axis=0)
    img_ids = [img_ids[i][0] for i in range(len(img_ids))]
    generations = [int(re.findall("gen(\d+)", img_id)[0]) if 'gen' in img_id else -1 for img_id in img_ids]
    if savefile:
        np.savez(os.path.join(backup_dir, "codes_all.npz"), codes_all=codes_all, generations=generations)
    return codes_all, generations

def load_multithread_codes_mat(backup_dir, savefile=False, thread_num=1):
    """ load all the code mat file in the experiment folder and summarize it into nparrays"""
    # make sure enough codes for requested size
    if "codes_all.npz" in os.listdir(backup_dir):
        # if the summary table exist, just read from it!
        with np.load(os.path.join(backup_dir, "codes_all.npz")) as data:
            codes_all = data["codes_all"]
            generations = data["generations"]
        return codes_all, generations
    codes_col = []
    generations_col = []
    for thread in range(thread_num):
        codes_fns = sorted([fn for fn in os.listdir(backup_dir) if "thread%03d_code.mat" % thread in fn])
        codes_all = []
        img_ids = []
        for i, fn in enumerate(codes_fns[:]):
            matdata = loadmat(os.path.join(backup_dir, fn))
            codes_all.append(matdata["codes"])
            img_ids.extend(list(matdata["ids"]))

        codes_all = np.concatenate(tuple(codes_all), axis=0)
        img_ids = np.concatenate(tuple(img_ids), axis=0)
        img_ids = [img_ids[i][0] for i in range(len(img_ids))]
        generations = np.array([int(re.findall("gen(\d+)", img_id)[0]) if 'gen' in img_id else -1 for img_id in img_ids])
        codes_col.append(codes_all)
        generations_col.append(generations)
        if savefile:
            np.savez(os.path.join(backup_dir, "codes_all_thread%03d.npz" % thread),
                     codes_all=codes_all, generations=generations)
    return codes_col, generations_col

def scores_imgname_summary(trialdir, savefile=True):
    """ """
    if "scores_all.npz" in os.listdir(trialdir):
        # if the summary table exist, just read from it!
        with np.load(os.path.join(trialdir, "scores_all.npz")) as data:
            scores = data["scores"]
            generations = data["generations"]
            image_ids = data["image_ids"]
        return scores, image_ids, generations

    scorefns = sorted([fn for fn in os.listdir(trialdir) if '.npz' in fn and 'scores_end_block' in fn])
    scores = []
    generations = []
    image_ids = []
    for scorefn in scorefns:
        geni = re.findall(r"scores_end_block(\d+).npz", scorefn)
        scoref = np.load(os.path.join(trialdir, scorefn), allow_pickle=False)
        cur_score = scoref['scores']
        scores.append(cur_score)
        image_ids.extend(list(scoref['image_ids']))
        generations.extend([int(geni[0])] * len(cur_score))
    scores = np.array(scores)
    generations = np.array(generations)
    if savefile:
        np.savez(os.path.join(trialdir, "scores_all.npz"), scores=scores, generations=generations, image_ids=image_ids)
    return scores, image_ids, generations

def scores_summary(CurDataDir, steps = 300, population_size = 40, regenerate=False):
    """Obsolete for the one above! better and more automatic"""
    ScoreEvolveTable = np.full((steps, population_size,), np.NAN)
    ImagefnTable = [[""] * population_size for i in range(steps)]
    fncatalog = os.listdir(CurDataDir)
    if "scores_summary_table.npz" in fncatalog and (not regenerate):
        # if the summary table exist, just read from it!
        with np.load(os.path.join(CurDataDir, "scores_summary_table.npz")) as data:
            ScoreEvolveTable = data['ScoreEvolveTable']
            ImagefnTable = data['ImagefnTable']
        return ScoreEvolveTable, ImagefnTable
    startnum = 0
    for stepi in range(startnum, steps):
        try:
            with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
                score_tmp = data['scores']
                image_ids = data['image_ids']
                ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
                if stepi==0:
                    image_fns = image_ids
                else:
                    image_fns = []
                    for imgid in image_ids:
                        fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and ('.npy' in fn)]
                        assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
                        image_fns.append(fn_tmp_list[0])
                ImagefnTable[stepi][0:len(score_tmp)] = image_fns
                # FIXME: 1st generation natural stimuli is not in the directory! so it's not possible to get the file name there. Here just put the codeid
        except FileNotFoundError:
            if stepi == 0:
                startnum += 1
                steps += 1
                continue
            else:
                print("maximum steps is %d." % stepi)
                ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
                ImagefnTable = ImagefnTable[0:stepi]
                steps = stepi
                break
        ImagefnTable = np.asarray(ImagefnTable)
    savez(os.path.join(CurDataDir, "scores_summary_table.npz"),
                {"ScoreEvolveTable": ScoreEvolveTable, "ImagefnTable": ImagefnTable})
    return ScoreEvolveTable, ImagefnTable


def select_image(CurDataDir, lb=None, ub=None, trial_rng = None):
    '''Filter the Samples that has Score in a given range
    Can be used to find level sets of activation function
    trial_rng slice the trial number direction
    '''
    fncatalog = os.listdir(CurDataDir)
    ScoreEvolveTable, ImageidTable = scores_summary(CurDataDir)
    # it will automatic read the existing summary or generate one.
    if ub is None:
        ub = np.nanmax(ScoreEvolveTable)+1
    if lb is None:
        lb = np.nanmin(ScoreEvolveTable)-1
    if trial_rng is not None:
        assert type(trial_rng) is tuple
        assert len(trial_rng) is 2
        ScoreEvolveTable = ScoreEvolveTable[slice(*trial_rng), :]
        ImageidTable = ImageidTable[slice(*trial_rng), :]
    imgid_list = ImageidTable[np.logical_and(ScoreEvolveTable > lb, ScoreEvolveTable < ub)]
    score_list = ScoreEvolveTable[np.logical_and(ScoreEvolveTable > lb, ScoreEvolveTable < ub)]
    image_fn= []
    for imgid in imgid_list:
        fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and '.npy' in fn]
        assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
        image_fn.append(fn_tmp_list[0])
    code_array = []
    for imagefn in image_fn:
        code = np.load(os.path.join(CurDataDir, imagefn), allow_pickle=False).flatten()
        code_array.append(code.copy())
        # img_tmp = utils.generator.visualize(code_tmp)
    return code_array, score_list, imgid_list

#%% Visualization Routines

import matplotlib.pyplot as plt

def visualize_score_trajectory(CurDataDir, steps=300, population_size=40, title_str="",
                               save=False, exp_title_str='', savedir=''):
    ScoreEvolveTable = np.full((steps, population_size,), np.NAN)
    startnum=0
    for stepi in range(startnum, steps):
        try:
            with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
                score_tmp = data['scores']
                ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
        except FileNotFoundError:
            if stepi == 0:
                startnum += 1
                steps += 1
                continue
            else:
                print("maximum steps is %d." % stepi)
                ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
                steps = stepi
                break

    gen_slice = np.arange(startnum, steps).reshape((-1, 1))
    gen_num = np.repeat(gen_slice, population_size, 1)

    AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
    MaxScore = np.nanmax(ScoreEvolveTable, axis=1)

    figh = plt.figure()
    plt.scatter(gen_num, ScoreEvolveTable, s=16, alpha=0.6, label="all score")
    plt.plot(gen_slice, AvgScore, color='black', label="Average score")
    plt.plot(gen_slice, MaxScore, color='red', label="Max score")
    plt.xlabel("generation #")
    plt.ylabel("CNN unit score")
    plt.title("Optimization Trajectory of Score\n" + title_str)
    plt.legend()
    if save:
        if savedir=='':
            savedir = CurDataDir
        plt.savefig(os.path.join(savedir, exp_title_str + "score_traj"))
    plt.show()
    return figh

def visualize_score_trajectory_cmp (CurDataDir_list, steps=300, population_size=40, title_str_list="",
                               save=False, exp_title_str='', savedir=''):
    assert len(CurDataDir_list) == len(title_str_list)
    figh = plt.figure()
    for CurDataDir, title_str in zip(CurDataDir_list, title_str_list):
        ScoreEvolveTable = np.full((steps, population_size,), np.NAN)
        startnum=0
        for stepi in range(startnum, steps):
            try:
                with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
                    score_tmp = data['scores']
                    ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
            except FileNotFoundError:
                if stepi == 0:
                    startnum += 1
                    steps += 1
                    continue
                else:
                    print("maximum steps is %d." % stepi)
                    ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
                    steps = stepi
                    break

        gen_slice = np.arange(startnum, steps).reshape((-1, 1))
        gen_num = np.repeat(gen_slice, population_size, 1)

        AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
        MaxScore = np.nanmax(ScoreEvolveTable, axis=1)


        plt.scatter(gen_num, ScoreEvolveTable, s=16, alpha=0.3, label="all score "+ title_str)
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")

    plt.xlabel("generation #")
    plt.ylabel("CNN unit score")
    plt.title("Optimization Trajectory of Score Comparison\n" + exp_title_str)
    plt.legend()
    if save:
        if savedir=='':
            savedir = CurDataDir
        plt.savefig(os.path.join(savedir, exp_title_str + "score_traj"))
    plt.show()
    return figh


def visualize_image_score_each_block(CurDataDir, block_num, save=False, exp_title_str='', title_cmap=plt.cm.viridis, col_n=6, savedir=''):
    '''
    # CurDataDir:  "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
    # block_num: the number of block to visualize 20
    # title_cmap: define the colormap to do the code, plt.cm.viridis
    # col_n: number of column in a plot 6
    # FIXED: on Oct. 7th support new name format, and align the score are image correctly
    '''
    fncatalog = os.listdir(CurDataDir)
    fn_score_gen = [fn for fn in fncatalog if
                    (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]
    assert len(fn_score_gen) is 1, "not correct number of score files"
    with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
        score_gen = data['scores']
        image_ids = data['image_ids']
    fn_image_gen = []
    for imgid in image_ids:
        fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and '.bmp' in fn]
        assert len(fn_tmp_list) is 1, "Image file not found or wrong Image file number"
        fn_image_gen.append(fn_tmp_list[0])
    image_num = len(fn_image_gen)

    assert len(score_gen) is image_num, "image and score number do not match"
    lb = score_gen.min()
    ub = score_gen.max()
    if ub == lb:
        cmap_flag = False
    else:
        cmap_flag = True

    row_n = np.ceil(image_num / col_n)
    figW = 12
    figH = figW / col_n * row_n + 1
    # figs, axes = plt.subplots(int(row_n), col_n, figsize=[figW, figH])
    fig = plt.figure(figsize=[figW, figH])
    for i, imagefn in enumerate(fn_image_gen):
        img_tmp = plt.imread(os.path.join(CurDataDir, imagefn))
        score_tmp = score_gen[i]
        plt.subplot(row_n, col_n, i + 1)
        plt.imshow(img_tmp)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        if cmap_flag:  # color the titles with a heatmap!
            plt.title("{0:.2f}".format(score_tmp), fontsize=16,
                      color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
        else:
            plt.title("{0:.2f}".format(score_tmp), fontsize=16)

    plt.suptitle(exp_title_str + "Block{0:03}".format(block_num), fontsize=16)
    plt.tight_layout(h_pad=0.1, w_pad=0, rect=(0, 0, 0.95, 0.9))
    if save:
        plt.savefig(os.path.join(savedir, exp_title_str + "Block{0:03}".format(block_num)))
    plt.show()
    return fig

def visualize_img_list(img_list, scores=None, ncol=11, nrow=11, title_cmap=plt.cm.viridis, show=True):
    """Visualize images from a list and maybe label the score on it!"""
    if scores is not None and not title_cmap == None:
        cmap_flag = True
        ub = scores.max()
        lb = scores.min()
    else:
        cmap_flag = False
    if not len(img_list) <= ncol * nrow:
        ncol = int(np.ceil(len(img_list) / nrow))
    figH = 30
    figW = figH / nrow * ncol + 1
    fig = plt.figure(figsize=[figW, figH])
    for i, img in enumerate(img_list):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(img[:])
        plt.axis('off')
        if cmap_flag:  # color the titles with a heatmap!
            plt.title("{0:.2f}".format(scores[i]), fontsize=16,
                      color=title_cmap((scores[i] - lb) / (ub - lb)))  # normalize a value between [0,1]
        elif scores != None:
            plt.title("{0:.2f}".format(scores[i]), fontsize=16)
        else:
            pass
    if not show:
        plt.show()
    return fig

from Generator import Generator
generator = Generator()
def gen_visualize_image_score_each_block(CurDataDir, block_num, save=False, exp_title_str='', title_cmap=plt.cm.viridis, col_n=6, savedir=''):
    '''
    # CurDataDir:  "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
    # block_num: the number of block to visualize 20
    # title_cmap: define the colormap to do the code, plt.cm.viridis
    # col_n: number of column in a plot 6
    # FIXED: on Oct. 7th support new name format, and align the score are image correctly
    '''
    fncatalog = os.listdir(CurDataDir)
    fn_score_gen = [fn for fn in fncatalog if
                    (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]
    assert len(fn_score_gen) is 1, "not correct number of score files"
    with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
        score_gen = data['scores']
        image_ids = data['image_ids']
    fn_image_gen = []
    for imgid in image_ids:
        fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and '.npy' in fn]
        assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
        fn_image_gen.append(fn_tmp_list[0])
    image_num = len(fn_image_gen)

    assert len(score_gen) is image_num, "image and score number do not match"
    lb = score_gen.min()
    ub = score_gen.max()
    if ub == lb:
        cmap_flag = False
    else:
        cmap_flag = True

    row_n = np.ceil(image_num / col_n)
    figW = 12
    figH = figW / col_n * row_n + 1
    fig = plt.figure(figsize=[figW, figH])
    for i, imagefn in enumerate(fn_image_gen):
        code_tmp = np.load(os.path.join(CurDataDir, imagefn), allow_pickle=False).flatten()
        img_tmp = generator.visualize(code_tmp)
        # img_tmp = plt.imread(os.path.join(CurDataDir, imagefn))
        score_tmp = score_gen[i]
        plt.subplot(row_n, col_n, i + 1)
        plt.imshow(img_tmp)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        if cmap_flag:  # color the titles with a heatmap!
            plt.title("{0:.2f}".format(score_tmp), fontsize=16,
                      color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
        else:
            plt.title("{0:.2f}".format(score_tmp), fontsize=16)

    plt.suptitle(exp_title_str + "\nBlock{0:03}".format(block_num), fontsize=16)
    plt.tight_layout(h_pad=0.1, w_pad=0, rect=(0, 0, 0.95, 0.9))
    if save:
        plt.savefig(os.path.join(savedir, exp_title_str + "Block{0:03}".format(block_num) + ".png"))
    plt.show()
    return fig

# def visualize_image_evolution(CurDataDir, save=True, exp_title_str='', col_n=10, savedir='', cmap_flag=True, title_cmap=plt.cm.viridis):
#     '''
#     # CurDataDir:  "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
#     # block_num: the number of block to visualize 20
#     # title_cmap: define the colormap to do the code, plt.cm.viridis
#     # col_n: number of column in a plot 6
#     # FIXED: on Oct. 7th support new name format, and align the score are image correctly
#     '''
#
#     image_num = len(self.img_ids)
#     gen_num = self.generations.max() + 1
#     row_n = np.ceil(gen_num / col_n)
#     figW = 12
#     figH = figW / col_n * row_n + 1
#     fig = plt.figure(figsize=[figW, figH])
#     for geni in range(gen_num):
#         code_tmp = self.codes_all[self.generations == geni, :].mean(axis=0)
#         img_tmp = generator.visualize(code_tmp)
#         plt.subplot(row_n, col_n, geni + 1)
#         plt.imshow(img_tmp)
#         plt.axis('off')
#         if cmap_flag:  # color the titles with a heatmap!
#             plt.title("{:d} {0:.2f}".format(geni, score_tmp), fontsize=16,
#                       color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
#         else:
#             plt.title("{:d} {0:.2f}".format(geni, score_tmp), fontsize=16)
#
#
#     plt.suptitle(exp_title_str, fontsize=16)
#     plt.tight_layout(h_pad=0.1, w_pad=0, rect=(0, 0, 0.95, 0.9))
#     if save:
#         plt.savefig(os.path.join(savedir, exp_title_str + ".png"))
#     # plt.show()
#     return fig
#
# def visualize_image_evolution(CurDataDir, save=False, exp_title_str='', title_cmap=plt.cm.viridis,
#                               col_n=6, savedir=''):
#     '''
#     # CurDataDir:  "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
#     # block_num: the number of block to visualize 20
#     # title_cmap: define the colormap to do the code, plt.cm.viridis
#     # col_n: number of column in a plot 6
#     # FIXED: on Oct. 7th support new name format, and align the score are image correctly
#     '''
#     fncatalog = os.listdir(CurDataDir)
#     with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
#         score_gen = data['scores']
#         image_ids = data['image_ids']
#
#     image_num = len(fn_image_gen)
#
#     assert len(score_gen) is image_num, "image and score number do not match"
#     lb = score_gen.min()
#     ub = score_gen.max()
#     if ub == lb:
#         cmap_flag = False
#     else:
#         cmap_flag = True
#
#     row_n = np.ceil(image_num / col_n)
#     figW = 12
#     figH = figW / col_n * row_n + 1
#     fig = plt.figure(figsize=[figW, figH])
#     for i, imagefn in enumerate(fn_image_gen):
#         code_tmp = np.load(os.path.join(CurDataDir, imagefn), allow_pickle=False).flatten()
#         img_tmp = generator.visualize(code_tmp)
#         # img_tmp = plt.imread(os.path.join(CurDataDir, imagefn))
#         score_tmp = score_gen[i]
#         plt.subplot(row_n, col_n, i + 1)
#         plt.imshow(img_tmp)
#         plt.axis('off')
#         if cmap_flag:  # color the titles with a heatmap!
#             plt.title("{0:.2f}".format(score_tmp), fontsize=16,
#                       color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
#         else:
#             plt.title("{0:.2f}".format(score_tmp), fontsize=16)
#
#     plt.suptitle(exp_title_str, fontsize=16)
#     plt.tight_layout(h_pad=0.1, w_pad=0, rect=(0, 0, 0.95, 0.9))
#     if save:
#         if savedir == '':
#             savedir = CurDataDir
#         plt.savefig(os.path.join(savedir, exp_title_str ))
#     # plt.show()
#     return fig


def visualize_all(CurDataDir, save=True, title_str=''):
    SaveImgDir = os.path.join(CurDataDir, "sum_img/")
    if not os.path.isdir(SaveImgDir):
        os.mkdir(SaveImgDir)
    for num in range(1, 301):
        try:
            fig = visualize_image_score_each_block(CurDataDir, block_num=num,
                                                         save=save, savedir=SaveImgDir, exp_title_str=title_str)
            fig.clf()
        except AssertionError:
            print("Show and Save %d number of image visualizations. " % (num) )
            break
    visualize_score_trajectory(CurDataDir, title_str="Normal_CNN: No noise",
                                     save=save, savedir=SaveImgDir, exp_title_str=title_str)


def cmp_image_score_across_trial(neuron_dir, trial_list, vis_image_num=10, block_num = 299,
                                 exp_title_str="Method evolving result compare", save=False, savedir=''):
    ''' Generated image comparison across different methods.
    neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
    trial_list = ['choleskycma_sgm3_trial2', 'choleskycma_sgm1_trial2', 'choleskycma_sgm3_uf10_trial0', 'choleskycma_sgm3_uf5_trial0', 'cma_trial0_noeig_sgm5', 'genetic_trial0']
    vis_image_num : how many images to show in a row
    block_num : the image block to show.
    '''
    figW = vis_image_num * 2.5
    figH = len(trial_list) * 2.5 + 1
    col_n = vis_image_num
    row_n = len(trial_list)
    fig = plt.figure(figsize=[figW, figH])
    for trial_j, trial_title in enumerate(trial_list):
        CurDataDir = os.path.join(neuron_dir, trial_title)
        fncatalog = os.listdir(CurDataDir)
        fn_score_gen = [fn for fn in fncatalog if
                        (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]
        assert len(fn_score_gen) is 1, "not correct number of score files"
        with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
            score_gen = data['scores']
            image_ids = data['image_ids']
        idx = np.argsort(
            - score_gen)  # Note the minus sign for best scores sorting. use positive sign for worst score sorting
        score_gen = score_gen[idx]
        image_ids = image_ids[idx]
        fn_image_gen = []
        use_img = not (len([fn for fn in fncatalog if (image_ids[0] in fn) and ('.bmp' in fn)]) == 0)
        # True, if there is bmp rendered files. False, if there is only code, we have to render it through Generator
        for imgid in image_ids[0: vis_image_num]:
            if use_img:
                fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and ('.bmp' in fn)]
                assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
                fn_image_gen.append(fn_tmp_list[0])
            if not use_img:
                fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and ('.npy' in fn)]
                assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
                fn_image_gen.append(fn_tmp_list[0])
        image_num = len(fn_image_gen)
        for i, imagefn in enumerate(fn_image_gen):
            if use_img:
                img_tmp = plt.imread(os.path.join(CurDataDir, imagefn))
            else:
                code_tmp = np.load(os.path.join(CurDataDir, imagefn), allow_pickle=False).flatten()
                img_tmp = generator.visualize(code_tmp)
            score_tmp = score_gen[i]
            plt.subplot(row_n, col_n, trial_j * col_n + i + 1)
            plt.imshow(img_tmp)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel(trial_title)
            else:
                plt.axis('off')
            plt.title("{0:.2f}".format(score_tmp), fontsize=16)
            # if cmap_flag:  # color the titles with a heatmap!
            #     plt.title("{0:.2f}".format(score_tmp), fontsize=16,
            #               color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
            # else:
            #     plt.title("{0:.2f}".format(score_tmp), fontsize=16)

    plt.suptitle(exp_title_str + "\nBlock{0:03}".format(block_num), fontsize=16)
    if save:
        plt.savefig(os.path.join(savedir, exp_title_str + "Block{0:03}".format(block_num)))
    plt.show()
    return fig
#%%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
class ExpDataManage:
    '''The structure to load the data from one experiment and do analysis to the codes
    and generate figures out of it!'''
    def __init__(self, work_dir, trial_title):
        self.work_dir = work_dir
        self.trial_title = trial_title
        self.load_codes()  # codes_all, generations
        self.load_scores()  # img_ids, scores, score_generations
        # Normally the first 2 generations are filler
        # if len(self.score_generations) < len(self.generations):
        #     codes_all = codes_all[]
        # self.generations
        # self.codes_all
        # self.img_ids
        # self.scores


    def load_codes(self, codedir=None, savefile=True):
        """ unlike load_codes, also returns name of load
        Like score summary
        """
        if codedir == None:
            codedir = self.work_dir
        if "codes_all.npz" in os.listdir(codedir):
            # if the summary table exist, just read from it!
            with np.load(os.path.join(codedir, "codes_all.npz")) as data:
                codes_all = data["codes_all"]
                generations = data["generations"]
            self.generations, self.codes_all = generations, codes_all
            return codes_all, generations

        codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
        codes = []
        generations = []
        for codefn in codefns:
            code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
            codes.append(code)
            geni = re.findall(r"gen(\d+)_\d+", codefn)
            generations.append(int(geni[0]))
        codes = np.array(codes)
        generations = np.array(generations)
        if savefile:
            np.savez(os.path.join(codedir, "codes_all.npz"), codes_all=codes, generations=generations)
        self.generations, self.codes_all = generations, codes
        return codes, generations

    def load_scores(self, trialdir=None, savefile=True):
        """ adapt from scores imagename summary"""
        if trialdir == None:
            trialdir = self.work_dir
        if "scores_all.npz" in os.listdir(trialdir):
            # if the summary table exist, just read from it!
            with np.load(os.path.join(trialdir, "scores_all.npz")) as data:
                scores = data["scores"]
                generations = data["generations"]
                image_ids = data["image_ids"]
            self.scores, self.image_ids = scores, image_ids
            self.score_generations = generations
            return scores, image_ids, generations

        scorefns = sorted([fn for fn in os.listdir(trialdir) if '.npz' in fn and 'scores_end_block' in fn])
        scores = []
        generations = []
        image_ids = []
        for scorefn in scorefns:
            geni = re.findall(r"scores_end_block(\d+).npz", scorefn)
            scoref = np.load(os.path.join(trialdir, scorefn), allow_pickle=False)
            cur_score = scoref['scores']
            scores.extend(list(cur_score))
            image_ids.extend(list(scoref['image_ids']))
            generations.extend([int(geni[0])] * len(cur_score))
        scores = np.array(scores)  # 1d array
        generations = np.array(generations)  # 1d array
        if savefile:
            np.savez(os.path.join(trialdir, "scores_all.npz"), scores=scores, generations=generations,
                     image_ids=image_ids)
        self.scores, self.image_ids = scores, image_ids
        self.score_generations = generations
        return scores, image_ids, generations

    def clear_codes(self, codedir=None):
        """ unlike load_codes, also returns name of load """
        if codedir == None:
            codedir = self.work_dir
        # make sure enough codes for requested size
        codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn and 'gen' in fn])
        if not os.path.isfile(os.path.join(codedir, "codes_all.npz")):
            self.load_codes(codedir, savefile=True)
        for fn in codefns:
            os.remove(os.path.join(codedir, fn))
        return

    def clear_scores(self, trialdir=None):
        """ unlike load_codes, also returns name of load """
        if trialdir == None:
            trialdir = self.work_dir
        # make sure enough codes for requested size
        scorefns = sorted([fn for fn in os.listdir(trialdir) if '.npz' in fn and 'scores_end_block' in fn])
        if not os.path.isfile(os.path.join(trialdir, "scores_all.npz")):
            self.load_scores(trialdir, savefile=True)
        for fn in scorefns:
            os.remove(os.path.join(trialdir, fn))
        return

    def visualize_score_trajectory(self, save=False, title_str="", exp_title_str='', savedir=''):
        if title_str == "":
            title_str = self.trial_title
        AvgScore = []
        MaxScore = []
        for geni in range(self.generations.min(), self.generations.max() + 1):
            AvgScore.append(np.nanmean(self.scores[self.generations == geni]))
            MaxScore.append(np.nanmax(self.scores[self.generations == geni]))
        AvgScore = np.array(AvgScore)
        MaxScore = np.array(MaxScore)

        figh = plt.figure()
        plt.scatter(self.generations, self.scores, s=16, alpha=0.6, label="all score")
        plt.plot(self.generations, AvgScore, color='black', label="Average score")
        plt.plot(self.generations, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n" + title_str)
        plt.legend()
        if save:
            if savedir == '':
                savedir = self.work_dir
            plt.savefig(os.path.join(savedir, exp_title_str + "score_traj.png"))
        plt.show()
        return figh

    def code_norm_evolve(self, generations, codes_all, savefig=True, show=False, savedir="", trial_title=""):
        code_norm = np.sum(self.codes_all ** 2, axis=1)  # np.sqrt()
        model = LinearRegression().fit(self.generations.reshape(-1, 1), code_norm)
        if savefig or show:
            plt.figure(figsize=[10, 6])
            plt.scatter(self.generations, code_norm, s=10, alpha=0.7)
            plt.title("Code Norm Square ~ Gen \n %s\n  Linear Coefficient %.f Intercept %.f" % (
            self.trial_title, model.coef_[0], model.intercept_))
            plt.ylabel("Code Norm Squared")
            plt.xlabel("Generations")
            if savefig:
                plt.savefig(os.path.join(savedir, "code_norm_evolution.png"))
            if show:
                plt.show()
        return model.coef_[0], model.intercept_

    def visualize_norm_trajectory(self):
        figh = plt.figure()
        code_norm = np.sum(self.codes_all ** 2, axis=1)  # np.sqrt()
        model = LinearRegression().fit(self.generations.reshape(-1, 1), code_norm)
        plt.figure(figsize=[10, 6])
        plt.scatter(self.generations, code_norm, s=10, alpha=0.7)
        plt.title("Code Norm Square ~ Gen \n %s\n  Linear Coefficient %.f Intercept %.f" % (
        self.trial_title, model.coef_[0], model.intercept_))
        plt.ylabel("Code Norm^2")
        plt.xlabel("Generations")
        plt.savefig("code_norm_evolution.png")
        plt.show()
        return figh

    def visualize_image_evolution(self):
        figh = plt.figure()
        return figh

    def visualize_image_gen(self):
        figh = plt.figure()
        return figh



