import argparse
import gc
import multiprocessing
import os

import numpy as np
from iteration_utilities import deepflatten
from sklearn.utils import shuffle


def int2bin(num, padding=False, pad_to=8):
    lst = []
    while num > 1:
        lst.insert(0, num % 2)
        num = num // 2
    lst.insert(0, num)
    if padding:
        while len(lst) < pad_to:
            lst.insert(0, 0)
    return lst


def prep(x_list, window_size, window_step):
    windows = []
    for line in x_list:
        n = len(line)
        for j in range(0, n - window_size + 1, window_step):
            windows.append(line[j:j + window_size])
    return np.array(windows)


def window_to_binary(data, window_size):
    bin_data = []
    for i, x in enumerate(data):
        tmp = list(deepflatten(x))
        tmp = [int2bin(ele, padding=True) for ele in tmp]
        bin_data.append(tmp)
        if i % 1000 == 0:
            print("Converted %d/%d" % (i, len(data)))
    bin_data = np.array(bin_data, dtype=np.uint8)
    return bin_data.reshape((bin_data.shape[0], window_size, 32, 8))


def deduplicate(X_mal, X_ben):
    mal_set = {tuple(window.reshape(-1).tolist()) for window in X_mal}
    ben_set = {tuple(window.reshape(-1).tolist()) for window in X_ben}
    duplicate_set = mal_set & ben_set
    mal_set -= duplicate_set
    ben_set -= duplicate_set

    X_mal = np.array(sorted(mal_set), dtype=np.uint8).reshape((-1, X_mal.shape[1], X_mal.shape[2])
                                                             if len(mal_set) else (0, X_mal.shape[1], X_mal.shape[2]))
    X_ben = np.array(sorted(ben_set), dtype=np.uint8).reshape((-1, X_ben.shape[1], X_ben.shape[2])
                                                             if len(ben_set) else (0, X_ben.shape[1], X_ben.shape[2]))
    return X_mal, X_ben


def remove_cross_partition_duplicates(partitions):
    owners = {}
    for name, (X_ben, X_mal) in partitions.items():
        for window in X_ben:
            key = tuple(window.reshape(-1).tolist())
            owners.setdefault(key, set()).add((name, 0))
        for window in X_mal:
            key = tuple(window.reshape(-1).tolist())
            owners.setdefault(key, set()).add((name, 1))

    duplicate_keys = {
        key for key, locations in owners.items()
        if len(locations) > 1
    }
    output = {}
    for name, (X_ben, X_mal) in partitions.items():
        keep_ben = [
            window for window in X_ben
            if tuple(window.reshape(-1).tolist()) not in duplicate_keys
        ]
        keep_mal = [
            window for window in X_mal
            if tuple(window.reshape(-1).tolist()) not in duplicate_keys
        ]
        X_ben = np.array(keep_ben, dtype=np.uint8).reshape(
            (-1, X_ben.shape[1], X_ben.shape[2]))
        X_mal = np.array(keep_mal, dtype=np.uint8).reshape(
            (-1, X_mal.shape[1], X_mal.shape[2]))
        output[name] = (X_ben, X_mal)
    return output


def balance(X_ben, X_mal, seed):
    if len(X_ben) > len(X_mal) * 1.2:
        X_ben = shuffle(X_ben, random_state=seed)[:len(X_mal)]
    elif len(X_ben) * 1.2 < len(X_mal):
        X_mal = shuffle(X_mal, random_state=seed)[:len(X_ben)]
    return X_ben, X_mal


def prepare_partition(benign_bytes, malicious_bytes, window_size, window_step, seed):
    X_ben = prep(benign_bytes, window_size, window_step)
    X_mal = prep(malicious_bytes, window_size, window_step)
    X_ben = np.unique(X_ben.reshape((-1, window_size, 32)), axis=0) if len(X_ben) else np.empty((0, window_size, 32), dtype=np.uint8)
    X_mal = np.unique(X_mal.reshape((-1, window_size, 32)), axis=0) if len(X_mal) else np.empty((0, window_size, 32), dtype=np.uint8)
    X_mal, X_ben = deduplicate(X_mal, X_ben)
    X_ben, X_mal = balance(X_ben, X_mal, seed)
    X_ben, X_mal = shuffle(X_ben, random_state=seed), shuffle(X_mal, random_state=seed)
    return X_ben, X_mal


def save_partition(name, k, fold, X_ben, X_mal, window_size):
    Y_mal = np.append(
        np.zeros(shape=(len(X_mal), 1)),
        np.ones(shape=(len(X_mal), 1)),
        1)
    Y_ben = np.append(
        np.ones(shape=(len(X_ben), 1)),
        np.zeros(shape=(len(X_ben), 1)),
        1)

    X = np.append(X_mal, X_ben, 0)
    Y = np.append(Y_mal, Y_ben, 0)
    X, Y = shuffle(X, Y, random_state=1)
    X = window_to_binary(X, window_size)

    suffix = "%.3d" % k
    if fold is not None:
        suffix += "-%.1d" % fold

    np.save(os.path.join("data", "X_" + name + "-" + suffix + ".npy"),
            X, allow_pickle=True)
    np.save(os.path.join("data", "Y_" + name + "-" + suffix + ".npy"),
            Y, allow_pickle=True)


def split_raw_data(benign_bytes, malicious_bytes, test_fraction, seed):
    rng = np.random.default_rng(seed)
    test_indices = []
    fold_indices = []

    for data in (benign_bytes, malicious_bytes):
        indices = rng.permutation(len(data))
        test_count = max(1, int(len(data) * test_fraction))
        test_indices.append(indices[:test_count])
        fold_indices.append(np.array_split(indices[test_count:], 4))

    benign_test = benign_bytes[test_indices[0]]
    malicious_test = malicious_bytes[test_indices[1]]
    benign_folds = fold_indices[0]
    malicious_folds = fold_indices[1]

    folds = []
    for fold in range(4):
        benign_val = benign_bytes[benign_folds[fold]]
        malicious_val = malicious_bytes[malicious_folds[fold]]
        benign_train_indices = np.concatenate([
            benign_folds[i] for i in range(4) if i != fold
        ])
        malicious_train_indices = np.concatenate([
            malicious_folds[i] for i in range(4) if i != fold
        ])
        folds.append((
            (benign_bytes[benign_train_indices],
             malicious_bytes[malicious_train_indices]),
            (benign_val, malicious_val)))

    return (benign_test, malicious_test), folds


def thread_function(k, window_size, window_step, benign_bytes, malicious_bytes,
                   test_fraction, seed):
    test, folds = split_raw_data(
        benign_bytes, malicious_bytes, test_fraction, seed)

    partitions = {}
    partitions["test"] = prepare_partition(
        test[0], test[1], window_size, window_step, seed)

    for fold, (train, val) in enumerate(folds):
        partitions["train-%d" % fold] = prepare_partition(
            train[0], train[1], window_size, window_step, seed + fold)
        partitions["val-%d" % fold] = prepare_partition(
            val[0], val[1], window_size, window_step, seed + fold)

    partitions = remove_cross_partition_duplicates(partitions)

    X_ben_test, X_mal_test = partitions["test"]
    save_partition("test", k, None, X_ben_test, X_mal_test, window_size)

    for fold in range(4):
        X_ben_train, X_mal_train = partitions["train-%d" % fold]
        X_ben_val, X_mal_val = partitions["val-%d" % fold]
        save_partition("train", k, fold, X_ben_train, X_mal_train, window_size)
        save_partition("val", k, fold, X_ben_val, X_mal_val, window_size)

    gc.collect()
    print("Data sets exported for k=%d" % k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-num",
        type=int,
        default=1,
        help="Number of CPU cores for parallel execution.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of each raw data class reserved as the test set.")
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed used for splitting raw sessions.")
    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")

    gc.enable()
    benign_bytes = np.load(os.path.join("data", "benign_bytes.npy"),
                           allow_pickle=True)
    malicious_bytes = np.load(os.path.join("data", "malicious_bytes.npy"),
                              allow_pickle=True)

    window_sizes = [4, 6, 8, 10, 12]
    window_steps = [1, 2, 4, 6, 8]

    job_args = []
    k = 0
    for window_size in window_sizes:
        for window_step in window_steps:
            job_args.append(
                (k, window_size, window_step, benign_bytes, malicious_bytes,
                 args.test_fraction, args.seed))
            k += 1

    with multiprocessing.Pool(processes=args.cpu_num) as pool:
        pool.starmap(thread_function, job_args)
