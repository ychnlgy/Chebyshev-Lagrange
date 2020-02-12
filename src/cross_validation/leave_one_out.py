import torch, collections, random, numpy

def leave_one_out(*tensors):
    assert tensors
    lengths = set(map(len, tensors))
    assert len(lengths) == 1
    N = lengths.pop()
    for i in range(N):
        test_tensors = [t[i:i+1] for t in tensors]
        train_tensors = [_get_train(v, i) for v in tensors]
        yield (train_tensors, test_tensors)

def iter_leaveone(uids, arrs):
    splits = partition_by_uid(uids, arrs)
    for i in range(len(splits)):
        test = splits[i]
        data = numpy.concatenate(splits[:i] + splits[i+1:], axis=0)
        yield data, test

def partition_by_uid(uids, arrs):
    d = collections.defaultdict(list)
    for uid, row in zip(uids, arrs):
        d[int(uid)].append(row)
    splits = list(map(numpy.stack, d.values()))
    random.shuffle(splits)
    return splits

# === PRIVATE ===

def _get_train(v, i):
    return torch.cat([v[:i], v[i+1:]])
