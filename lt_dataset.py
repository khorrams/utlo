# added by authors
import os
import json
import argparse
import numpy as np

def dataset_lt(args):
    pwd = os.path.dirname(args.fn)
    with open(args.fn) as f:
        main_ = json.load(f)

    if main_["labels"]:
        main = np.array(main_["labels"], dtype=object)
    else:
        exit(0)

    classes = get_img_num_per_cls(main, args)
    lt_ds = gen_imbalanced_data(main, classes)
    lt = {"labels": lt_ds}

    filename = f"lt_{args.imf}"
    if args.reverse:
        filename += "_reverse"
    with open(os.path.join(pwd, f"{filename}.json"), "w") as f:
        json.dump(lt, f)

def get_img_num_per_cls(ds, args):
    classes = dict()
    for d in ds:
        if d[1] in classes:
            classes[d[1]] += 1
        else:
            classes[d[1]] = 1
    cls_num = len(classes)

    if args.dname == "flowers" or args.dname == "animals":
        classes = {k: v for k, v in reversed(sorted(classes.items(), key=lambda item: item[1]))}
        img_max = classes[max(classes, key=classes.get)]
    elif args.dname.startswith("cifar") or args.dname == "lsun":
        classes = {k: v for k, v in reversed(sorted(classes.items(), key=lambda item: item[0]))}
        img_max = classes[0]
    else:
        raise ValueError

    for id, (k, v) in enumerate(classes.items()):
        if args.reverse:
            num = img_max * (1/args.imf ** ((cls_num - 1 - id) / (cls_num - 1.0)))
            if v < num:
                print(f"num images ({v}) is smaller than long-tailed value ({num})..")
            classes[k] = min(int(num), v)
        else:
            num = img_max * (1/args.imf ** (id / (cls_num - 1.0)))
            classes[k] = min(int(num), v)
    return classes

def gen_imbalanced_data(ds, classes):
    new_data = []
    for id, (k, v) in enumerate(classes.items()):
        idx = np.where(ds[:, 1] == k)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:v]
        new_data.extend([[x[0], id] for x in ds[selec_idx, ...]])
    return new_data

def shot_labels(args):
    pwd = os.path.dirname(args.fn)
    with open(args.fn) as f:
        main_ = json.load(f)

    if main_["labels"]:
        main = np.array(main_["labels"], dtype=object)
    else:
        exit(0)

    classes = get_img_num_per_cls(main, args)

    if args.dname == "cifar10":
        inds = [0, 3, 6, 10]
    elif args.dname == "cifar100":
        inds = [0, 35, 70, 100]
    elif args.dname == "flowers":
        inds = [0, 20, 50, 102]
    elif args.dname == "animals":
        inds = [0, 5, 10, 20]
    elif args.dname == "lsun":
        inds = [0, 3, 4, 5]
    else:
        raise ValueError

    reordered_data = []
    for id, (k, v) in enumerate(classes.items()):
        idx = np.where(main[:, 1] == k)[0]
        reordered_data.extend([[x[0], id] for x in main[idx, ...]])
    reordered_data = np.array(reordered_data, dtype=object)

    for i in range(len(inds)-1):
        cur_shot = []
        for cls_id in range(inds[i], inds[i+1]):
            idx = np.where(reordered_data[:, 1] == cls_id)[0]
            cur_shot.extend(reordered_data[idx, ...].tolist())

        shot = {"labels": cur_shot}
        filename = f"class_{inds[i]}_{inds[i+1] - 1}"
        with open(os.path.join(pwd, f"{filename}.json"), "w") as f:
            json.dump(shot, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongTail Dataset')
    parser.add_argument('--fn', required=True, metavar='PATH', type=str, help='path to the dataset json file')
    parser.add_argument('--dname', required=True, choices=['cifar10', 'cifar100', 'animals', 'flowers', 'lsun'],
                        type=str, help='name of the dataset')
    parser.add_argument('--imf', required=True, default=25, type=int, help='imbalance factor')
    parser.add_argument('--reverse', action="store_true", help='reverse the imabalnce')
    parser.add_argument('--shot-labels', action="store_true", help='number of samples in the lt class')
    args = parser.parse_args()

    dataset_lt(args)
    if args.shot_labels:
        shot_labels(args)
