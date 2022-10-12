import os.path
from os import listdir, scandir, remove
from os.path import isfile, join
fp = "/home/alp/workspace/code/sign_language_transfer/experiments/"
subfolders = [ f.path for f in scandir(fp) if f.is_dir()]
for i in subfolders:
    try:
        p = join(i,"checkpoints/")
        if os.path.isdir(p):
            onlyfiles = [f for f in listdir(p) if isfile(join(p, f))]
            onlyfiles = [x for x in onlyfiles if "loss" not in x]
            onlyfiles = sorted(onlyfiles, key=lambda x: float(x.split("_")[-1][3:-4]))
            print(onlyfiles)
            deleted = onlyfiles[:-2] if len(onlyfiles) > 2 else []
            for j in deleted:
                print(join(p,j))
                remove(join(p,j))
    except:
        print(onlyfiles)

    try:
        p = join(i, "val_features_challenge/")
        if os.path.isdir(p):
            onlyfiles = [f for f in listdir(p) if isfile(join(p, f))]
            onlyfiles = [x for x in onlyfiles if "loss" not in x]
            onlyfiles = sorted(onlyfiles, key=lambda x: float(x.split("_")[-1][:-4]))
            print(onlyfiles)
            deleted = onlyfiles[:-2] if len(onlyfiles) > 2 else []
            for j in deleted:
                print(join(p,j))
                remove(join(p,j))
    except:
        print(onlyfiles)
