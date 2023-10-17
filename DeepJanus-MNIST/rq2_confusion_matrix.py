import glob

import numpy as np
import pandas as pd


def common_member(a, b):
    result = [i for i in a if i in b]
    return result


# Python code to get difference of two lists
def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


for idx in reversed(range(1, 5)):
    print("Attention Threshold = %d" % idx)
    FILES_RANDOM = glob.glob('runs/run_random_at' + str(idx) + '/archive/*.json')
    FILES_ATT = glob.glob('runs/run_heatmap_at' + str(idx) + '/archive/*.json')

    all_seeds_random = []
    for file in FILES_RANDOM:
        df_random = pd.read_json(file, typ='series')
        all_seeds_random.append(df_random['seed'])

    all_seeds_att = []
    for file in FILES_ATT:
        df_att = pd.read_json(file, typ='series')
        all_seeds_att.append(df_att['seed'])

    print("all_seeds_random: ", all_seeds_random)
    print("all_seeds_att: ", all_seeds_att)

    seeds_in_common = common_member(all_seeds_random, all_seeds_att)

    print("ATT yes RND yes: %d => %s" % (len(set(seeds_in_common)), list(set(seeds_in_common))))

    print("ATT no RND yes: %d => %s" % (
        len(set(Diff(seeds_in_common, all_seeds_random))), list(set(Diff(seeds_in_common, all_seeds_random)))))

    print("ATT yes RND no: %d => %s" %
          (len(set(Diff(seeds_in_common, all_seeds_att))), list(set(Diff(seeds_in_common, all_seeds_att)))))

    print("ATT no RND no: %d\n" % (100 - len(set(seeds_in_common + all_seeds_random))))
