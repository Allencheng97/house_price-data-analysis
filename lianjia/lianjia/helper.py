import json
import pandas as pd
import numpy as np
import math
def jsontranscsv(sourcepath,newpath):
    f=open(sourcepath)
    records = [json.loads(line) for line in f.readlines()]
    df = pd.DataFrame(records)
    df.to_csv(newpath,encoding='gb18030')
    f.close()

def cap(x, quantile=[0.01, 0.99]):  # cap method helper function
    delete_list = list()
    head, tail = x.quantile(quantile).values.tolist()
    return head, tail


def choose_best_split(sample_set, var, min_sample):

    score_median_list = math.median(sample_set, var)
    median_len = len(score_median_list)
    sample_cnt = sample_set.shape[0]
    sample1_cnt = sum(sample_set['target'])
    sample0_cnt = sample_cnt - sample1_cnt
    Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

    bestGini = 0.0;
    bestSplit_point = 0.0;
    bestSplit_position = 0.0
    for i in range(median_len):
        left = sample_set[sample_set[var] < score_median_list[i]]
        right = sample_set[sample_set[var] > score_median_list[i]]

        left_cnt = left.shape[0];
        right_cnt = right.shape[0]
        left1_cnt = sum(left['target']);
        right1_cnt = sum(right['target'])
        left0_cnt = left_cnt - left1_cnt;
        right0_cnt = right_cnt - right1_cnt
        left_ratio = left_cnt / sample_cnt;
        right_ratio = right_cnt / sample_cnt

        if left_cnt < min_sample or right_cnt < min_sample:
            continue

        Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
        Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
        Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
        if Gini_temp > bestGini:
            bestGini = Gini_temp;
            bestSplit_point = score_median_list[i]
            if median_len > 1:
                bestSplit_position = i / (median_len - 1)
            else:
                bestSplit_position = i / median_len
        else:
            continue

    Gini = Gini - bestGini
    return bestSplit_point, bestSplit_position