# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-26 2020-05-26)

import sys
import logging
import argparse
import traceback
import pandas as pd

# Logger
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parse
def get_args():
    """Score:
            <key1, key2, score>
    Recommend: subset 2000 ~ 3000 utts from trainset as a cohort set and use asnorm with top-n=200 ~ 400.
    """
    parser = argparse.ArgumentParser(
            description="Score Normalization.")

    parser.add_argument("--method", default="asnorm", type=str,
                        choices=["snorm", "asnorm"],
                        help="Choices to select a score normalization.")

    parser.add_argument("--top-n", type=int, default=300,
                        help="Used in AS-Norm.")
    
    parser.add_argument("--second-cohort", type=str, default="true", choices=["true", "false"],
                        help="If true, get cohort key from the second field of score.")

    parser.add_argument("--cross-select", type=str, default="false", choices=["true", "false"],
                        help="Used in AS-Norm. "
                             "If true, select top n enroll/test keys by test/enroll_cohort scores. "
                             "If false, select top n enroll/test keys by enroll/test_cohort scores.")

    parser.add_argument("input_score", metavar="enroll-test-score", type=str,
                        help="Original score path for <enroll, test>.")

    parser.add_argument("enroll_cohort_score", metavar="enroll-cohort-score", type=str,
                        help="Score file path for <enroll, cohort>.")

    parser.add_argument("test_cohort_score", metavar="test-cohort-score", type=str,
                        help="Score file path for <test, cohort>.")

    parser.add_argument("output_score", metavar="output-score-path", type=str,
                        help="Output score path for <enroll, test> after score normalization.")

    args = parser.parse_args()

    return args

def load_score(score_path, names, sep=" "):
    logger.info("Load score form {} ...".format(score_path))
    df = pd.read_csv(score_path, sep=sep, names=names)
    return df

def save_score(score, score_path, sep=" "):
    logger.info("Save score to {} ...".format(score_path))
    df = pd.DataFrame(score)
    df.to_csv(score_path, header=None, sep=sep, index=False)

def snorm(args):
    """ Symmetrical Normalization.
    Reference: Kenny, P. (2010). Bayesian speaker verification with heavy-tailed priors. Paper presented at the Odyssey.
    """
    enroll_test_names = ["enroll", "test", "score"]

    if args.second_cohort == "true":
        enroll_cohort_names = ["enroll", "cohort", "score"]
        test_cohort_names = ["test", "cohort", "score"]
    else:
        enroll_cohort_names = ["cohort", "enroll", "score"]
        test_cohort_names = ["cohort", "test", "score"]

    input_score = load_score(args.input_score, enroll_test_names)
    enroll_cohort_score = load_score(args.enroll_cohort_score, enroll_cohort_names)
    test_cohort_score = load_score(args.test_cohort_score, test_cohort_names)

    output_score = []

    logger.info("Use Symmetrical Normalization (S-Norm) to normalize scores ...")

    # This .groupby function is really an efficient method than 'for' grammar.
    enroll_group = enroll_cohort_score.groupby("enroll")
    test_group = test_cohort_score.groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                       (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    logger.info("Normalize scores done.")
    save_score(output_score, args.output_score)

def asnorm(args):
    """ Adaptive Symmetrical Normalization.
    Reference: Cumani, S., Batzu, P. D., Colibro, D., Vair, C., Laface, P., & Vasilakakis, V. (2011). Comparison of 
               speaker recognition approaches for real applications. Paper presented at the Twelfth Annual Conference 
               of the International Speech Communication Association.

               Cai, Danwei, et al. “The DKU-SMIIP System for NIST 2018 Speaker Recognition Evaluation.” Interspeech 2019, 
               2019, pp. 4370–4374.

    Recommend: Matejka, P., Novotný, O., Plchot, O., Burget, L., Sánchez, M. D., & Cernocký, J. (2017). Analysis of 
               Score Normalization in Multilingual Speaker Recognition. Paper presented at the Interspeech.

    """
    enroll_test_names = ["enroll", "test", "score"]

    if args.second_cohort == "true":
        enroll_cohort_names = ["enroll", "cohort", "score"]
        test_cohort_names = ["test", "cohort", "score"]
    else:
        enroll_cohort_names = ["cohort", "enroll", "score"]
        test_cohort_names = ["cohort", "test", "score"]

    input_score = load_score(args.input_score, enroll_test_names)
    enroll_cohort_score = load_score(args.enroll_cohort_score, enroll_cohort_names)
    test_cohort_score = load_score(args.test_cohort_score, test_cohort_names)

    output_score = []

    logger.info("Use Adaptive Symmetrical Normalization (AS-Norm) to normalize scores ...")

    # Note that, .sort_values function will return NoneType with inplace=True and .head function will return a DataFrame object.
    # The order sort->groupby is equal to groupby->sort, so there is no problem about independence of trials.
    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)

    if args.cross_select == "true":
        logger.info("Select top n scores by cross method.")
        # The SQL grammar is used to implement the cross selection based on pandas.
        # Let A is enroll_test table, B is enroll_cohort table and C is test_cohort table.
        # To get a test_group (select "test:cohort" pairs) where the cohort utterances' scores is selected by enroll_top_n,
        # we should get the D table by concatenating AxC with "enroll" key firstly and then
        # we could get the target E table by concatenating BxD wiht "test"&"cohort" key.
        # Finally, the E table should be grouped by "enroll"&"test" key to make sure the group key is unique.
        enroll_top_n = enroll_cohort_score.groupby("enroll").head(args.top_n)[["enroll", "cohort"]]
        test_group = pd.merge(pd.merge(input_score[["enroll", "test"]], enroll_top_n, on="enroll"), 
                              test_cohort_score, on=["test", "cohort"]).groupby(["enroll", "test"])

        test_top_n = test_cohort_score.groupby("test").head(args.top_n)[["test", "cohort"]]
        enroll_group = pd.merge(pd.merge(input_score[["enroll", "test"]], test_top_n, on="test"), 
                                enroll_cohort_score, on=["enroll", "cohort"]).groupby(["enroll", "test"])
    else:
        enroll_group = enroll_cohort_score.groupby("enroll").head(args.top_n).groupby("enroll")
        test_group = test_cohort_score.groupby("test").head(args.top_n).groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        if args.cross_select == "true":
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] + \
                                 (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                                (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    logger.info("Normalize scores done.")
    save_score(output_score, args.output_score)

def main():
    print(" ".join(sys.argv))
    args = get_args()
    try:
        if args.method == "snorm":
            snorm(args)
        elif args.method == "asnorm":
            asnorm(args)
        else:
            raise TypeError("Do not support {} score normalization.".format(args.method))
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

