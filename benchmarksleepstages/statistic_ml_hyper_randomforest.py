from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import argparse
import sys
import os
# 添加上级目录到模块搜索路径中
sys.path.insert(0, os.path.abspath(r'/root/ceshi'))
from sleep_stage_config import Config
from dataset_builder_loader.data_loader import *



def main(args):
    cfg = Config()
    print("...Loading dataset into memory...")
    data_loader = DataLoader(cfg, args.modality, args.num_classes, 20)
    df_train, df_test, featnames = data_loader.load_df_dataset()
    print("...Done...")
    scoring = "accuracy"
    params = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
    }
    df_train['stages'] = df_train['stages'].apply(lambda x: cast_sleep_stages(x, args.num_classes))
    experiment = "Random Forest"
    classifier = RandomForestClassifier(n_jobs=5, max_features='sqrt', n_estimators=300, oob_score=True)
    print("..Scoring function is %s..." % scoring)
    print("Available parameters for classifier : %s" % experiment)
    print(classifier.get_params().keys())
    gs = optimize_parameters(df_train, featnames, classifier, params, scoring, experiment, args.num_classes,
                             output_path=cfg.HP_CV_OUTPUT_FOLDER)
    gsres = display_parameters(gs)

    with open(cfg.HP_CV_OUTPUT_FOLDER + "%d_stages_%s_%s_.pkl" % (args.num_classes, experiment, args.modality), "wb") as f:
        pickle.dump(gs, f)

    predictions = gs.predict(df_test[featnames])

    print(" - Acuracy: %.4f" % (metrics.accuracy_score(df_test["stages"], predictions)))
    print(" - F1: %.4f" % (metrics.f1_score(df_test["stages"], predictions, average="macro")))
    print(" - Best parameter: %s" % gs.best_estimator_)
    gsres.to_csv(os.path.join(cfg.HP_CV_OUTPUT_FOLDER, "%d_stages_%s_%s_%s.csv" %
                              (args.num_classes, experiment, args.modality, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    print("...Generated file '%s'..." % cfg.HP_CV_OUTPUT_FOLDER)
    print("...All done...")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all', help='the raw modality to use')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')
    parser.add_argument('--eval_method', type=str, default='macro',
                        help='The weighting method for classifier level metrics')
    parser.add_argument('--hrv_win_len', type=int, default=270,
                        help='window length to decide which h5 file to use, units=secs')
    return parser.parse_args(argv)


if __name__ == "__main__":
    # 如果没有通过命令行传入参数，则直接在代码中指定默认参数
    if len(sys.argv[1:]) == 0:
        # 这里可以修改需要的参数，例如：--modality all --num_classes 3
        arg_list = ['--modality', 'all', '--num_classes', '3']
    else:
        arg_list = sys.argv[1:]
    main(parse_arguments(arg_list))
