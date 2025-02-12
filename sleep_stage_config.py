import os

class Config(object):
    def __init__(self):
        # 公共基础路径
        CODE_BASE_PATH = r"/root/ceshi"
        DATA_BASE_PATH = r"/root/autodl-tmp/mesa"
        TEMP_BASE_PATH = r"/root/autodl-tmp/ouputs"

        
        # setup the modality, notice in MESA activity measurement is "activity counts" not ENMO
        self.FEATURE_TYPE_DICT = {'hr': 'HR', 'hrv': 'HRV', 'acc': 'ENMO', 'all': 'ENMO_HRV'}
        # set up the child folder directory with respect to sleep period and sleep recording period
        self.ANALYSIS_SUB_FOLDER = {'sleep_period': "sp_summary", "recording_period": "summary"}
        self.SUMMARY_FOLDER = {'r': 'summary', 's': 'sp_summary'}

        #####################

        # download HRV data from MESA website and store it to here
        self.HR_PATH = os.path.join(DATA_BASE_PATH, "polysomnography/annotations-rpoints")
        # download actigraphy data from MESA website and store it to here
        self.ACC_PATH = os.path.join(DATA_BASE_PATH, "actigraphy")        
        # A file maps actigraphy and RR intervals by sleep epoch index
        self.OVERLAP_PATH = os.path.join(DATA_BASE_PATH, "overlap/mesa-actigraphy-psg-overlap.csv")

        
        #####################
        # A feature list file that is corresponding to
        self.FEATURE_LIST = os.path.join(CODE_BASE_PATH, "assets", "feature_list.csv")  #  "../assets/feature_list.csv"
        self.TRAIN_TEST_SPLIT = os.path.join(CODE_BASE_PATH, "assets", "train_test_pid_split.csv")
        # A readable table for all algorithms
        self.ALG_READABLE_FILE = os.path.join(CODE_BASE_PATH, "assets", "alg_readable.csv")


        #####################
        # A directory to store aligned "Actigraphy" and "RR interval" and extracted features
        self.CSV30_DATA_PATH = os.path.join(TEMP_BASE_PATH, "HRV30s_ACC_CSV", "Aligned_final")
        # a directory for storing H5 files for train, val and test
        self.H5_OUTPUT_PATH = os.path.join(TEMP_BASE_PATH, "HRV30s_ACC30s_H5")
        # a standardrised H5 file contains all input features
        self.HRV30_ACC_STD_PATH = os.path.join(TEMP_BASE_PATH, "HRV30s_ACC30s_H5", "hrv30s_acc30s_full_feat_stand.h5")
        # a standardrise transformer derived from training dataset
        self.STANDARDISER = {30: os.path.join(TEMP_BASE_PATH, "HRV30s_ACC30s_H5", "HRV30s_ACC30s_full_feat_stand.h5_std_transformer")}
        # Deep learning H5 cache file for windowed data.
        self.NN_ACC_HRV = os.path.join(TEMP_BASE_PATH, "HRV30s_ACC30s_H5", "nn_acc_hrv30s_%d.h5")        
        # the directory to store
        self.EXPERIMENT_RESULTS_ROOT_FOLDER = os.path.join(TEMP_BASE_PATH, "Results")
        self.HP_CV_OUTPUT_FOLDER = os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "HP_CV_TUNING")
        # The following setting is for deep learning
        # this is the place to save all experiments' outputs
        self.STAGE_OUTPUT_FOLDER_HRV30s = {
            2: os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "2stages"),
            3: os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "3stages"),
            4: os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "4stages"),
            5: os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "5stages")
        }
        self.CNN_FOLDER = os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "HP_CV_TUNING", "CNN")
        self.LSTM_FOLDER = os.path.join(TEMP_BASE_PATH, "Results", "HRV30s_ACC", "HP_CV_TUNING", "LSTM")

        # 初始化时自动检测并创建相关目录
        self.ensure_directories_exist()

        # Added for multi-class support
        self.STAGE_NAMES = {
            2: ['Wake', 'Sleep'],
            3: ['Wake', 'NREM', 'REM'],
            4: ['Wake', 'Light', 'Deep', 'REM'],
            5: ['Wake', 'N1', 'N2', 'N3', 'REM']
        }

    def ensure_directories_exist(self):
        """
        检查配置中所有需要的路径是否存在，
        如果不存在则创建空的文件夹。
        针对文件路径，会创建其父目录。
        """
        # 针对存在文件路径的构建，获取其所在目录
        paths_to_check = [
            # 目录路径
            self.HR_PATH,
            self.ACC_PATH,
            self.CSV30_DATA_PATH,
            self.H5_OUTPUT_PATH,
            self.EXPERIMENT_RESULTS_ROOT_FOLDER,
            self.HP_CV_OUTPUT_FOLDER,
            self.CNN_FOLDER,
            self.LSTM_FOLDER,
        ]

        # 针对文件路径，确保其父目录存在
        paths_to_check += [
            os.path.dirname(self.OVERLAP_PATH),
            os.path.dirname(self.FEATURE_LIST),
            os.path.dirname(self.TRAIN_TEST_SPLIT),
            os.path.dirname(self.ALG_READABLE_FILE),
            os.path.dirname(self.HRV30_ACC_STD_PATH),
            os.path.dirname(self.NN_ACC_HRV),
        ]

        # STANDARDISER 字典中存储的是文件路径
        for path in self.STANDARDISER.values():
            paths_to_check.append(os.path.dirname(path))

        # STAGE_OUTPUT_FOLDER_HRV30s 中存储的是目录路径
        paths_to_check += list(self.STAGE_OUTPUT_FOLDER_HRV30s.values())

        #此处添加自定义路径
        TEMP_BASE_PATH =r"/root/autodl-tmp/tmp"
        paths_to_check += [
            TEMP_BASE_PATH,
            os.path.join(TEMP_BASE_PATH, "act_features"),
            os.path.join(TEMP_BASE_PATH, "hrv_features"),
        ]


        # 去重处理
        unique_paths = set(paths_to_check)

        for path in unique_paths:
            if not os.path.exists(path):
                print(f"创建目录: {path}")
                os.makedirs(path, exist_ok=True)
            else:
                print(f"目录已存在: {path}")    


if __name__ == "__main__":
    config = Config()
    config.ensure_directories_exist()