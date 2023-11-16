# This file contains path configurations for a speech recognition project.
import os

root_path = "../"
bsc_path = root_path + "src/bsc/"

wav_path = bsc_path + "wav/"
phones_path = bsc_path + "phones/"
words_path = bsc_path + "words/"
test_path = root_path + "src/test/"

phones_extract_path = bsc_path + "phones_extract/"
words_extract_path = bsc_path + "words_extract/"
stat_params_path = bsc_path + "stat_params/"
segments_path = bsc_path + "segments/"


phone_seg_anno_path = bsc_path + "phone_seg_anno/"
phone_seg_random_path = bsc_path + "phone_seg_random/"

phone_seg_random_log_path = bsc_path + "phone_seg_random_log/"
phone_seg_anno_log_path = bsc_path + "phone_seg_anno_log/"

phone_seg_random_rec_path = "/home/ldlmdl/Documents/wavln/src/bsc/phone_seg_random/"
phone_seg_anno_rec_path = "/home/ldlmdl/Documents/wavln/src/bsc/phone_seg_anno/"

bsc_use_path = bsc_path + "use/"

model_save_dir = root_path + "model_save/"
model_eng_save_dir = model_save_dir + "eng/"
model_man_save_dir = model_save_dir + "man/"
# NOTE: don't put file paths here, only directory. 

def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)
            # print(globals()[name])
