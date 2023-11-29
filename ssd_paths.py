import os

ssd_base_path = "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/featln/"
ssd_src_path = ssd_base_path + "src/"
ssd_aishell_path = ssd_src_path + "aishell/"
as_original_path = ssd_aishell_path + "original/"
as_tar_path = ssd_aishell_path + "tar/"
as_wav_path = ssd_aishell_path + "wav/"
as_phones_extract_path = ssd_aishell_path + "phones_extract/"

as_phone_seg_anno_path = ssd_aishell_path + "phone_seg_anno/"
as_phone_seg_anno_new_path = ssd_aishell_path + "phone_seg_anno_new/"
as_phone_seg_random_path = ssd_aishell_path + "phone_seg_random/"

as_phone_seg_anno_log_path = ssd_aishell_path + "phone_seg_anno_log/"
as_phone_seg_anno_new_log_path = ssd_aishell_path + "phone_seg_anno_new_log/"
as_phone_seg_random_log_path = ssd_aishell_path + "phone_seg_random_log/"

as_phone_seg_anno_MF_path = ssd_aishell_path + "phone_seg_anno_MF/"
as_phone_seg_anno_new_MF_path = ssd_aishell_path + "phone_seg_anno_new_MF/"
as_phone_seg_random_MF_path = ssd_aishell_path + "phone_seg_random_MF/"

as_use_path = ssd_aishell_path + "use/"

sbsc_path = ssd_src_path + "bsc/"
sbsc_use_path = sbsc_path + "use/"

def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)
            # print(globals()[name])