# featln
Feature Learning

## Introduction
This is a reproduced version of our Feature Learning project. 


## Environment Setup
1. 
2. :python paths.py

## Preprocessing (BSC)
1. :python preproc_alignment_extract_bsc.py phone (not yet usable)
2. :python preproc_samplecut.py phone-random; python preproc_samplecut.py phone-anno (not tested here)
3. :python preproc_checkcut.py (if no n_frames)
4. :python preproc_deletebad.py
5. :python preproc_wav2mfcc_singleprocessing.py
6. :python preproc_addglobalindex.py
7. :python preproc_integrate_mfccs.py
8. :python preproc_separate_trainvaltest_by_speaker.py

Now everything is ready to use under bsc/use/ 


## Preprocessing (AISHELL)
1. :preproc_man_extracttext.ipynb [separate 汉字 and pinyin]
2. :preproc_man_resampleaudio.ipynb [resample from 44100Hz to 16000Hz]
3. :preproc_man_charsiu.ipynb [get alignment and annotation, sufficient to extract valid and test sets only]
4. :preproc_man_length_stat_vis.ipynb [inspect dataset phone distribution and get stats for random sampling]
5. :python preproc_man_samplecut.py phone-random/phone-anno and phone-random-bind/phone-anno-bind
6. :preproc_man_separate_trainvaltest_by_speaker.ipynb [separate training, validation and testing datasets by speaker]
7. :preproc_man_token_detonolize.ipynb [delete tone numbers in pinyin]
8. :python preproc_man_addglobalindex.py
9. :python preproc_man_wav2mfcc_multiprocessing_MF.py
10. :python preproc_man_integrate_mfccs.py / :python preproc_man_integrate_mfccs_stepbystep.py [this is much faster for large size integration]