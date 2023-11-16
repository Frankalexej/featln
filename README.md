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
6. :python preproc_addglobal.py
7. :python preproc_integrate_mfccs.py
8. :python preproc_separate_trainvaltest_by_speaker.py

Now everything is ready to use under bsc/use/ 