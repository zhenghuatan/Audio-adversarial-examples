# Audio-adversarial-examples
Datasets of audio adversarial examples for deep speech recognition systems and Python code of a detection system

Description:
This page contains speech adversarial examples generated through attacking deep speech recognition systems, together with the Python source code for detecting these adversarial examples. Both white-box and black-box targeted attacks are included. For details, refer to [1]. 
 
Databases of adversarial examples:
Here are the two speech datasets of adversarial examples: 
dataset for white-box attacks (198 MB) and 
dataset for black-box attacks (88 MB). 
Both normal and adversarial examples are included and each dataset contains equal number of normal and adversarial examples. The dataset for white-box attacks was built upon the Mozilla Common Voice dataset [2] and the dataset for black-box attacks was built upon the Google Speech Command dataset [3]. Both the Mozilla Common Voice dataset and the Google Speech Command dataset are under the Creative Commons license, so are the adversarial datasets. 

Source code: 
Source code for adversarial attack detection based on convolutional neural networks is available here: [source code in Python](/adversarial_trainA_testA.py). 

Citations:

[1] Saeid Samizade, Zheng-Hua Tan, Chao Shen and Xiaohong Guan, “Adversarial Example Detection by Classification for Deep Speech Recognition”, ICASSP 2020.

[2] Mozilla common voice dataset. [Online]. Available: https://voice.mozilla.org/en/datasets, accessed 2019. 

[3] Pete Warden, “Speech commands: A dataset for limited- vocabulary speech recognition,” CoRR, vol. abs/1804.03209, 2018. [Online]. Available: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz, accessed 2019. 
