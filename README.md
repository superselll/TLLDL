# Two Label Problem

## ACNE04 Dataset
The ACNE04 dataset can be downloaded from [Baidu](https://pan.baidu.com/s/15JQlymnhnEmEt8Q5zpJQDw) (pw: fbrm) or [Google](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ?usp=sharing).

## run
our method: python train_acne_markov.py
base line: python train_acne_original.py

# One Label Problem

## Datasets
Due to file size limitations, we only provide the Yeast_alpha dataset. 
Remaining datasets can be downloaded from [Baidu](https://pan.baidu.com/s/1NWGIh3NsQvcI4w9I4f9jEA) (pw: b7ke)

## Run
our method: python train_Yeast.py
compared method(don't use markov method to train the transform matrix,but give it directly): python train_Yeast_nomarkov.py


# TLLDL to traditional LDL

## Datasets
Due to file size limitations, we only provide the Yeast_alpha dataset. 
Remaining datasets can be downloaded from [here](https://palm.seu.edu.cn/xgeng/LDL/download.htm), and put them in TLLDL-master/Use_TLLDL_in_TranditionalLDL

## iislld
run iislldDemo.m
if you want to run the bseline(traditional iislld), please set use=2(iislldDemo.m line 42); 
if you want apply our method to traditional iislld, please set use=1(iislldDemo.m line 42); 

## ptbayes
run ptbayesDemo.m
if you want to run the bseline(traditional ptbayes), please set use=2(ptbayesDemo.m line 40); 
if you want apply our method to traditional ptbayes, please set use=1(ptbayesDemo.m line 40); 

## bfgslld
run bfgslldDemo.m
if you want to run the bseline(traditional bfgslld), please set use=2(bfgslldDemo.m line 41 and bfgsProcess line 35); 
if you want apply our method to traditional bfgslld, please set use=1(bfgslldDemo.m line 41 and bfgsProcess line 35); 
