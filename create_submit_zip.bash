#!/bin/bash

ND_EN_JSON="./output/baseline_multi_english_Nov-30_16-41-39-836930/1133/nugget_english_test_submission.json"
ND_CN_JSON="./output/baseline_multi_chinese_Nov-30_19-13-59-725318/2544/nugget_chinese_test_submission.json"

DQ_EN_JSON="./output/baseline_multi_english_Nov-30_16-41-39-836930/6006/quality_english_test_submission.json"
DQ_CN_JSON="./output/baseline_multi_chinese_Nov-30_19-13-59-725318/4464/quality_chinese_test_submission.json"

mkdir -p ./submit/SLSTC_ND/run1
cp $ND_EN_JSON ./submit/SLSTC_ND/run1/en.json
cp $ND_CN_JSON ./submit/SLSTC_ND/run1/cn.json
cd ./submit
zip -r ./SLSTC_ND.zip ./SLSTC_ND

cd ../

mkdir -p ./submit/SLSTC_DQ/run1
cp $DQ_EN_JSON ./submit/SLSTC_DQ/run1/en.json
cp $DQ_CN_JSON ./submit/SLSTC_DQ/run1/cn.json
cd ./submit
zip -r ./SLSTC_DQ.zip ./SLSTC_DQ
