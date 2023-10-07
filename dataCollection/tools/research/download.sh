#!/bin/bash

RESEARCH_DATASETS=(
    "devign" "sap" "bigvul" "secbench" 
    "d2a" "codexglue" "reveal" "ivdetect" 
    "deepwukong"
)

for dataset in "${RESEARCH_DATASETS[@]}"; do
    dir=../../data
    if [[ ! -e $dir/$dataset/ ]]; then
        mkdir $dir/$dataset/
        echo "Creating $dir..."
    fi
done

#bigvul
wget -O ../../data/bigvul/big-vul-msr20.csv https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/raw/master/all_c_cpp_release2.0.csv
#secbench
wget -O ../../data/secbench/secbench.csv https://github.com/TQRG/secbench/raw/master/dataset/secbench.csv
mv ../../data/secbench/secbench.csv ../../data/secbench/github-secbench-patches.csv
#sap
wget -O ../../data/sap/pontas-sap-msr19.csv https://github.com/SAP/project-kb/raw/master/MSR2019/dataset/vulas_db_msr2019_release.csv
echo "$(echo -n 'cve_id,project,sha,type\n'; cat ../../data/sap/pontas-sap-msr19.csv)" > ../../data/sap/pontas-sap-msr19.csv
#devign
gdown -O ../../data/devign/projects/ https://drive.google.com/uc\?id\=1Nk_U52_gVHYfnNk-pcXlnxssOBrmSllV

gdown -O ../../data/devign/projects/ https://drive.google.com/uc\?id\=1RhyA-cZl2oiNb-IJOHYw4waBgvLzViTr

#d2a
D2A_BASE_URL="https://dax-cdn.cdn.appdomain.cloud/dax-d2a/1.1.0"

# Declare an array with the dataset filenames
datasets=("ffmpeg.tar.gz" "httpd.tar.gz" "libav.tar.gz" "libtiff.tar.gz" "nginx.tar.gz" "openssl.tar.gz" "splits.tar.gz" "d2a_leaderboard_dat.tar.gz")

echo "Available datasets for download:"
# Print available datasets with indices
for i in "${!datasets[@]}"; do
  echo "$i) ${datasets[$i]}"
done

# Ask user for dataset selection
read -p "Enter the number of the dataset you want to download (e.g. 0 for ffmpeg.tar.gz): " choice

# Check if choice is valid
if [[ "$choice" -ge 0 && "$choice" -lt "${#datasets[@]}" ]]; then
  # Use wget to download the selected dataset
  wget "${D2A_BASE_URL}/${datasets[$choice]}"
  echo "Downloaded ${datasets[$choice]}"
else
  echo "Invalid choice."
fi

#codexglue(defect detection)
gdown -O ../../data/codexglue/ https://drive.google.com/uc\?id\=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

#reveal
#data from IVDetect for the official repo's data link broken
#https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch
gdown --folder -O ../../data/reveal/ https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy

#ivdetect
#example test dataset
#https://drive.google.com/file/d/1LHOC4JDpnQ7gWnEHGfc4soQYHAPomlNp/view?usp=sharing
gdown -O ../../data/ivdetect https://drive.google.com/uc\?id\=1uMnm7_W9DgXN4AbJ0iUir052H1AF4hA1
################################
#The Dataset used in the ivdetect:
#Fan et al.[1]: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing
#Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy
#FFMPeg+Qemu [3]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

#deepwukong
wget -O ../../data/deepwukong/Data.7z "https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50&download=1"
