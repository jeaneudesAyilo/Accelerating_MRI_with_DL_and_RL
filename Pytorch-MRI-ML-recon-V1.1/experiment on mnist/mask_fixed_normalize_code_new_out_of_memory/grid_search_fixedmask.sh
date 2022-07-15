#!/usr/bin/bash

#-------------------------------
#Args pass EXP & CONFIG
#-------------------------------


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--experiment) EXP="$2"; shift ;;
        -c|--config) CONFIG="$2";shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "[INFO] EXPERIMENT: $EXP Launched"
echo "[INFO] CONFIG  : $CONFIG"

#-------------------------------
#Hyperparameters
#-------------------------------
#https://www.tutorialkart.com/bash-shell-scripting/bash-loop-through-indices-of-array/
#https://www.baeldung.com/linux/csv-parsing


#-------------------------------
#Experiment Names
#-------------------------------

DIR=config_$EXP
HYPER='./hyperp_train_fixedmask.sh'
now=$(date +"%y%m%d-%H%M")
SLEEP_TIME="0.3s"


if [ -d $DIR ]; then
    echo "Deleting old dir $DIR!"
    rm -rf $DIR
fi

mkdir $DIR

#-------------------------------
#Run Parameter search
#-------------------------------

counter=0

while read line
do

    echo "#############################"
    echo "$line"


    new_config=$DIR/$EXP.$counter.json
    new_sbatch=$DIR/$EXP.$counter.sh
   
    echo "Creating $counter config $new_config !"
    cp -rf $CONFIG  $new_config

    echo "Creating $counter file to sbatch $new_sbatch"   
    cp -rf $HYPER $new_sbatch
    
    export line
    export new_config
    python ./modify_dico_fixedmask_out_of_memory.py  

    sed -i  -e "s;--CONFIG--;$new_config;"  $new_sbatch

    if [[ $((counter%9)) -eq 0 ]]; then
    sleep $SLEEP_TIME
    fi

    counter=$((counter+1))

done < /data1/home/jean-eudes.ayilo/Pytorch-MRI-ML-recon-V1.1/mnist_experiment/mask_fixed_normalize_code_new/result/out_of_memory.csv

#executer : source grid_search_fixedmask.sh -e learn_fixedmask -c config_fixedmask.json
