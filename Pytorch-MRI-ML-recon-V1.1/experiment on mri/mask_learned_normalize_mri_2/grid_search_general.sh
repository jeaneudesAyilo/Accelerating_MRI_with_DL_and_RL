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

MASK_DIM=([121,145] [4,4])
ACCELERATION=(0.125 0.25)
SPIRIT_BLOCK=(1 5 10)
KERNEL=([11,11] [13,13])
STD_NOISE=(0.0175 0.05)
LR_MASK=(1e-2 1e-3 1e-4)
LR_OTHER=(1e-2 1e-3 1e-4)
ACS_TYPE=(no square)


#-------------------------------
#Experiment Names
#-------------------------------
counter=0
#DIR=config_$EXP
DIR=config_$EXP
HYPER='./hyperp_train_general.sh'
now=$(date +"%y%m%d-%H%M")
SLEEP_TIME="1s"



if [ -d $DIR ]; then
    echo "Deleting old dir $DIR!"
    rm -rf $DIR
fi

mkdir $DIR

#-------------------------------
#Run Parameter search
#-------------------------------


for mask_dim in ${MASK_DIM[@]} ; do
for acc in ${ACCELERATION[@]} ; do
for bloc in ${SPIRIT_BLOCK[@]} ; do
for kernel in ${KERNEL[@]} ; do
for std_noise in ${STD_NOISE[@]} ; do
for lr_mask in ${LR_MASK[@]} ; do
for lr_other in ${LR_OTHER[@]} ; do
for acs_type in ${ACS_TYPE[@]} ; do


echo "#############################" 
echo "mask_dim : $mask_dim  acs_type : $acs_type acc : $acc bloc : $bloc  kernel : $kernel std_noise : $std_noise lr_mask : $lr_mask lr_other : $lr_other"

echo " "

#create new config file, new sbatch file and new experiment name

new_config=$DIR/$EXP.$counter.json
new_sbatch=$DIR/$EXP.$counter.sh

#new_config=$DIR/lr_other_$lr_other-s_blocks_$s_blocks-kernel_$kernel.json
#new_sbatch=$DIR/lr_other_$lr_other-s_blocks_$s_blocks-kernel_$kernel.sh
#new_exp=$EXP/lr_other_$lr_other-s_blocks_$s_blocks-kernel_$kernel

echo "Creating $counter config $new_config !"

cp -rf $CONFIG  $new_config

echo "Creating $counter file to sbatch $new_sbatch"
cp -rf $HYPER $new_sbatch

#-------------------------------
#config
#-------------------------------
##export the grid search variables in order to incorpore them into the config file


export new_config
export mask_dim
export acc
export bloc
export kernel
export std_noise
export lr_mask
export lr_other
export acs_type
#sed -i  -e "s;\"exp_name\".*;\"exp_name\": \"$new_exp\",;" $new_config
#sed -i  -e "s;\"kernel1\".*;\"kernel1\": $kernel,;" $new_config
#sed -i  -e "s;\"spirit_blocks\".*;\"spirit_blocks\": $s_blocks,;" $new_config
#sed -i  -e "s;\"lr_other\".*;\"lr_other\": $lr_other,;" $new_config

python ./modify_dico_general.py


#trouver un moyen de mettre le dirname qu'il y a dans le .py a la place de new_config

#-------------------------------
#sbatch file
#-------------------------------

sed -i  -e "s;--CONFIG--;$new_config;"  $new_sbatch


#-------------------------------
#slurm log
#-------------------------------

#logpath="slurm/log/$new_exp/$now", logfile="$logpath/$EXP.$counter.out"

#logpath="slurm/log/$new_exp/$now"
#logpath="slurm/mask_dim_$mask_dim-acc_$acc-bloc_$bloc-kernel_$kernel-std_noise_$std_noise-lr_mask_$lr_mask-lr_other_$lr_other"
#mkdir -p $logpath
#logfile="$logpath/$EXP.$counter.out"

#echo $logfile

#-------------------------------
#waiting, scheduler after 10 jobs
#-------------------------------

#sbatch -o $logfile $new_sbatch
counter=$((counter+1))

sleep 1 # pause to be kind to the scheduler

if [[ $((counter%9)) -eq 0 ]]; then
sleep $SLEEP_TIME
fi

done
done
done
done
done
done
done
done
#executer : source grid_search_general.sh -e learn_general -c config_general.json

