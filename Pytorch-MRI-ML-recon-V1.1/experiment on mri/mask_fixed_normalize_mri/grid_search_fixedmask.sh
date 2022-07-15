#!/usr/bin/bash

#-------------------------------
#Args pass EXP & CONFIG
#-------------------------------


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--experiment) EXP="$2"; shift ;;
        -c|--config) CONFIG="$2";shift ;;
        -m|--list_mask_df) LIST_MASK_DF="$2";shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "[INFO] EXPERIMENT: $EXP Launched"
echo "[INFO] CONFIG  : $CONFIG"
echo "[INFO] LIST_MASK_DF  : $LIST_MASK_DF"

#-------------------------------
#Hyperparameters
#-------------------------------
#https://www.tutorialkart.com/bash-shell-scripting/bash-loop-through-indices-of-array/
#https://www.baeldung.com/linux/csv-parsing

#LIST_MASK_DF is a df  with 2 columns, that we load without the head giving the columns names. A better solution would be 
#execute the py script which generates that df, and give the number of random mask as argument (argparse). then get the path to the dfcsv
arr_mask_typ=( $(tail -n +2 $LIST_MASK_DF | cut -d ',' -f1) )
arr_mask_seed=( $(tail -n +2 $LIST_MASK_DF | cut -d ',' -f2) )

LR_OTHER=(1e-2 1e-3 1e-4)
KERNEL=([11,11] [13,13])
SPIRIT_BLOCK=(1 5 10)
ACS_TYPE=(no square)

#-------------------------------
#Experiment Names
#-------------------------------
counter=0
#DIR=config_$EXP
DIR=config_$EXP
HYPER='./hyperp_train_fixedmask.sh'
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
for index in "${!arr_mask_typ[@]}" ; do
for lr_other in ${LR_OTHER[@]} ; do
for kernel in ${KERNEL[@]} ; do
for bloc in ${SPIRIT_BLOCK[@]} ; do
for acs_type in ${ACS_TYPE[@]} ; do

#use the index to reach the list a the same position. we use a seed for reproductibility
mask_typ=${arr_mask_typ[$index]}
mask_seed=${arr_mask_seed[$index]}

echo "#############################" 
echo "${arr_mask_typ[$index]}"
echo "${arr_mask_seed[$index]}" 
echo "Learning Rate: $lr_other"
echo "kernel: $kernel"
echo "bloc: $bloc"
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
##export the grid variables in order to incorpore them into the config file
export mask_typ
export mask_seed
export new_config
export kernel
export bloc
export lr_other
export acs_type

#sed -i  -e "s;\"exp_name\".*;\"exp_name\": \"$new_exp\",;" $new_config
#sed -i  -e "s;\"kernel1\".*;\"kernel1\": $kernel,;" $new_config
#sed -i  -e "s;\"spirit_blocks\".*;\"spirit_blocks\": $s_blocks,;" $new_config
#sed -i  -e "s;\"lr_other\".*;\"lr_other\": $lr_other,;" $new_config

python modify_dico_fixedmask.py


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
#logpath="slurm/$new_exp"
#mkdir -p $logpath
#logfile="$logpath/$EXP.$counter.out"

#echo $logfile

#-------------------------------
#waiting, scheduler after 10 jobs
#-------------------------------
#echo "This is the directory"

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