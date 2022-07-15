import os
import json
import re
import parse

def convert_list(a): #a is a string like "[3,3]" or ["3", "3"]
    if type(a) == str:
        return [int(dim) for dim in  re.sub('\[|\]','',a).split(",")]
    elif type(a) == list:
        return [int(dim) for dim in  a]

##Cette demarche est un peu derisoire, car a chaque fois qu'on aura besoin d'ajouter un nouveau hyperparametre dans le gridsearch, faudra modifier ce fichier en ajoutant la nouvelle variable exportee

with open(os.environ["new_config"], "r") as fp:
     configuration = json.load(fp)
        

#line est une expression du genre config_name = "./mask_dim_4_4_acc_0.125_spirit_bloc_10_std_noise_0.03_kernel_5_5_lr_mask_0.0001_lr_other_0.01"
##et on veut extraire de line les parametres pour le fichier de config

format_string_config = "./mask_dim_{mask_dim[0]}_{mask_dim[1]}_acc_{acc}_spirit_bloc_{bloc}_std_noise_{std_noise}_kernel_{kernel[0]}_{kernel[1]}_lr_mask_{lr_mask}_lr_other_{lr_other}"

#format_string_exp_name = "spirit_bloc : {bloc} ; k_local {k_local};norm {norm} ; kernel : {kernel}; std_noise : {std_noise}; lr_mask : {lr_mask}; lr_other : {lr_other}"

parsed_config = parse.parse(format_string_config, os.environ["line"])  

parsed_config_named = parsed_config.named


configuration["mask_dim"]= convert_list(list(parsed_config_named["mask_dim"].values()))
configuration["acc"]= float(parsed_config_named["acc"])
configuration["spirit_block"]= int(parsed_config_named["bloc"])
configuration["kernel1"]=  convert_list(list(parsed_config_named["kernel"].values()))
configuration["std_noise"]= float(parsed_config_named["std_noise"])
configuration["lr_mask"]= float(parsed_config_named["lr_mask"])
configuration["lr_other"]= float(parsed_config_named["lr_other"])

##une fois la modification faite, sauvegarder le nouveau dico
with open(os.environ["new_config"], "w") as fp:
     json.dump(configuration, fp)        
        
#https://stackoverflow.com/questions/17435056/read-bash-variables-into-a-python-script
#https://stackoverflow.com/questions/16618071/can-i-export-a-variable-to-the-environment-from-a-bash-script-without-sourcing-i