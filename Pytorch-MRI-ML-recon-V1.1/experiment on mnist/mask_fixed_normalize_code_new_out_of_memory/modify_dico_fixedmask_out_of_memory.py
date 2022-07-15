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
        
#line est une expression du genre config_name = "./mask_typ_caipiranha_mask_seed_42083_mask_dim_4_4_spirit_bloc_10_kernel_5_5_lr_other_0.001"
##et on veut extraire de line les parametres pour le fichier de config


format_string_config = "./mask_typ_{mask_typ}_mask_seed_{mask_seed}_mask_dim_{mask_dim[0]}_{mask_dim[1]}_spirit_bloc_{spirit_block}_kernel_{kernel1[0]}_{kernel1[1]}_lr_other_{lr_other}"

#format_string_exp_name = "spirit_bloc : {bloc} ; k_local {k_local};norm {norm} ; kernel : {kernel}; std_noise : {std_noise}; lr_mask : {lr_mask}; lr_other : {lr_other}"

parsed_config = parse.parse(format_string_config, os.environ["line"])  

parsed_config_named = parsed_config.named

configuration["kernel1"]= convert_list(list(parsed_config_named["kernel1"].values()))
configuration["spirit_block"]= int(parsed_config_named["spirit_block"])
configuration["lr_other"]= float(parsed_config_named["lr_other"])
configuration["mask_typ"]= parsed_config_named["mask_typ"]
configuration["mask_seed"]= int(parsed_config_named["mask_seed"])

##une fois la modification faite, sauvegardee le nouveau dico
with open(os.environ["new_config"], "w") as fp:
     json.dump(configuration, fp)        
        
#https://stackoverflow.com/questions/17435056/read-bash-variables-into-a-python-script
#https://stackoverflow.com/questions/16618071/can-i-export-a-variable-to-the-environment-from-a-bash-script-without-sourcing-i