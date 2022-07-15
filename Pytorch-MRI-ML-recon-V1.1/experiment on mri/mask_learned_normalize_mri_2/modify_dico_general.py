import os
import json
import re

def convert_list(a): #a is a string like "[3,3]" or ["3", "3"]
    if type(a) == str:
        a = a.replace("[","") ; a = a.replace("]","")
        return [int(dim) for dim in  a.split(",")]
    elif type(a) == list:
        return [int(dim) for dim in a]

##Cette demarche est un peu derisoire, car a chaque fois qu'on aura besoin d'ajouter un nouveau hyperparametre dans le gridsearch, faudra modifier ce fichier en ajoutant la nouvelle variable exportee

with open(os.environ["new_config"], "r") as fp:
     configuration = json.load(fp)
        
##convert the exported variables into their right type      

#since os.environ["kernel"] is of the form  "[3,3]", one need to put it in the normal form of list, ie [3,3]
#print('this is os.environ["mask_dim"] :' ,os.environ["mask_dim"])
configuration["mask_dim"]= convert_list(os.environ["mask_dim"])
configuration["acc"]= float(os.environ["acc"])
configuration["spirit_block"]= int(os.environ["bloc"])
configuration["kernel1"]= convert_list(os.environ["kernel"])
configuration["std_noise"]= float(os.environ["std_noise"])
configuration["lr_mask"]= float(os.environ["lr_mask"])
configuration["lr_other"]= float(os.environ["lr_other"])
configuration["acs_type"]= os.environ["acs_type"]


##une fois la modification faite, sauvegarder le nouveau dico
with open(os.environ["new_config"], "w") as fp:
     json.dump(configuration, fp)        
        
#https://stackoverflow.com/questions/17435056/read-bash-variables-into-a-python-script
#https://stackoverflow.com/questions/16618071/can-i-export-a-variable-to-the-environment-from-a-bash-script-without-sourcing-i