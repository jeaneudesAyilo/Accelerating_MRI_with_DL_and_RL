import os
import json
import re

def convert_list(a): #a is a string like "[3,3]"
     return [int(dim) for dim in  re.sub('\[|\]','',a).split(",")]

##Cette demarche est un peu derisoire, car a chaque fois qu'on aura besoin d'ajouter un nouveau hyperparametre dans le gridsearch, faudra modifier ce fichier en ajoutant la nouvelle variable exportee

with open(os.environ["new_config"], "r") as fp:
     configuration = json.load(fp)
        
##convert the exported variables into their right type      

#since os.environ["kernel"] is of the form  "[3,3]", one need to put it in the normal form of list, ie [3,3]

configuration["kernel1"]= convert_list(os.environ["kernel"]) 
configuration["spirit_block"]= int(os.environ["bloc"])
configuration["lr_other"]= float(os.environ["lr_other"])
configuration["mask_typ"]= os.environ["mask_typ"]
configuration["mask_seed"]= int(os.environ["mask_seed"])

##une fois la modification faite, sauvegardee le nouveau dico
with open(os.environ["new_config"], "w") as fp:
     json.dump(configuration, fp)        
        
#https://stackoverflow.com/questions/17435056/read-bash-variables-into-a-python-script
#https://stackoverflow.com/questions/16618071/can-i-export-a-variable-to-the-environment-from-a-bash-script-without-sourcing-i