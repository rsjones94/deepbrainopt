
import s4l_v1.model as ml
import numpy as np

import lead_specs as ls


lead_name = 'left_electrode'
lead_params = ls.MEDTRONIC_B33005

ents = ml.AllEntities()
lead_ents = [i for i in ents if lead_name in i.Name]
lead_ents_names = [i.Name for i in lead_ents]
lead_ents_nums = [int(i.split(' ')[-1]) if len(i.split(' ')) > 1 else 0 for i in lead_ents_names]

#sort the lists by names
lead_ents_nums, lead_ents = zip(*sorted(zip(lead_ents_nums, lead_ents)))

for ent,params in zip(lead_ents,lead_params):
    ent.Name = params['name']
    ent.MaterialName = params['material']
    
    rc = np.array(params['color']) / 2**8
    ent.Color = ml.Color(rc[0], rc[1], rc[2], rc[3])