import numpy as np
import numpy as np
import random


## Pick Random Subset
def random_subset(all_formulas, k_min=1, k_max=4):
    k = random.randint(k_min, min(k_max, len(all_formulas)))
    return random.sample(all_formulas, k)

## One-Hot Encoding
def encode_state(known, ALL_FORMULAS):
    OneHot_Map = {f:i for i,f in enumerate(ALL_FORMULAS)}
    v = np.zeros(len(ALL_FORMULAS))
    for f in known:
        v[OneHot_Map[f]] = 1
    return v

####################################################################################
