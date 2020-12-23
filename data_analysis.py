import numpy as np
import matplotlib.pyplot as plt
from fof_kdtree import FoF


candidate_list = FoF(galaxy_arr, luminous_galaxy_data, max_velocity=2000, linking_length_factor=0.4, virial_radius=2)

# cleaned_candidates = {}
# for k,v in candidate_list.items():
#     cleaned_k, cleaned_v = interloper_removal(k,v)
#     cleaned_candidates[cleaned_k] = cleaned_v

# virial_masses = {}
# for center,candidate in candidate_list.items():
#     mass, vel_disp, radius = vm_estimator(candidate)
#     virial_masses[center] = mass
# print(virial_masses)




# -----
# virial_masses (M_sun)
# {(150.10754288169696, 2.5575009868093432, 0.502, -20.68, 831036.0, 115644.98275119808): 5.52462884e+15, 
# (149.5197754535226, 1.83467112187136, 0.5025, -20.505, 359023.0, 115729.9052420429): 6.17663675e+15}

# virial_radius and escape_velocity
# {(150.2773967224679, 1.7571426724087982, 0.5, -20.906, 309409.0, 115304.79153846153): (<Quantity 3.95035035 Mpc>, <Quantity 7097.60005292 km / s>), 
# (150.11257822273575, 2.5560953018745893, 0.5, -23.613, 827001.0, 115304.79153846153): (<Quantity 3.75294887 Mpc>, <Quantity 6572.29537795 km / s>)}