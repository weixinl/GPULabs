import numpy as np
import matplotlib.pyplot as plt

# dim_list = [500, 1000]
# cpu_list = [178.660000, 746.140000]
# gpu_list = [2.920000, 9.940000]
# speedup_list=[]
# dim_text_list = []
# for i in range(2):
#     speedup_list.append(float(cpu_list[i])/gpu_list[i])
#     dim_text_list.append(str(dim_list[i]))
# plt.xlabel("dimension")
# plt.ylabel("speedup")
# plt.bar(dim_text_list,speedup_list)
# plt.savefig("imgs/experiment1.jpg")

iter_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
cpu_list = [8.880000, 19.360000, 37.020000, 66.390000, 118.490000, 242.740000, 478.490000]
gpu_list = [0.500000, 0.620000, 0.790000, 1.170000, 1.930000, 3.470000, 6.520000]
speedup_list=[]
iter_text_list = []
for i in range(7):
    speedup_list.append(float(cpu_list[i])/gpu_list[i])
    iter_text_list.append(str(iter_list[i]))
plt.xlabel("iteration")
plt.ylabel("speedup")
plt.bar(iter_text_list,speedup_list)
plt.savefig("imgs/experiment2.jpg")