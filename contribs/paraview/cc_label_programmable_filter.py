# This file can be used from Paraview's programmable filter to transform cc_label property on a grid
# to CC properties, such as volume (sub cell count), using a python script as follows :
# Paraview programmable filter script

# Note : check [X] Copy Arrays to get original arrays

import numpy as np
input0 = inputs[0]
cc_label = input0.PointData["cc_label"]
cc_table = np.genfromtxt("cc_table.csv" , dtype=None, names=True, delimiter=';', autostrip=True)
cc_count = cc_table["count"]
cc_rank = cc_table["rank"]
N = cc_label.size
cc_volume = np.array( [] , dtype=np.uint32 )
cc_volume.resize(N)
cc_volume.reshape([N])
cc_procid = np.array( [] , dtype=np.uint32 )
cc_procid.resize(N)
cc_procid.reshape([N])
for i in range(N):
    assert i < cc_volume.size , "index %d out of cc_volume bounds (%d)" % (i,cc_volume.size)
    label_idx = int(cc_label[i])
    if label_idx >= 0:
      assert label_idx < cc_count.size , "%d out of cc_count bounds (%d)" % (label_idx,cc_count.size)
      assert label_idx < cc_rank.size , "%d out of cc_rank bounds (%d)" % (label_idx,cc_count.size)
      cc_volume[i] = cc_count[label_idx]
      cc_procid[i] = cc_rank[label_idx]
    else:
      cc_volume[i] = 0
output.PointData.append( cc_volume , "cc_volume" )
output.PointData.append( cc_procid , "cc_procid" )

