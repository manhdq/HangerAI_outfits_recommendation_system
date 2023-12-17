# %%
import numpy as np

# %%
a = np.random.randint(9, size=(4, 4))
a

# %%
indx, indy = np.triu_indices(4, k=1)
indx, indy

# %%
a[indx], a[indy]

# %%
a[indx] * a[indy]

# %%
import os.path as osp
