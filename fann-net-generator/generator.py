import os
import sys

import numpy as np
import sklearn
import pickle
import json

mlp = pickle.load(open(sys.argv[1], "rb"))
outnet = open(sys.argv[2], "w")

netdesc = {
    "norm": {
        "mode": "simple"
    },
    "layers": []
}

for layer in range(0, len(mlp.coefs_)):
    layerdesc = {
        "size": len(mlp.intercepts_[layer]),
        "activation": mlp.activation,
        "weights": np.transpose(mlp.coefs_[layer], (1, 0)).tolist(),
        "bias": mlp.intercepts_[layer].tolist()
    }
    netdesc["layers"].append(layerdesc)

netdesc["layers"][-1]["activation"] = mlp.out_activation_


scaler_file = sys.argv[1][:-4] + "_scaler.sav"
if os.path.exists(scaler_file):
    scaler = pickle.load(open(scaler_file, "rb"))
    netdesc["norm"]["mode"] = "minmax"
    netdesc["norm"]["min_params"] = list(scaler.data_min_)
    netdesc["norm"]["max_params"] = list(scaler.data_max_)


print(json.dumps(netdesc, indent=2), file=outnet)
