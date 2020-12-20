import sys

import numpy as np
import sklearn
import pickle
import json

mlp = pickle.load(open(sys.argv[1], "rb"))
outnet = open(sys.argv[2], "w")

netdesc = {"layers": []}

for layer in range(0, len(mlp.coefs_)):
    layerdesc = {
        "size": len(mlp.intercepts_[layer]),
        "activation": mlp.activation,
        "weights": np.transpose(mlp.coefs_[layer], (1, 0)).tolist(),
        "bias": mlp.intercepts_[layer].tolist()
    }
    netdesc["layers"].append(layerdesc)

netdesc["layers"][-1]["activation"] = mlp.out_activation_

print(json.dumps(netdesc, indent=2), file=outnet)
