from infer import InferenceHelper, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from models import UnetAdaptiveBins
import model_io
import numpy as np
import torch
import torch.nn as nn

infer_helper = InferenceHelper(dataset="nyu")


# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

plt.imshow(predicted_depth[0][0], cmap="plasma")
plt.show()
