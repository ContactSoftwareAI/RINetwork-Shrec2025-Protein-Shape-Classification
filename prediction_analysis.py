import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("test_set_2_run1.csv")
#data = pd.read_csv("test_set_2_run2.csv")
filenames = data["anonymised_protein_id"]
labels = data["predicted_label"]



#data = pd.read_csv("test.csv")
#data = pd.read_csv("test2.csv")
#filenames = data["filename"]
#labels = data["class_id"]


plt.figure(1)
plt.hist(labels, bins = 97)
plt.show()