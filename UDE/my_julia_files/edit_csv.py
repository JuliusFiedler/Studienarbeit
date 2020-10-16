# %%
import csv
import numpy as np
import pandas as pd

# %%
folder = 'C:\\Users\\Julius\\Documents\\Studium_Elektrotechnik\\Studienarbeit\\github\\Studienarbeit\\Latex\\RST-DiplomMasterStud-Arbeit\\images\\'
name = "volterra_tspan_variation"
csv_name = folder + 'errors_' + name 

# %%
data =[]
with open(csv_name + '.csv', 'r', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
    for row in spamreader:
        data.append(row)
# print(data)
data_new = np.copy(data)
data_new[2] = data[3]
data_new[3] = data[5]
data_new[4] = data[2]
data_new[5] = data[4]


# %%
with open(csv_name + "2" + '.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerows(data_new)

# %%
data =[]
with open(csv_name + '.csv', 'r', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
    a = 0
    for row in spamreader:
        # if a==0:
        #     data = np.array(row)
        #     a=1
        # data = np.concatenate((data, row), axis=0)
        data.append(np.asarray(row, dtype=str))
npdata = np.ones((len(data), len(data[0])) ,dtype=str)
# %%

# %%
