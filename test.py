import numpy as np

from sleepens import StaticSleepEnsemble

se = StaticSleepEnsemble(verbose=2)

file1 = "/mnt/d/Documents/Research/sleepens/data/mat/data1.mat"
file2 = "/mnt/d/Documents/Research/sleepens/data/mat/data2.mat"
file3 = "/mnt/d/Documents/Research/sleepens/data/mat/data3.mat"
file4 = "/mnt/d/Documents/Research/sleepens/data/mat/data4.mat"
file5 = "/mnt/d/Documents/Research/sleepens/data/mat/data5.mat"
file6 = "/mnt/d/Documents/Research/sleepens/data/mat/data6.mat"
file7 = "/mnt/d/Documents/Research/sleepens/data/mat/test.mat"
file8 = "/mnt/d/Documents/Research/sleepens/data/mat/train1.mat"
file9 = "/mnt/d/Documents/Research/sleepens/data/mat/train2.mat"
file10 = "/mnt/d/Documents/Research/sleepens/data/mat/train3.mat"
file11 = "/mnt/d/Documents/Research/sleepens/data/mat/train4.mat"
files = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11]

ds_list = []
for file in files:
	ds = se.process(file, labels=True)
	ds_list.append(ds)
	ds.write("/mnt/d/Documents/Research/sleepens/data")


big_ds = ds_list[0]
for ds in ds_list[1:]:
	big_ds.append(ds)
print("WRITING")
big_ds.name = 'big_ds'
big_ds.write("/mnt/d/Documents/Research/sleepens/data")
