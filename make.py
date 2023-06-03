from glob import glob
import os
train_path = os.getcwd()+'/train/'
val_path = os.getcwd()+'/val/'
label = 0
arr = []
filename = []

for root, ds, fs in os.walk(train_path):
    for d in ds:
        arr.append(d)
    
    arr.sort()
    for i in range(0, 1000):
        filepath = os.path.join(root, arr[i])
        #print(filepath)
        list = os.listdir(filepath)
        for j in range(len(list)):
            filename.append("train/"+arr[i]+"/"+list[j]+"\t"+str(i)+"\n")
    with open('train_map.txt', 'w') as f:
        f.writelines(filename) 
    break

arr = []
filename = []

for root, ds, fs in os.walk(val_path):
    for d in ds:
        arr.append(d)
    
    arr.sort()
    for i in range(0, 1000):
        filepath = os.path.join(root, arr[i])
        #print(filepath)
        list = os.listdir(file-path)
        for j in range(len(list)):
            filename.append("val/"+arr[i]+"/"+list[j]+"\t"+str(i)+"\n")
    with open('val_map.txt', 'w') as f:
        f.writelines(filename) 
    break
