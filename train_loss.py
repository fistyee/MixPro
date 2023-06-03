import sys
aa=[]
import pdb
ans = []
with open('attn_cos.txt','r') as f:
    for line in f.readlines():
        data = line.split('\n\t')
        #print(data)
        if data[0][0:1] == 'G':#aradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0\n':
            continue
        for str in data:
            sub_str = str.split('\t')
            aa.append(sub_str)

    #print(aa[-10])a
    #print(aa)
    aa = list(reversed(aa))
    for i in range(9, len(aa), 31):
        ans.append(aa[i])
    loss = []
    #print(ans)
    for i in range(len(ans)-14):
        loss.append(float(ans[i][-3][5:9]))
    print(len(loss))
    loss = list(reversed(loss))
    print(loss)
