import re,string
import pandas as pd
with open('~/sherlock.txt','r') as f:
  contents=f.read()
  contents=contents.translate(None,string.punctuation).lower()#lower all the characters
  words=contents.split()
d={}
f.close()

for word in words:
    temp=word.lower()
    regex=re.compile('[%s]' %re.escape((string.punctuation)))
    temp=regex.sub('',temp)
    if word in d:
        d[temp]+=1
    else:
        d[temp]=1

for k in sorted(d,key=d.get,reverse=True):
    print(k,d[k])
    
'''
df=pd.DataFrame(freq)
print(df.sort_values([1]))   #arranged according to keys ascending
print(df.sort_values([0]))   #arranged according to values ascending
print(df.sort_values([0],ascending=False))     #arranged according to values descending
'''

        
