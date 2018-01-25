import math
import timeit
start=timeit.default_timer()
pi=math.pi
lst=[]
lst.append(0.2000)
for i in range(1,1000):
    new=(((lst[i-1]+pi)*100)-int((lst[i-1]+pi)*100))
    #new=new-new%0.0001
    lst.append(new)

formatted_list=[[ '%.4f' % elem for elem in lst ]]
stop=timeit.default_timer()
print(stop-start)