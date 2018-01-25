import timeit
start=timeit.default_timer()
n=1
n_old=1
print('1 1')
print('2 1')
for i in range(3,11):
    new=n+n_old
    n_old=n
    n=new
    print('%d %d'%(i,new))
    
stop=timeit.default_timer()
print(stop-start)
    