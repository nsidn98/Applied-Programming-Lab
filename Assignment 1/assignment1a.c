#include<stdio.h>
#include<time.h>
int main(){
  clock_t start,end;
  start=clock();
  int i;
  int n=1;
  int n_old=1;
  int new;
  double time_used;
  printf("1 %d\n",n);
  printf("2 %d\n",n_old);
  for(i=3;i<=10;i++){
    new=n+n_old;
    n_old=n;
    n=new;
    printf("%d %d\n",i,new);
  }
  end=clock();
  time_used=((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Time taken=%lf\n",time_used);
}