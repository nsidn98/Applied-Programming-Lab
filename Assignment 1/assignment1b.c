#include<stdio.h>
double pi=3.141592653589793;
int main(){
  double n[1000];
  int k;
  double new;
  //float pi=M_PI;
  n[0]=0.2;
  for(k=1;k<1000;k++){
    new=((n[k-1]+pi)*100);
    n[k]=new-(int)new;
  }
  
  for(k=0;k<1000;k++){
    printf("%lf\n",n[k]);
  }
}