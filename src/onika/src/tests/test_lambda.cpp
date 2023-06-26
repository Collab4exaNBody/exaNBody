#include <onika/lambda_tools.h>
#include <onika/task/task_pool_item.h>

#include <iostream>

using namespace onika;
using Allocator = typename task::TaskPoolItem::TaskAllocator;
Allocator global_task_allocator{};

int myfunction(double x, float y)
{
  return x<y ? 3 : 5;
}

int main()
{
  std::cout<<"myfunction is compatible with int(double,float) : "<< std::boolalpha << lambda_is_compatible_with_v<decltype(myfunction),int,double,float> << "\n";
  std::cout<<"myfunction is compatible with int(double,int) : "<< std::boolalpha << lambda_is_compatible_with_v<decltype(myfunction),int,double,int> << "\n";
  std::cout<<"myfunction is compatible with int(int,int) : "<< std::boolalpha << lambda_is_compatible_with_v<decltype(myfunction),int,int,int> << "\n";
  std::cout<<"myfunction is compatible with void(double,float) : "<< std::boolalpha << lambda_is_compatible_with_v<decltype(myfunction),void,double,float> << "\n";
  std::cout<<"myfunction is compatible with int(double) : "<< std::boolalpha << lambda_is_compatible_with_v<decltype(myfunction),int,double> << "\n";

  int x=3;
  auto f1 = [x]()->void { std::cout<<"x="<<x<<std::endl; };
  auto t1 = task::TaskPoolItem::lambda( global_task_allocator , f1 );
  t1->execute(); t1->free();
  
  auto f2 = [x](float y, double z)->void { std::cout<<"x="<<x<<" y="<<y<<" z="<<z<< std::endl; };
  auto t2 = task::TaskPoolItem::lambda( global_task_allocator , f2 );
  t2->execute( 1.23f , 9.87 );
  t2->execute( 1.23f , 9 );
  t2->free();

}

