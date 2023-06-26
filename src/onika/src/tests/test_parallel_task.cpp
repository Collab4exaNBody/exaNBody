#include <iostream>
#include <fstream>
#include <onika/stream_utils.h>
#include <onika/oarray_stream.h>
#include <onika/debug.h>
#include <onika/force_assert.h>
#include <onika/task/parallel_execution.h>
#include <utility>

namespace onika
{
  template<class T, class U> struct PrintableFormattedObject< std::pair<T,U> >
  {
    const std::pair<T,U>& v;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const { return out << v.first << "," << v.second; }
  };
  
  template<class T, class U> static inline PrintableFormattedObject< std::pair<T,U> > format_pair(const std::pair<T,U>& v) { return {v}; }
}

void print_parrallel_task( const onika::task::ParallelTask & pt )
{
  using namespace onika;

  int nd = pt.span().ndims;
  std::cout<<"ND = "<<nd<<std::endl;

  assert( pt.stencil().m_ndims == nd );
  std::cout<<"stencil low   = "<<format_array(pt.stencil().m_low,nd)<<std::endl;
  std::cout<<"stencil size  = "<<format_array(pt.stencil().m_size,nd)<<std::endl;
  std::cout<<"stencil cells = "<<pt.stencil().nb_cells()<<std::endl;
  std::cout<<"stencil bits  = "<<int(pt.stencil().m_nbits)<<std::endl;
  size_t n = pt.stencil().nb_cells();
  auto bsize = pt.stencil().box_size();
  for(size_t j=0;j<n;j++)
  {
    auto c = index_to_coord( j , bsize );
    c = array_add( c , pt.stencil().m_low );
    auto ro = pt.stencil().ro_mask( j );
    auto rw = pt.stencil().rw_mask( j );
    if(ro!=0 || rw!=0) std::cout<<"\t"<<format_array(c.data(),nd)<<" ro="<<ro<<" rw="<<rw<<std::endl;
  }
}

template<class SpanT>
void test_span_coord_conv( SpanT sp )
{
  using namespace onika;
  
  constexpr size_t ND = sp.ndims;
  dac::abstract_box_span_t span ( sp );
  size_t n = span.nb_cells();

  std::cout<<"test span : low="<<format_array(span.lower_bound_nd<ND>())<<" size="<<format_array(span.box_size_nd<ND>());
  if(span.border<1000) std::cout<<" border="<<span.border;
  std::cout<<" n="<<n<<" nb="<<span.box_cells()<<" nh="<<span.hole_cells()<<" ..."<<std::flush;
  
  for(size_t i=0;i<n;i++)
  {
    auto c = span.index_to_coord<ND>(i);
    //std::cout<<std::endl<<"\t"<<format_array(c.data(),ND)<<std::flush;
    ONIKA_FORCE_ASSERT( span.inside(c) );
    ONIKA_FORCE_ASSERT( span.coord_to_index(c) == i );
  }

  std::cout<<" Ok"<<std::endl<<std::flush;
}

void flush_in_parallel_section()
{
  _Pragma("omp parallel")
  {
    _Pragma("omp single")
    {
      onika::task::default_ptask_queue().stop();
      _Pragma("omp taskgroup")
      {
        onika::task::default_ptask_queue().start();
      }
    }
  }
}

#define RUN_TEST(label) if( ( test==(++n) || test<0 ) && std::cout<<n<<" : "<<label<<"\n" ) if(test!=-2)

int main(int argc,char*argv[])
{
  using namespace onika;
  using onika::task::default_ptask_queue ;

  int test=-2;
  int n = 0;
  if(argc>1) test = std::atoi(argv[1]);

  std::cout<<"sizeof(ParallelTask) = " << sizeof( typename task::ParallelTask ) << std::endl;

  // 1.
  RUN_TEST("index to span 3D coord")
  {
    dac::abstract_box_span_t sp = dac::box_span_t<3> { {1,1,1} , {3,3,3} , 1 };
    size_t n = sp.nb_cells();
    std::cout<<"ndims="<<sp.ndims<<std::endl;
    std::cout<<"nb_cells = "<<n<< ", box_cells="<<sp.box_cells()<<", hole_cells="<<sp.hole_cells()<< std::endl;
    for(size_t i=0;i<n;i++)
    {
      auto c = sp.index_to_coord<3>( i );
      std::cout<<i<<" : "<< format_array(c) << std::endl;
      assert( sp.inside(c) );
    }
  }

  // 2.
  RUN_TEST("index to span 2D coord")
  {
    dac::box_span_t<2> span { {1,1} , { 6 , 1 } };
    dac::abstract_box_span_t sp = span;
    size_t n = sp.nb_cells();
    std::cout<<"ndims="<<sp.ndims<<std::endl;
    std::cout<<"nb_cells = "<<n<< ", box_cells="<<sp.box_cells()<<", hole_cells="<<sp.hole_cells()<< std::endl;
    for(size_t i=0;i<n;i++)
    {
      auto c = sp.index_to_coord<2>( i );
      std::cout<<i<<" : "<< format_array(c) << std::endl;
      assert( sp.inside(c) );
    }
  }

  // 3. 1D left-right
  RUN_TEST("1D left-right stencil")
  {
    using central = dac::stencil_element_t< dac::DataSlices<> , dac::DataSlices<dac::whole_t> >;
    using left    = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> , -1 >;
    using right   = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> ,  1 >;
    dac::stencil_t<central,left,right> stencil_1d{};
    std::vector<double> my_array(10,0.0);
    dac::box_span_t<1> span { {0} , { my_array.size() } };
    auto ac = make_access_controler( my_array , stencil_1d );
    
    auto && pt = 
    default_ptask_queue() <<
    onika_parallel_for( span , ac )
    {
      ONIKA_STDOUT_OSTREAM << "item at "<< item_coord[0] << " = " << ac << std::endl;
    };
    pt.finalize();
    print_parrallel_task( * pt.m_ptask );
  }

  // 4. 2D Heat equation like stencil
  RUN_TEST("2D heat stencil")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    using left           = dac::stencil_element_t< R , F , -1 ,  0 >;
    using right          = dac::stencil_element_t< R , F ,  1 ,  0 >;
    using bottom         = dac::stencil_element_t< R , F ,  0 , -1 >;
    using top            = dac::stencil_element_t< R , F ,  0 ,  1 >;
    dac::stencil_t<central,left,right,bottom,top> stencil_2d{};
    std::pair<double,double> my_array[8][8];
    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2> span { {0,0} , ac.size() };

    auto && pt = 
    default_ptask_queue() << 
    onika_parallel_for( span , ac )
    {
      ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
    };
    pt.finalize();
    print_parrallel_task( * pt.m_ptask );
  }

  // 5. 2D Heat equation like stencil with granularity = 2
  RUN_TEST("2D heat with granularity=2")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    using left           = dac::stencil_element_t< R , F , -1 ,  0 >;
    using right          = dac::stencil_element_t< R , F ,  1 ,  0 >;
    using bottom         = dac::stencil_element_t< R , F ,  0 , -1 >;
    using top            = dac::stencil_element_t< R , F ,  0 ,  1 >;
    dac::stencil_t<central,left,right,bottom,top> stencil_2d{};
    std::pair<double,double> my_array[16][16];
    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2 , 2 > span { {0,0} , ac.size() };

    default_ptask_queue() <<
    onika_parallel_for( span , ac )
    {
#     pragma omp critical(dbg_mesg)
      ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
    };
    //print_parrallel_task( * pt.ptask );

    flush_in_parallel_section();
  }

  // 6. 2D dual-data access
  RUN_TEST("2D dual data access with distinct stencils")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    using left           = dac::stencil_element_t< R , F , -1 ,  0 >;
    using right          = dac::stencil_element_t< R , F ,  1 ,  0 >;
    using bottom         = dac::stencil_element_t< R , F ,  0 , -1 >;
    using top            = dac::stencil_element_t< R , F ,  0 ,  1 >;

    using central2       = dac::stencil_element_t< dac::DataSlices<> , dac::DataSlices<dac::whole_t> >;
    using left2          = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> , -1 , -1 >;
    using right2         = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> ,  1 ,  1 >;
    
    dac::stencil_t<central,left,right,bottom,top> stencil1{};
    dac::stencil_t<central2,left2,right2> stencil2{};

    std::pair<double,double> my_array[8][8];
    int other_array[8][8];
    auto ac1 = make_access_controler( my_array , stencil1 );
    auto ac2 = make_access_controler( other_array , stencil2 );
    assert( ac1.size() == ac2.size() );
    dac::box_span_t<2> span { {0,0} , ac1.size() };

    using central3        = dac::stencil_element_t< dac::DataSlices<> , R >;
    dac::stencil_t<central3> stencil3{};
    auto ac3 = make_access_controler( my_array , stencil3 );
    dac::box_span_t<2> span2 { {1,1} , { 6 , 1 } };

    default_ptask_queue() << // attach to queue operator
      onika_parallel_for( span , ac1 , ac2 )
      {
        _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " : ac1 = " << format_pair(ac1) <<" , ac2 = "<< ac2 << std::endl;
      }
      >> // sequence-after operator
      onika_parallel_for( span2 , ac3 )
      {
        _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " : ac3 = " << ac3 << std::endl;
      }
    ;
      
    flush_in_parallel_section();
  }

  // 7.
  RUN_TEST("span coord conversion")
  {
    test_span_coord_conv( dac::box_span_t<3> { {1,1,1} , {25-2,25-2,26-2} , 1 } ); // from real world case microjet

    test_span_coord_conv( dac::box_span_t<2> { {0,0} , {1,1} } );
    test_span_coord_conv( dac::box_span_t<2> { {0,0} , {5,5} } );
    test_span_coord_conv( dac::box_span_t<2> { {0,0} , {5,5} , 1 } );
    test_span_coord_conv( dac::box_span_t<2> { {1,1} , {5,5} , 1 } );
    test_span_coord_conv( dac::box_span_t<3> { {0,0,0} , {1,1,1} } );
    test_span_coord_conv( dac::box_span_t<3> { {0,0,0} , {3,4,5} } );
    test_span_coord_conv( dac::box_span_t<3> { {1,0,1} , {5,4,3} } );
    test_span_coord_conv( dac::box_span_t<3> { {0,0,0} , {3,4,5} , 1 } );
    test_span_coord_conv( dac::box_span_t<3> { {1,0,1} , {5,4,3} , 1 } );
    test_span_coord_conv( dac::box_span_t<3> { {1,2,1} , {7,8,9} , 2 } );
  }

  // 8. detached co-parallel tasks
  RUN_TEST("2d pair accessor")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    dac::stencil_t<central> stencil_2d{};
    std::pair<double,double> my_array[8][8];
    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2 > span { {1,1} , {7,7} , 1 };
    
    auto && pt =
    default_ptask_queue() << onika_parallel_for( span , ac )
    {
      ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
    };
    pt.finalize();
    print_parrallel_task( * pt.m_ptask );

    flush_in_parallel_section();
  }

  // 9. local 2d stencil with granularity
  RUN_TEST("2D stencil with granularity")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    dac::stencil_t<central> stencil_2d{};
    std::pair<double,double> my_array[8][8];
    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2 , 2 > span { ac.size() };

    default_ptask_queue() <<
    onika_parallel_for( span , ac )
    {
      _Pragma("omp critical(dbg_mesg)")  ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
    };
    
    flush_in_parallel_section();
  }

  // 10. Sequential task
  RUN_TEST("0D parallel task")
  {
    using pair_double = std::pair<double,double>;
    pair_double valueA = { 1.0 , 2.0 };
    pair_double valueB = { 3.0 , 4.0 };
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using S = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< F , S >;
    dac::stencil_t<central> stencil_0d{};

    auto acc_a = make_access_controler( valueA , stencil_0d );
    auto acc_b = make_access_controler( valueB , stencil_0d );
    
    static_assert( dac::DataDecompositionTraits<pair_double>::ND == 0 );
    
    default_ptask_queue() <<
    onika_task( acc_a , acc_b )
    {
      std::cout<< format_pair(acc_a) << " , " << format_pair(acc_b) << std::endl;
      //std::cout<< acc_a.first << " , " << acc_b.second << std::endl;
    };
    
    flush_in_parallel_section();
  }

  // 11. detached "parallel for" coupled with simple task
  RUN_TEST("detached parallel for with fulfill task companion")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    dac::stencil_t<central> stencil_2d{};
    std::pair<double,double> my_array[8][8];    
    for(int j=0;j<8;j++) for(int i=0;i<8;i++) my_array[j][i] = { j*10.0+i+11.0 , i+j+2.0 };   
    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2> span { ac.size() };
    
    {
    
      //(
      
      default_ptask_queue() <<
      
      onika_detached_parallel_for( span , ac )
      {
        _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
      }
      ||
      onika_fulfill_task( )
      {
        _Pragma("omp critical(dbg_mesg)") { onika_ctx.fulfill_span().to_stream( std::cout ); std::cout<<std::endl<<std::flush; }
        
        size_t n = onika_ctx.fulfill_span().nb_cells();
        for(size_t i=0;i<n;i++)
        {
          /* if( i == 50 ) { std::this_thread::sleep_for( std::chrono::milliseconds(500) ); } */
          auto c = onika_ctx.fulfill_span().template index_to_coord<2>(i);
          onika_ctx.fulfill( c );
        }
      }

      >> onika::task::flush();
      
      //).print(std::cout);
    }
    
    flush_in_parallel_section();
  }

  // 12. parallel for coupled with reduction
  RUN_TEST("parallel for with reduction data write accessor")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    dac::stencil_t<central> stencil_2d{};
    
    std::pair<double,double> my_array[4][4];
    auto ac = dac::make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2> span { ac.size() };
    
    double sum = 0.0;
    auto sum_ac = dac::make_reduction_access_controler( sum , dac::reduction_add );
    
    default_ptask_queue() <<
    onika_parallel_for( span , ac , sum_ac )
    {
      _Pragma("omp critical(dbg_mesg)") std::cout << onika::format_array(item_coord) << std::endl;
      auto && [a1,a2] = ac;
      a2 += a1;
      sum_ac += 1.0;
    };
    
    flush_in_parallel_section();
    
    std::cout<<"sum = " << sum << std::endl;
  }

  // 13. read-only test
  RUN_TEST("read only accessor")
  {
    std::vector<double> my_array(4*4*4,0.0);
    auto array_view = dac::make_array_3d_view( my_array.data() , {4,4,4} );
    auto array_element = dac::make_access_controler( array_view , dac::make_default_ro_stencil(array_view) );
    dac::box_span_t<3> span { array_element.size() };
    
    default_ptask_queue() <<
    onika_parallel_for( span , array_element )
    {
      _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< onika::format_array(item_coord) << " = " << array_element << std::endl;
      static_assert( dac::is_const_lvalue_ref_v<decltype(array_element)> , "should be a const reference" );
      //array_element = 3.0; // if uncommented, must fail to compile
    };
    
    flush_in_parallel_section();
  }

  // 14. multi value 3D arrays test
  RUN_TEST("multi value 3D array accessor")
  {
    std::vector<ssize_t> my_array(4*4*4 * 3, 0.0);
    auto array_view = dac::make_nvalues_array_3d_view( my_array.data() , 3 , {4,4,4} );    
    auto element = dac::make_access_controler( array_view , dac::make_default_ro_stencil(array_view) );
    for(size_t a=0;a<element.count();a++)
    {
      my_array[a*3+0] = a % 4;
      my_array[a*3+1] = (a/4) % 4;
      my_array[a*3+2] = a/(4*4);
    }

    dac::box_span_t<3> span { element.size() };
    
    default_ptask_queue() <<
    onika_parallel_for( span , element )
    {
      _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< onika::format_array(item_coord) << " = " << onika::format_array(element.data(),element.size()) << std::endl;
      static_assert( std::is_same_v<decltype(element),onika::dac::Array1DView<const ssize_t>> , "should be a Array1DView<const long>" );
      //array_element = 3.0; // if uncommented, must fail to compile
    };
    
    flush_in_parallel_section();
  }

  // 15. parallel task sequenced-before a single task
  RUN_TEST("parallel for sequenced-after another one")
  {
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position
    using central        = dac::stencil_element_t< R , F >;
    dac::stencil_t<central> stencil_2d{};
    std::pair<double,double> my_array[8][8];    
    for(int j=0;j<8;j++) for(int i=0;i<8;i++) my_array[j][i] = { j*10.0+i+11.0 , i+j+2.0 };   
    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2> span { ac.size() };

    default_ptask_queue() << // attach to queue operator
      onika_parallel_for( span , ac )
      {
        _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
      }
      >> // sequence-after operator
      onika_task()
      {
        _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "single task executing after" << std::endl;
      }
      >> onika::task::flush()
      ;
      
    flush_in_parallel_section();
  }


  // 16. 2D DAG based execution with conditional tasking
  RUN_TEST("2D Dag based execution")
  {
    std::cout<< "--------- test "<<n<<" -------" << std::endl;
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position

    using central        = dac::stencil_element_t< R , F >;
    using left           = dac::stencil_element_t< R , F , -1 ,  0 >;
    using top_left       = dac::stencil_element_t< R , F , -1 ,  1 >;
    using bottom_left    = dac::stencil_element_t< R , F , -1 , -1 >;
    using right          = dac::stencil_element_t< R , F ,  1 ,  0 >;
    using top_right      = dac::stencil_element_t< R , F ,  1 ,  1 >;
    using bottom_right   = dac::stencil_element_t< R , F ,  1 , -1 >;
    using bottom         = dac::stencil_element_t< R , F ,  0 , -1 >;
    using top            = dac::stencil_element_t< R , F ,  0 ,  1 >;
    dac::stencil_t<central,left,right,bottom,top,top_left,bottom_left,top_right,bottom_right> stencil_2d{};
    std::pair<double,double> my_array[8][8];
    for(int j=0;j<8;j++) for(int i=0;i<8;i++) my_array[j][i] = { i+1 , j+1 };

    auto ac = make_access_controler( my_array , stencil_2d );
    dac::box_span_t<2> span { ac.size() };

    default_ptask_queue() << // attach to queue operator

      onika_parallel_for( span , ac )
      {
        _Pragma("omp critical(dbg_mesg)") ONIKA_STDOUT_OSTREAM << "item at "<< format_array(item_coord) << " = " << format_pair(ac) << std::endl;
      }
      / [](oarray_t<size_t,2> coord) -> uint64_t { return (coord[0]+coord[1])%2; }
      
      >> onika::task::flush() ;
      
    flush_in_parallel_section();
  }


  onika::task::ParallelTaskExecutor::clear_ptexecutor_cache();
}

