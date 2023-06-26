#include <onika/dag/dag_algorithm.h>
#include <onika/dag/dag_stream.h>
#include <onika/dag/dag_filter.h>
#include <onika/dag/dag2_stream.h>
#include <onika/dac/dac.h>
#include <iostream>
#include <fstream>
#include <onika/oarray_stream.h>
#include <onika/grid_grain.h>
#include <onika/zcurve.h>
#include <onika/memory/allocator.h>

template<class T>
struct vector_2d
{
  size_t width = 0;
  size_t height = 0;
  std::vector<T> values;
  inline void resize( const onika::oarray_t<size_t,2>& c ) { width=c[0]; height=c[1]; values.resize(width*height,0.0); }
  inline onika::oarray_t<size_t,2> size() const { return { width , height }; }
  inline auto& operator [] ( const onika::oarray_t<size_t,2>& c ) { return values[c[1]*width+c[0]]; }
  inline const auto& operator [] ( const onika::oarray_t<size_t,2>& c ) const { return values[c[1]*width+c[0]]; }
};

bool g_wave_dag = true;
bool g_grid_dag = true;
bool g_draw_legend = true;
bool g_masked_dag = true;
bool g_gwmotion = false;
bool g_wavegroup = true;
bool g_usepatch = true;
bool g_transreduc = true;

#define __DAC_AUTO_RET_EXPR( ... ) -> decltype( __VA_ARGS__ ) { return __VA_ARGS__ ; }

namespace onika
{
  namespace dac
  {
    template<class T>
    struct DataDecompositionTraits< vector_2d<T> >
    {
      using slices_t = typename DataSlicing<T>::slices_t;
      static constexpr size_t ND = 2;
      using item_coord_t = item_nd_coord_t<ND>;
      static inline constexpr item_coord_t zero_coord = ItemCoordTypeHelper<ND>::zero;
      using value_t = vector_2d<T>;
      using reference_t = value_t &;
      using pointer_t = value_t *;
      using item_t = T;
      ONIKA_HOST_DEVICE_FUNC static inline pointer_t pointer(reference_t v) { return &v; }
      ONIKA_HOST_DEVICE_FUNC static inline void* pointer_at(value_t& v , const item_coord_t& c) { return (void*)( & v[c] ); }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr item_coord_t size(reference_t v) { return v.size(); }
      ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t count(reference_t v) { return v.width * v.height; }
      ONIKA_HOST_DEVICE_FUNC static inline item_t& at(value_t& v , const item_coord_t& c) { return v[c]; }
    };
  }
}

template<size_t Nd , class GetCoordF >
void print_dag( const onika::dag::WorkShareDAG<Nd>& dag , const GetCoordF& get_coord_func)
{
  size_t n = dag.number_of_items();
  for(size_t i=0;i<n;i++)
  {
    std::cout<<i<<" @ "<< onika::format_array( get_coord_func(i) ) << " :";
    for(const auto& d : dag.item_deps(i))
    {
      std::cout <<" "<< onika::format_array(d);
    }
    std::cout<<std::endl;
  }
}

template<size_t Nd , class GetCoordF >
void print_dag( const onika::dag::WorkShareDAG2<Nd>& dag , const GetCoordF& get_coord_func)
{
  size_t n = dag.number_of_items();
  for(size_t i=0;i<n;i++)
  {
    std::cout<<i<<" @ "<< onika::format_array( get_coord_func(i) ) << " :";
    for(const auto& d : dag.item_deps(i))
    {
      std::cout <<" "<< onika::format_array( get_coord_func(d) );
    }
    std::cout<<std::endl;
  }
  std::cout << "total tasks = "<<n<<", total edges = "<<dag.m_deps.size()<<std::endl;
}


template<size_t Nd>
void print_co_dag( const onika::dag::WorkShareDAG<Nd>& dag , const onika::dag::WorkShareDAG<Nd>& co_dag )
{
  size_t n = dag.number_of_items();
  assert( n == co_dag.number_of_items() );
  for(size_t i=0;i<n;i++)
  {
    std::cout<<i<<" @ "<< onika::format_array( dag.item_coord(i) ) << " :";
    for(const auto& d : dag.item_deps(i))
    {
      std::cout <<" "<< onika::format_array(d);
    }
    std::cout<<" |";
    for(const auto& d : co_dag.item_deps(i))
    {
      std::cout <<" "<< onika::format_array(d);
    }
    std::cout<<std::endl;
  }
}


template<class DataT, class StencilA, class SpanA, class StencilB, class SpanB>
static inline void test_co_dag( const std::string& name, DataT& data , StencilA sta, SpanA spa, StencilB stb, SpanB spb )
{
  using namespace onika;

  auto ac_a = make_access_controler( data , sta );
  auto ac_b = make_access_controler( data , stb );
  static_assert( ac_a.ND == ac_b.ND , "accessors dimensionality mismatch" );
  constexpr size_t Nd = ac_a.ND;

  dac::AbstractStencil stencil_a( make_flat_tuple(ac_a) );
  dac::AbstractStencil stencil_b( make_flat_tuple(ac_b) );
  auto stdeps = onika::dac::stencil_co_dep_graph<Nd>( stencil_a , stencil_b );
  
  oarray_t<int,Nd> stdeplo={};
  oarray_t<int,Nd> stdephi={};
  if( ! stdeps.empty() )
  {
    stdeplo = stdephi = *(stdeps.begin());
    for(const auto& x:stdeps) for(size_t k=0;k<Nd;k++) 
    {
      stdeplo[k] = std::min( stdeplo[k] , x[k] );
      stdephi[k] = std::max( stdephi[k] , x[k] );
    }
  }
  for(size_t k=0;k<Nd;k++) stdephi[k] = stdephi[k] - stdeplo[k] + 1;

  std::cout<<"stencil co-dep box size = "<< onika::format_array(stdephi) <<std::endl;
  std::cout<<"stencil co-dep count = "<<stdeps.size()<<std::endl;
  std::vector< oarray_t<int,Nd> > sorted_stencil_deps( stdeps.begin() , stdeps.end() );
  std::sort( sorted_stencil_deps.begin() , sorted_stencil_deps.end() );
  std::cout<<"stencil co-dep positions :"<<std::endl;
  for(const auto& d:sorted_stencil_deps) std::cout<< "  " << onika::format_array(d) << std::endl;
  
  dac::abstract_box_span_t span_a ( spa );
  dac::abstract_box_span_t span_b ( spb );
  
  auto dag_a = dag::make_stencil_dag<Nd>( span_a , stencil_a );
//  task::shift_dag_coords( dag_a , span_a.lower_bound_nd<Nd>() );
  auto co_dep_pattern = dac::stencil_co_dep_graph<Nd>( stencil_a, stencil_b );
  auto co_dag = dag::make_co_stencil_dag( span_b, dag_a, co_dep_pattern );
  
  std::cout<<"generated co-DAG :"<<std::endl;
  print_co_dag( dag_a , co_dag );
  std::string fname = name+".dot-n2-Kfdp";
  std::cout<<"write co-dag graph to "<< fname << std::endl;
  std::ofstream fout(fname);
  std::function< oarray_t<size_t,Nd>(size_t) > coord_func = [&dag_a](size_t i) -> oarray_t<size_t,Nd> { return dag_a.item_coord(i); };
  std::function< bool(const oarray_t<size_t,Nd>& c) > mask_func = [&span_b](const oarray_t<size_t,Nd>& c) -> bool { return span_b.inside(c); };
  onika::dag::dag_to_dot( co_dag , span_a.box_size_nd<Nd>() , fout , 0.0 , 1, true , coord_func , mask_func );
}

// generate png with : dot -Kfdp -n -Tpng -o out.png out.dot
auto g_null_cost_func = [](size_t) -> uint64_t{return 1;};
auto g_field_name = [](uint64_t x) -> std::string
       {
         static const char* st[]={"","F","R","FR","XXX"};
         if(x>4) x=4;
         return st[x];
       };

template<class DataT, class StencilT , class GrainSizeConstant = std::integral_constant<size_t,1> , class CostFunc=decltype(g_null_cost_func) , class NameFunc=decltype(g_field_name) >
static inline void test_dag( const std::string& name, DataT& data , StencilT st , GrainSizeConstant grainsize = GrainSizeConstant{}
                           , const CostFunc& cost_func = g_null_cost_func , const NameFunc& name_func = g_field_name )
{
  using namespace onika;

  std::cout << "--------- "<< name <<" ------------------" << std::endl;

  auto ac = make_access_controler( data , st );
  dac::AbstractStencil stencil( make_flat_tuple(ac) );
  auto stdims = stencil.box_size<ac.ND>(); // dac::stencil_box_size( stencil , std::integral_constant<size_t,ac.ND>{} );
  std::cout<<"stencil box size = "<< onika::format_array(stdims) << std::endl;

  using DotConf = onika::dag::Dag2DotConfig<ac.ND>;

  // stencil output
  {
    std::string fname = name+"-stencil.dot-n2-Kfdp";
    std::cout<<"write stencil to "<< fname << std::endl;
    std::ofstream fout(fname);
    onika::dac::stencil_dot( fout , stencil , name_func );
  }

  // stencil dependences analysis
  {
    auto stdeps = onika::dac::stencil_dep_graph<ac.ND>( stencil , grainsize );
    oarray_t<int,ac.ND> stdeplo={};
    oarray_t<int,ac.ND> stdephi={};
    if( ! stdeps.empty() )
    {
      stdeplo = stdephi = *(stdeps.begin());
      for(const auto& x:stdeps) for(size_t k=0;k<ac.ND;k++) 
      {
        stdeplo[k] = std::min( stdeplo[k] , x[k] );
        stdephi[k] = std::max( stdephi[k] , x[k] );
      }
    }
    for(size_t k=0;k<ac.ND;k++) stdephi[k] = stdephi[k] - stdeplo[k] + 1;

    std::cout<<"stencil dependency box size = "<< onika::format_array(stdephi) <<std::endl;
    std::cout<<"stencil dependences count = "<<stdeps.size()<<std::endl;
    std::vector< oarray_t<int,ac.ND> > sorted_stencil_deps( stdeps.begin() , stdeps.end() );
    std::sort( sorted_stencil_deps.begin() , sorted_stencil_deps.end() );
    std::cout<<"stencil dependency positions :"<<std::endl;
    for(const auto& d:sorted_stencil_deps) std::cout<< "  " << onika::format_array(d) << std::endl;
    
    std::string fname = name+"-dep-stencil.dot-n2-Kfdp";
    std::cout<<"write stencil graph to "<< fname << std::endl;
    std::ofstream fout(fname);
    onika::dac::stencil_dep_dot( fout , stdeps );
  }

  // DAG generation
  {
    auto zero = ac.zero_coord;
    dac::box_span_t<ac.ND , grainsize > sp = { zero , ac.size() };
    dac::abstract_box_span_t span(sp);
    
    auto dag = dag::make_stencil_dag2<ac.ND>( span , stencil, nullptr, nullptr, g_usepatch, g_transreduc);
/*    if( dag.number_of_items() < 512 )
    {
      std::cout<<"checking against legacy implementation"<<std::endl;
      assert( dag == dag::make_stencil_dag_legacy<ac.ND>( span , stencil ) );
      std::cout<<"checking against V2 implementation"<<std::endl;
      assert_equal_dag( dag , dag::make_stencil_dag2<ac.ND>( span , stencil ) );
    }
*/
    
    std::cout<<"generated DAG :"<<std::endl;
    print_dag( dag , [&dag](size_t i){return dag.item_coord(i);} );

    auto coarse_domain = span.template coarse_domain_nd<ac.ND> ();

    if(g_wave_dag)
    if( grainsize==1 )
    {
      std::string fname = name+".dot-Kdot";
      std::cout<<"write "<< fname << " ..." << std::endl;
      std::ofstream fout1(fname);
      DotConf c; c.add_legend=g_draw_legend; c.add_bounds_corner=true;
      onika::dag::dag_to_dot( dag , coarse_domain , fout1 , std::move(c) );
    }

    if(g_wave_dag)
    {
      std::string fname = name+".dot-n2-Kfdp";
      std::cout<<"write "<< fname << " ..." << std::endl;
      std::ofstream fout2(fname);
      DotConf c;  c.bbenlarge={0.1,0.1}; c.urenlarge={1.25,1.25};  c.grainsize=grainsize; c.add_legend=g_draw_legend; c.add_bounds_corner=(grainsize>1) ;
      onika::dag::dag_to_dot( dag , coarse_domain , fout2 , std::move(c) );
    }

    if(g_wavegroup)
    {
      std::string fname = name+"-wg.dot-n2-Kfdp";
      std::cout<<"write "<< fname << " ..." << std::endl;
      std::ofstream fout2(fname);
      DotConf c; c.bbenlarge={0.05,0.05}; c.grainsize=grainsize; c.fdp=true; c.add_legend=g_draw_legend; c.add_bounds_corner=true; c.wave_group=true;
      onika::dag::dag_to_dot( dag , coarse_domain , fout2 , std::move(c) );
    }

    if(g_grid_dag)
    {    
      std::string fname = name+"-fdp.dot-n2-Kfdp";
      std::cout<<"write "<< fname << " ..." << std::endl;
      std::ofstream fout2(fname);
      DotConf c; c.bbenlarge={0.1,0.1}; c.urenlarge={1.25,1.25}; c.gw=1.; c.grainsize=grainsize; c.fdp=true; c.add_legend=g_draw_legend; c.add_bounds_corner=(grainsize>1);
      onika::dag::dag_to_dot( dag , coarse_domain , fout2 ,  std::move(c) );
    }

    if(g_gwmotion)
    {
      for(int i=0;i<100;i++)
      {
        std::string fname = name+ "-mov" + std::to_string((i/10)%10) + std::to_string(i%10) + ".dot-n2-Kfdp";
        std::cout<<"write "<< fname << " ..." << std::endl;
        std::ofstream fout2(fname);
        DotConf c; c.gw=1.-i/99.; c.grainsize=grainsize; c.fdp=true; c.add_legend=g_draw_legend; c.add_bounds_corner=true; c.movie_bounds=true;
        onika::dag::dag_to_dot( dag , coarse_domain , fout2 ,  std::move(c) );
      }
    }

    
    if(g_masked_dag)
    {
      std::string fname = name+"-mask-fdp.dot-n2-Kfdp";
      std::cout<<"write "<< fname << " ..." << std::endl;
      std::ofstream fout2(fname);
      auto nodemaskfunc = [&cost_func](size_t i)->bool{return cost_func(i)!=0;} ;
      DotConf c; c.mask_func = nodemaskfunc; c.gw=1.; c.grainsize=grainsize; c.fdp=true; c.add_legend=g_draw_legend; c.add_bounds_corner=true; c.movie_bounds=false;
      onika::dag::dag_to_dot( dag , span.template coarse_domain_nd<ac.ND> () , fout2 ,  std::move(c) );

      if( grainsize==1 )
      {
        fout2.close();
        fname = name+"-mask.dot-Kdot";
        std::cout<<"write "<< fname << " ..." << std::endl;
        fout2.open(fname);
        DotConf c; c.mask_func = nodemaskfunc; c.add_legend=g_draw_legend; c.add_bounds_corner=true;
        onika::dag::dag_to_dot( dag , coarse_domain , fout2 , std::move(c) );
      }

      fout2.close();
      fname = name+"-reduced-fdp.dot-n2-Kfdp";
      std::cout<<"write "<< fname << " ..." << std::endl;
      fout2.open(fname);
      auto reduced_dag = onika::dag::filter_dag( dag , nodemaskfunc );
      {
        DotConf c; c.gw=1.; c.grainsize=grainsize; c.fdp=true; c.add_legend=g_draw_legend; c.add_bounds_corner=(grainsize>1);
        onika::dag::dag_to_dot( reduced_dag , coarse_domain , fout2 , std::move(c) );
      }
      
      if( grainsize==1 )
      {
        fout2.close();
        fname = name+"-reduced.dot-Kdot";
        std::cout<<"write "<< fname << " ..." << std::endl;
        fout2.open(fname);
        DotConf c; c.add_legend=g_draw_legend; c.add_bounds_corner=true;
        onika::dag::dag_to_dot( reduced_dag , coarse_domain , fout2 , std::move(c) );
      }

      fout2.close();
      fname = name+"-reduced.dot-n2-Kfdp";
      std::cout<<"write "<< fname << " ..." << std::endl;
      fout2.open(fname);
      {
        DotConf c; c.grainsize=grainsize; c.add_legend=g_draw_legend; c.add_bounds_corner=(grainsize>1);
        onika::dag::dag_to_dot( reduced_dag , coarse_domain , fout2 , std::move(c) );
      }
    }
    
  }

  std::cout << "-----------------------------------------" << std::endl << std::endl;
}

//#define RUN_TEST if(test==(++n)||test==-1) 
#define RUN_TEST(label) if( ( test==(++n) || test<0 ) && std::cout<<n<<" : "<<label<<"\n" ) if(test!=-2)

int main(int argc,char*argv[])
{
  static constexpr std::integral_constant<size_t,2> const_2{};
  static constexpr std::integral_constant<size_t,1> const_1{};

  using namespace onika;

  memory::GenericHostAllocator::set_cuda_enabled(false);

  int test = -2;
  int n = 0;
  for(int i=1;i<argc;i++)
  {
    std::string arg = argv[i];

         if( arg == "+wave_dag" ) g_wave_dag=true;
    else if( arg == "-wave_dag" ) g_wave_dag=false;

    else if( arg == "+grid_dag" ) g_grid_dag=true;
    else if( arg == "-grid_dag" ) g_grid_dag=false;

    else if( arg == "+draw_legend" ) g_draw_legend=true;
    else if( arg == "-draw_legend" ) g_draw_legend=false;

    else if( arg == "+masked_dag" ) g_masked_dag=true;
    else if( arg == "-masked_dag" ) g_masked_dag=false;

    else if( arg == "+gwmotion" ) g_gwmotion=true;
    else if( arg == "-gwmotion" ) g_gwmotion=false;

    else if( arg == "+wavegroup" ) g_wavegroup=true;
    else if( arg == "-wavegroup" ) g_wavegroup=false;

    else if( arg == "+usepatch" ) g_usepatch=true;
    else if( arg == "-usepatch" ) g_usepatch=false;

    else if( arg == "+transreduc" ) g_transreduc=true;
    else if( arg == "-transreduc" ) g_transreduc=false;

    else test = std::atoi(argv[i]);
  }

  // std::cout << "Selected test = "<<test<<"\n";

  // 1. 1D left-right
  RUN_TEST("simple1d")
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
    using central = dac::stencil_element_t< dac::DataSlices<> , dac::DataSlices<dac::whole_t> >;
    using left    = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> , -1 >;
    using right   = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> ,  1 >;
    //dac::stencil_t<central,left,right> stencil_1d{};
    dac::Stencil< central , dac::stencil_elements_t<left,right> , 1> stencil_1d{};
    std::vector<double> my_array(10,0.0);
    test_dag( "simple1d" , my_array , stencil_1d );
  }

  // 2. 2D + shape stencil
  RUN_TEST("simple2d")
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
    using central = dac::stencil_element_t< dac::DataSlices<> , dac::DataSlices<dac::whole_t> >;
    using left    = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> , -1 ,  0 >;
    using right   = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> ,  1 ,  0 >;
    using bottom  = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> ,  0 , -1 >;
    using top     = dac::stencil_element_t< dac::DataSlices<dac::whole_t> , dac::DataSlices<> ,  0 ,  1 >;
    dac::stencil_t<central,left,right,bottom,top> stencil_2d{};
    vector_2d<double> my_array;
    my_array.resize( {5,5} );
    test_dag( "simple2d" , my_array , stencil_2d );
  }

  // 3. 2D heat equation stencil with static 2D array
  RUN_TEST("northeast2d")
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
    using Cell = std::pair<double,double>;
    using Position = dac::pair_first_t;
    using Force = dac::pair_second_t;
    using central = dac::stencil_element_t< dac::DataSlices<Position> , dac::DataSlices<Force> >;
    using right   = dac::stencil_element_t< dac::DataSlices<Position> , dac::DataSlices<Force> ,  1 ,  0 >;
    using top     = dac::stencil_element_t< dac::DataSlices<Position> , dac::DataSlices<Force> ,  0 ,  1 >;
    dac::stencil_t<central,right,top> stencil_2d{};   
    Cell my_array[6][6];
    test_dag( "northeast2d" , my_array , stencil_2d , const_1 , g_null_cost_func , [](uint64_t x)->std::string
       {
         static const char* st[]={" ","R","F","RF","XXX"};
         if(x>4) x=4;
         return st[x];
       } );
  }

  // 4-7 2D MEAM-like stencil
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
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
    std::pair<double,double> my_array8[8][8];
    std::pair<double,double> my_array6[6][6];
    std::pair<double,double> my_array10[10][10];
    RUN_TEST("meam2d") { test_dag( "meam2d" , my_array8 , stencil_2d ); }
    RUN_TEST("meam2ds2") { test_dag( "meam2ds2" , my_array6 , dac::downscale_stencil(stencil_2d,const_2) ); }
    RUN_TEST("meam2dg2") { test_dag( "meam2dg2" , my_array10 , stencil_2d , const_2 ); }
    RUN_TEST("meam2ds2msk") { test_dag( "meam2ds2msk" , my_array8 , dac::downscale_stencil(stencil_2d,const_2) , const_1 , [](size_t i)->uint64_t{return (i%5)!=0;} ); }
  }

  // 8-10 3D MEAM-like stencil
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position

    dac::stencil_t<
        dac::stencil_element_t< R , F >
      , dac::stencil_element_t< R , F , -1 ,  0 , 0 >
      , dac::stencil_element_t< R , F , -1 ,  1 , 0 >
      , dac::stencil_element_t< R , F , -1 , -1 , 0 >
      , dac::stencil_element_t< R , F ,  1 ,  0 , 0 >
      , dac::stencil_element_t< R , F ,  1 ,  1 , 0 >
      , dac::stencil_element_t< R , F ,  1 , -1 , 0 >
      , dac::stencil_element_t< R , F ,  0 , -1 , 0 >
      , dac::stencil_element_t< R , F ,  0 ,  1 , 0 >

      , dac::stencil_element_t< R , F ,  0 ,  0 , -1 >
      , dac::stencil_element_t< R , F , -1 ,  0 , -1 >
      , dac::stencil_element_t< R , F , -1 ,  1 , -1 >
      , dac::stencil_element_t< R , F , -1 , -1 , -1 >
      , dac::stencil_element_t< R , F ,  1 ,  0 , -1 >
      , dac::stencil_element_t< R , F ,  1 ,  1 , -1 >
      , dac::stencil_element_t< R , F ,  1 , -1 , -1 >
      , dac::stencil_element_t< R , F ,  0 , -1 , -1 >
      , dac::stencil_element_t< R , F ,  0 ,  1 , -1 >

      , dac::stencil_element_t< R , F ,  0 ,  0 , 1 >
      , dac::stencil_element_t< R , F , -1 ,  0 , 1 >
      , dac::stencil_element_t< R , F , -1 ,  1 , 1 >
      , dac::stencil_element_t< R , F , -1 , -1 , 1 >
      , dac::stencil_element_t< R , F ,  1 ,  0 , 1 >
      , dac::stencil_element_t< R , F ,  1 ,  1 , 1 >
      , dac::stencil_element_t< R , F ,  1 , -1 , 1 >
      , dac::stencil_element_t< R , F ,  0 , -1 , 1 >
      , dac::stencil_element_t< R , F ,  0 ,  1 , 1 >
      > stencil_3d{};

    std::pair<double,double> my_array[5][5][5];

    RUN_TEST("meam3d") { test_dag( "meam3d" , my_array , stencil_3d ); }
    RUN_TEST("meam3ds2") { test_dag( "meam3ds2" , my_array , dac::downscale_stencil(stencil_3d,const_2) ); }
    RUN_TEST("meam3dg2") { test_dag( "meam3dg2" , my_array , stencil_3d , const_2 ); }
  }

 
  RUN_TEST("ghost_meam") // 11. simulates MEAM like potential after ghost update receives
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
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
    
    dac::stencil_t<central,left,right,bottom,top,top_left,bottom_left,top_right,bottom_right> stencil_a{};
  
    using central_write_r = dac::stencil_element_t< dac::DataSlices<> , R >;
    dac::stencil_t< central_write_r  > stencil_b{};
    std::pair<double,double> my_array[8][8];
    dac::box_span_t<2> span_a { {0,0} , {8,8} };
    dac::box_span_t<2 , 1> span_b { {1,1} , {6,6} };
    test_co_dag( "ghost_meam", my_array, stencil_a, span_a, stencil_b, span_b );
  }

  RUN_TEST("ghost_meam_l") // 12. simulates MEAM like potential after ghost update receives
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
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
    
    dac::stencil_t<central,left,right,bottom,top,top_left,bottom_left,top_right,bottom_right> stencil_a{};
  
    using central_write_r = dac::stencil_element_t< dac::DataSlices<> , R >;
    using left_write_r = dac::stencil_element_t< dac::DataSlices<> , R , -1 , 0>;
    dac::stencil_t< central_write_r , left_write_r > stencil_b{};
    std::pair<double,double> my_array[8][8];
    dac::box_span_t<2> span_a { {0,0} , {8,8} };
    dac::box_span_t<2 , 1> span_b { {1,1} , {6,6} };
    test_co_dag( "ghost_meam_l", my_array, stencil_a, span_a, stencil_b, span_b );
  }

  RUN_TEST("ghost_meam3d") // 13. simulates MEAM like potential after ghost update receives
  {
    //std::cout<< "--------- test "<<n<<" -------" << std::endl;
    using F = dac::DataSlices< dac::pair_first_t >; // RW force
    using R = dac::DataSlices< dac::pair_second_t >; // RO position

    dac::stencil_t<
        dac::stencil_element_t< R , F >
      , dac::stencil_element_t< R , F , -1 ,  0 , 0 >
      , dac::stencil_element_t< R , F , -1 ,  1 , 0 >
      , dac::stencil_element_t< R , F , -1 , -1 , 0 >
      , dac::stencil_element_t< R , F ,  1 ,  0 , 0 >
      , dac::stencil_element_t< R , F ,  1 ,  1 , 0 >
      , dac::stencil_element_t< R , F ,  1 , -1 , 0 >
      , dac::stencil_element_t< R , F ,  0 , -1 , 0 >
      , dac::stencil_element_t< R , F ,  0 ,  1 , 0 >

      , dac::stencil_element_t< R , F ,  0 ,  0 , -1 >
      , dac::stencil_element_t< R , F , -1 ,  0 , -1 >
      , dac::stencil_element_t< R , F , -1 ,  1 , -1 >
      , dac::stencil_element_t< R , F , -1 , -1 , -1 >
      , dac::stencil_element_t< R , F ,  1 ,  0 , -1 >
      , dac::stencil_element_t< R , F ,  1 ,  1 , -1 >
      , dac::stencil_element_t< R , F ,  1 , -1 , -1 >
      , dac::stencil_element_t< R , F ,  0 , -1 , -1 >
      , dac::stencil_element_t< R , F ,  0 ,  1 , -1 >

      , dac::stencil_element_t< R , F ,  0 ,  0 , 1 >
      , dac::stencil_element_t< R , F , -1 ,  0 , 1 >
      , dac::stencil_element_t< R , F , -1 ,  1 , 1 >
      , dac::stencil_element_t< R , F , -1 , -1 , 1 >
      , dac::stencil_element_t< R , F ,  1 ,  0 , 1 >
      , dac::stencil_element_t< R , F ,  1 ,  1 , 1 >
      , dac::stencil_element_t< R , F ,  1 , -1 , 1 >
      , dac::stencil_element_t< R , F ,  0 , -1 , 1 >
      , dac::stencil_element_t< R , F ,  0 ,  1 , 1 >
      > stencil_a{};
    
    using central_write_r = dac::stencil_element_t< dac::DataSlices<> , R >;
    //using left_write_r = dac::stencil_element_t< dac::DataSlices<> , R , -1 , 0>;
    dac::stencil_t< central_write_r /*, left_write_r*/ > stencil_b{};
    
    std::pair<double,double> my_array[8][8][8];
    dac::box_span_t<3> span_a { {0,0,0} , {8,8,8} };
    dac::box_span_t<3 , 1> span_b { {1,1,1} , {6,6,6} };

    test_co_dag( "ghost_meam3d", my_array, stencil_a, span_a, stencil_b, span_b );
  }

  RUN_TEST("z-order") // 14. Z-Order test
  {
    std::cout<<"2-D Z-curve, size = 8x8"<<std::endl;
    z_order_apply( onika::GridGrainPo2<2,3>{} , []( const oarray_t<size_t,2>& c ){ std::cout<<format_array(c)<<std::endl; } );
    std::cout<<"3-D Z-curve, size = 4x4x4"<<std::endl;
    z_order_apply( GridGrainPo2<3,2>{} , []( const oarray_t<size_t,3>& c ){ std::cout<<format_array(c)<<std::endl; } );
    std::cout<<"3-D grid scan, size = 1x1x1"<<std::endl;
    grid_grain_apply( GridGrain<3>{1} , []( const oarray_t<size_t,3>& c ){ std::cout<<format_array(c)<<std::endl; } );
    std::cout<<"2-D grid scan, size = 3x3"<<std::endl;
    grid_grain_apply( GridGrain<2>{3} , []( const oarray_t<size_t,2>& c ){ std::cout<<format_array(c)<<std::endl; } );
    std::cout<<"3-D grid scan, size = 3x3x3"<<std::endl;
    grid_grain_apply( GridGrain<3>{3} , []( const oarray_t<size_t,3>& c ){ std::cout<<format_array(c)<<std::endl; } );
  }
 
}

