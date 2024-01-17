#include <onika/dag/dag_algorithm.h>

#include <onika/dac/stencil.h>
#include <onika/dac/box_span.h>

#include <unordered_set>
#include <vector>
#include <onika/oarray.h>
#include <algorithm>
#include <cassert>

// debug prints
#include <iomanip>
#include <onika/oarray_stream.h>

namespace onika
{

  namespace dag
  {

    namespace workshare_dag_details
    {
      struct CellDependenceTemp
      {
        std::unordered_set<size_t> dep_in;
        ssize_t max_dist;
      };
    }

    using namespace workshare_dag_details;

    // ***************************** Access pattern to DAG *********************************
    // span information is NOT missing, because we only treat intra parallel task dependecies
    template<size_t Nd>
    WorkShareDAG<Nd> make_stencil_dag_legacy( const dac::abstract_box_span_t & span , const dac::AbstractStencil & stencil )
    {
      //static constexpr std::integral_constant<size_t,Nd> ndims{};
      assert( Nd == stencil.m_ndims );
      assert( Nd == span.ndims );
      //assert( stencil.sanity_check() );
          
      oarray_t<size_t,Nd> domain; initialize_array( domain , span.coarse_domain );
      const size_t n_cells = domain_size( domain );
      const size_t grainsize = span.grainsize * stencil.m_scaling;

      // convert stencil masks to list of neighbor dependencies
      auto nbh_deps = dac::stencil_dep_graph<Nd>( stencil , grainsize );

      oarray_t<size_t,Nd> stencil_dep_box_size = ZeroArray<size_t,Nd>::zero;
      for(const auto& d : nbh_deps)
      {
        for(size_t k=0;k<Nd;k++)
        {
          stencil_dep_box_size[k] = std::max( stencil_dep_box_size[k] , size_t(std::abs(d[k])) );
        }
      }
      for(size_t k=0;k<Nd;k++) stencil_dep_box_size[k] += 1;
      size_t stencil_dep_box_elements = domain_size( stencil_dep_box_size );
      assert( stencil_dep_box_elements > 0 );

      // if DAG happens to be trivial, even though it is not based on a local stencil, we inform caller that this DAG is trivial
      if( stencil_dep_box_elements == 1 )
      {
        return WorkShareDAG<Nd>{};
      }

      // stencil box filling curve
      // compute a traversal pattern for a rectangular area as large as the stencil bounding box
      // std::cout<<"traversal patch size = "<<format_array(stencil_dep_box_size)<<std::endl;
      oarray_t<size_t,Nd> patch_traversal[stencil_dep_box_elements];
      for(size_t i=0;i<stencil_dep_box_elements;i++)
      {
        auto c = index_to_coord( i, stencil_dep_box_size );
        bool rev = false;
        for(ssize_t j=(Nd-1);j>=0;j--)
        {
          if( rev ) c[j] = stencil_dep_box_size[j] - 1 - c[j];
          if( c[j]%2 != 0 ) rev = ! rev;
        }
        // std::cout<<std::setw(2)<< i<<" :" << format_array(c) <<std::endl;
        patch_traversal[i] = c;
      }

      std::vector<CellDependenceTemp> graph_tmp( n_cells , { {} , -1 } );

      // building waves of tasks with corresponding dependences
      auto cdims = domain;
      for(size_t i=0;i<Nd;i++) cdims[i] = (cdims[i]/stencil_dep_box_size[i])+1;
      size_t cncells = domain_size(cdims);
      
      for(size_t p=0;p<stencil_dep_box_elements;p++)
      {
        for(size_t i=0;i<cncells;i++)
        {
          auto c = to_signed( index_to_coord( i , cdims ) );
          for(size_t k=0;k<Nd;k++) c[k] = c[k] * stencil_dep_box_size[k] + patch_traversal[p][k];
          if( in_range(c,domain) )
          {
            size_t cidx = coord_to_index(c,domain);
            // look for dependencies around
            ssize_t md = 0;
            for( const auto& dpos : nbh_deps )
            {
              auto nbh_c = dpos;
              for(size_t k=0;k<Nd;k++) nbh_c[k] += c[k];
              if( in_range(nbh_c,domain) )
              {
                size_t d = coord_to_index( nbh_c , domain );
                if( graph_tmp[d].max_dist>=0 ) // it means item has been visited already, hence we build a dependence from it
                {
                  graph_tmp[cidx].dep_in.insert( d );
                  md = std::max( md , graph_tmp[d].max_dist + 1 );
                }
              }
            }
            //assert( p==0 || md>0 );
            graph_tmp[cidx].max_dist = md;
          }
        }
      }

      for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
      {
        assert( graph_tmp[cell_idx].max_dist >= 0 );
        graph_tmp[cell_idx].max_dist = -1;
      }

      // ***************** transitive reduction ****************
      // std::cout<<"transitive reduction"<<std::endl<<std::flush;
      auto transitive_closure = [&graph_tmp](size_t i) -> std::unordered_set<size_t>
        {
          std::unordered_set<size_t> closure = graph_tmp[i].dep_in;
          size_t n;
          do 
          {
            std::unordered_set<size_t> next;
            for(auto j:closure) for(auto k:graph_tmp[j].dep_in) { next.insert(k); }
            n = closure.size();
            closure.insert( next.begin() , next.end() );
          } while( closure.size() > n );
          return closure;
        };
      for(size_t i=0;i<n_cells;i++)
      {
  #     ifndef NDEBUG
        auto before_closure = transitive_closure(i);
  #     endif
      
        std::unordered_set<size_t> closure;
        for(auto j:graph_tmp[i].dep_in)
        {
          auto c = transitive_closure(j);
          closure.insert( c.begin() , c.end() );
        }
        std::unordered_set<size_t> reduced_deps;
        for(auto j:graph_tmp[i].dep_in)
        {
          if( closure.find(j) == closure.end() )
          {
            reduced_deps.insert( j );
          }
        }
        graph_tmp[i].dep_in = std::move( reduced_deps );

        assert( before_closure == transitive_closure(i) );
      }
      // *******************************************************


      // ***************** maximum distance to source ****************
      // std::cout<<"maximum distance search ..."<<std::endl<<std::flush;
      bool conv_max_dist = false;
      size_t conv_max_passes = 16;
      ssize_t max_dist = 0;
      do
      {
        if( conv_max_passes <= 0 ) std::abort(); // ensures there's no cycle
        -- conv_max_passes;
        conv_max_dist = true;
        max_dist = -1;
        for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
        {
          ssize_t dist = 0;
          for(auto j : graph_tmp[cell_idx].dep_in)
          {
            assert( graph_tmp[j].dep_in.find(cell_idx) == graph_tmp[j].dep_in.end() );
            dist = std::max( dist , graph_tmp[j].max_dist+1 );
          }
          conv_max_dist = conv_max_dist && ( graph_tmp[cell_idx].max_dist == dist );
          graph_tmp[cell_idx].max_dist = dist;
          max_dist = std::max( max_dist , dist );
        }
      } while( !conv_max_dist );
      // std::cout<<"max_dist="<<max_dist<<std::endl;
      // *******************************************************



      // ********* Check graph consistency *******
  #   ifndef NDEBUG
      auto path_exists = [&transitive_closure](size_t a, size_t b) -> bool
        {
          auto ca = transitive_closure(a);
          auto cb = transitive_closure(b);
          return ca.find(b)!=ca.end() || cb.find(a)!=cb.end();
        };

      for(size_t i=0;i<n_cells;i++)
      {
        auto c = index_to_coord( i, domain );
        for(const auto& nbh : nbh_deps)
        {
          auto d = c;
          for(size_t k=0;k<Nd;k++) d[k] += nbh[k];
          if( in_range(d,domain) )
          {
            size_t j = coord_to_index( d , domain );
            assert( path_exists(i,j) );          
          }
        }
      }
  #   endif    
      // *****************************************

      size_t n_deps = n_cells;
      std::vector<size_t> task_indices( n_cells );
      for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
      {
        n_deps += graph_tmp[cell_idx].dep_in.size();
        task_indices[cell_idx] = cell_idx;
      }
      // std::cout << "total deps = "<< n_deps-n_cells << std::endl;
      
      std::stable_sort( task_indices.begin() , task_indices.end() ,
        [&graph_tmp](size_t a, size_t b)->bool
        {
          return graph_tmp[a].max_dist < graph_tmp[b].max_dist;
        });

      auto sort_descending_maxdist = [&graph_tmp] ( size_t a , size_t b ) -> bool
        {
          assert(a<graph_tmp.size());
          assert(b<graph_tmp.size());
          auto da = graph_tmp[a].max_dist;
          auto db = graph_tmp[b].max_dist;
          return ( da > db ) || ( da == db && a < b );
        };

      // std::cout << "final copy"<< std::endl << std::flush;
      WorkShareDAG<Nd> result;
      result.m_start.clear();
      result.m_start.reserve( n_cells + 1 );
      result.m_deps.clear();
      result.m_deps.reserve( n_deps );
      result.m_start.push_back( result.m_deps.size() ); // push first offset ( 0 )
      
      std::vector<size_t> deps; // temporary
      
      for(auto i : task_indices)
      {
        result.m_deps.push_back( array_add( index_to_coord(i,domain) , span.lower_bound ) );
        
        deps.assign( graph_tmp[i].dep_in.begin() , graph_tmp[i].dep_in.end() );
        std::sort( deps.begin() , deps.end() , sort_descending_maxdist );
        
        for(auto d : deps) result.m_deps.push_back( array_add( index_to_coord(d,domain) , span.lower_bound ) );
        result.m_start.push_back( result.m_deps.size() );
      }

      assert_schedule_ordering( result );

      return result;
    }

//    template WorkShareDAG<0> make_stencil_dag<0>( const dac::abstract_box_span_t & , const dac::AbstractStencil & );
    template WorkShareDAG<1> make_stencil_dag_legacy<1>( const dac::abstract_box_span_t & , const dac::AbstractStencil & );
    template WorkShareDAG<2> make_stencil_dag_legacy<2>( const dac::abstract_box_span_t & , const dac::AbstractStencil & );
    template WorkShareDAG<3> make_stencil_dag_legacy<3>( const dac::abstract_box_span_t & , const dac::AbstractStencil & );
  }
}

