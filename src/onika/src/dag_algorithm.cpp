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

    // ****************** coordinate to node index **********************
    template<size_t Nd>
    std::unordered_map<oarray_t<size_t,Nd>,size_t> dag_item_index_map( const WorkShareDAG<Nd>& dag )
    {
      const size_t n_cells = dag.number_of_items();
      std::unordered_map< oarray_t<size_t,Nd> , size_t > item_index;
      item_index.reserve( dag.number_of_items() );
      for(size_t i=0;i<n_cells;i++)
      {
        item_index[ dag.item_coord(i) ] = i;
      }
      return item_index;
    }

    template std::unordered_map<oarray_t<size_t,1>,size_t> dag_item_index_map( const WorkShareDAG<1>& dag );
    template std::unordered_map<oarray_t<size_t,2>,size_t> dag_item_index_map( const WorkShareDAG<2>& dag );
    template std::unordered_map<oarray_t<size_t,3>,size_t> dag_item_index_map( const WorkShareDAG<3>& dag );
    


    // ****************** node dependency depth **********************
    template<size_t Nd>
    std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG<Nd>& dag )
    {
      auto item_index = dag_item_index_map(dag);      
      const size_t n_cells = dag.number_of_items();
      std::vector<uint16_t> node_dist( n_cells, 0 );
      
      bool conv=false;
      do
      {
        conv=true;
        for(size_t i=0;i<n_cells;i++)
        {
          int md = -1;
          for(const auto& d:dag.item_deps(i))
          {
            size_t di = item_index[d];
            md = std::max( md , int(node_dist[di]) );
          }
          ++md;
          assert( md >= node_dist[i] );
          assert( md < std::numeric_limits<uint16_t>::max() );
          if(md>node_dist[i]) { conv=false; node_dist[i]=md; }
        }
      } while(!conv);

      return node_dist;      
    }

    template std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG<1>& dag );
    template std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG<2>& dag );
    template std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG<3>& dag );


    template<size_t Nd>
    std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG2<Nd>& dag )
    {
      const size_t n_cells = dag.number_of_items();
      std::vector<uint16_t> node_dist( n_cells, 0 );
      bool converged = false;
      do
      {
        converged = true;
        for(size_t i=0;i<n_cells;i++)
        {
          auto d = node_dist[i];
          size_t ndeps = dag.item_dep_count(i);
          for(size_t j=0;j<ndeps;j++)
          {
            d = std::max( size_t(d) , size_t( node_dist[dag.item_dep_idx(i,j)]+1 ) );
          }
          if( d > node_dist[i] ) { node_dist[i]=d; converged=false; }
        }
      } while( ! converged );
      return node_dist;      
    }

    template std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG2<1>& dag );
    template std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG2<2>& dag );
    template std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG2<3>& dag );


    // ****************** simple 3d neigborhood DAG construction **********************
    WorkShareDAG<3> make_3d_neighborhood_exclusion_dag( const oarray_t<size_t,3>& dims )
    {
      dac::AbstractStencil nbh_3d;
      nbh_3d.m_ndims = 3;
      nbh_3d.m_nbits = 1;
      nbh_3d.m_low[0] = -1;
      nbh_3d.m_low[1] = -1;
      nbh_3d.m_low[2] = -1;
      nbh_3d.m_size[0] = 3;
      nbh_3d.m_size[1] = 3;
      nbh_3d.m_size[2] = 3;
      nbh_3d.clear_bits();
      assert( nbh_3d.nb_cells() == 27 );
      for(size_t i=0;i<27;i++)
      {
        nbh_3d.add_ro_mask( i , 1 );
        nbh_3d.add_rw_mask( i , 1 );
      }
      dac::box_span_t<3> sp = { {0,0,0} , dims };
      dac::abstract_box_span_t span(sp);
      return make_stencil_dag<3>( span, nbh_3d );
    }


    // ****************** dag spatial translation **********************
    template<size_t Nd>
    void shift_dag_coords( WorkShareDAG<Nd>& dag , const oarray_t<size_t,Nd>& v)
    {
      for(auto & c : dag.m_deps)
      {
        c = array_add( v , c );
      }
    }

    template void shift_dag_coords( WorkShareDAG<1>& dag , const oarray_t<size_t,1>& v );
    template void shift_dag_coords( WorkShareDAG<2>& dag , const oarray_t<size_t,2>& v );
    template void shift_dag_coords( WorkShareDAG<3>& dag , const oarray_t<size_t,3>& v );
  
    // ********************* co-scheduled inter parallel tasks dependencies **************************
    template<size_t Nd>
    WorkShareDAG<Nd> make_co_stencil_dag(
      const dac::abstract_box_span_t & co_span,
      const WorkShareDAG<Nd>& dag,
      const std::unordered_set< oarray_t<int,Nd> >& co_dep_graph )
    {
      size_t n_cells = dag.number_of_items();

      WorkShareDAG<Nd> result;
      result.m_rem_first_dep = false;
      result.m_start.clear();
      result.m_start.reserve( n_cells + 1 );
      result.m_deps.reserve( n_cells );

      auto coord_to_index_map = dag_item_index_map(dag);

      std::unordered_set< oarray_t<size_t,Nd> > co_deps;
      std::unordered_set< oarray_t<size_t,Nd> > co_deps_to_remove;
      std::unordered_set< oarray_t<size_t,Nd> > wave;
      std::unordered_set< oarray_t<size_t,Nd> > next_wave;

      // make stencil to co_stencil local dependency map
      for(size_t i=0;i<n_cells;i++)
      {
        result.m_start.push_back( result.m_deps.size() );
        auto c = dag.item_coord(i);
        co_deps.clear();
        for(const auto& d : co_dep_graph)
        {
          auto cd = array_add(c,d);
          if( co_span.inside(cd) )
          {
            co_deps.insert(cd);
          }
        }

        co_deps_to_remove.clear();
        next_wave.clear();
        next_wave.insert( dag.item_deps(i).begin() , dag.item_deps(i).end() );
        do
        {
          wave = next_wave;
          next_wave.clear();
          for( const auto& d : wave )
          {
            assert( coord_to_index_map.find(d) != coord_to_index_map.end() );
            size_t di = coord_to_index_map[d];
            assert( di < i );
            for(const auto & cd : result.item_deps(di) ) co_deps_to_remove.insert( cd );
            for(const auto & cd : dag.item_deps(di) ) next_wave.insert( cd );
          }
        }
        while( ! next_wave.empty() );
        
        for(const auto& cdr : co_deps_to_remove) co_deps.erase( cdr );
        
        for(const auto& cd : co_deps)
        {
          result.m_deps.push_back( cd );
        }
      }

      result.m_start.push_back( result.m_deps.size() );      
     
      return result; 
    }
    
    template WorkShareDAG<1> make_co_stencil_dag(const dac::abstract_box_span_t & co_span,const WorkShareDAG<1>& dag,const std::unordered_set< oarray_t<int,1> >& co_dep_graph );
    template WorkShareDAG<2> make_co_stencil_dag(const dac::abstract_box_span_t & co_span,const WorkShareDAG<2>& dag,const std::unordered_set< oarray_t<int,2> >& co_dep_graph );
    template WorkShareDAG<3> make_co_stencil_dag(const dac::abstract_box_span_t & co_span,const WorkShareDAG<3>& dag,const std::unordered_set< oarray_t<int,3> >& co_dep_graph );
  }
}

