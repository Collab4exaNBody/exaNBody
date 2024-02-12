/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid_dependency_graph.h>

#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <assert.h>

/*
#include <iostream>
using std::endl;
using std::cout;
using std::flush;
*/

namespace exanb
{

  GridDependencyPattern::GridDependencyPattern()
  {
    for(int w=0;w<8;w++)
    {
      int ci=w%2;
      int cj=(w/2)%2;
      int ck=(w/4)%2; 
      for(int i=-1;i<=1;i++)
      for(int j=-1;j<=1;j++)
      for(int k=-1;k<=1;k++)
      {
        pattern[ci][cj][ck].dep[i+1][j+1][k+1] = -1; // undefined
      }
    }

    int pass = 0;
    int nundef = 1;
    while( nundef>0 && pass<8 )
    {
      nundef = 0;
      for(int w=pass;w<8;w++)
      {
        int ci=w%2;
        int cj=(w/2)%2;
        int ck=(w/4)%2;
        
        for(int i=-1;i<=1;i++)
        for(int j=-1;j<=1;j++)
        for(int k=-1;k<=1;k++)
        {
          int nci = (ci+2+i) % 2;
          int ncj = (cj+2+j) % 2;
          int nck = (ck+2+k) % 2;
          int ni = -i;
          int nj = -j;
          int nk = -k;
          int nval = pattern[nci][ncj][nck].dep[ni+1][nj+1][nk+1];
          if( nval != -1 )
          {
            pattern[ci][cj][ck].dep[i+1][j+1][k+1] = ! nval ;
          }
          else if( w==pass )
          {
            pattern[ci][cj][ck].dep[i+1][j+1][k+1] = 0;
          }
          else if( pattern[ci][cj][ck].dep[i+1][j+1][k+1] == -1 ) { ++nundef; }
        }
      }
//      cout << "pass #"<<pass<<" nundef="<<nundef<<endl;
      ++pass;
    }
    
#   ifndef NDEBUG
    for(int w=0;w<8;w++)
    {
      int ci=w%2;
      int cj=(w/2)%2;
      int ck=(w/4)%2;
  //    cout << "Wave #"<<w<<endl;    
      for(int i=-1;i<=1;i++)
      for(int j=-1;j<=1;j++)
      for(int k=-1;k<=1;k++)
      {
        if( i!=0 || j!=0 || k!=0 )
        {
          int nci = (ci+2+i) % 2;
          int ncj = (cj+2+j) % 2;
          int nck = (ck+2+k) % 2;
          int ni = -i;
          int nj = -j;
          int nk = -k;
          int nbh_val = pattern[nci][ncj][nck].dep[ni+1][nj+1][nk+1];
          int my_val = pattern[ci][cj][ck].dep[i+1][j+1][k+1];
          assert( my_val == !nbh_val && my_val != -1 );
  //        cout << "\t("<<i<<','<<j<<','<<k<<") = "<< my_val << endl;
        }
      }
    }
#   endif

  }

  GridDependencyPattern GridDependencyGraph::s_grid_pattern{};

  void GridDependencyGraph::adj_matrix(std::vector<bool>& mat)
  {
    size_t n_cells = m_cell.size();
//    cout << "S²="<<n_cells*n_cells<<endl;
    mat.assign( n_cells*n_cells , false );
    //size_t no = 0;
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      const size_t* ptask = m_deps.data() + m_start[cell_i];
      size_t ndeps = m_start[cell_i+1] - m_start[cell_i];
      for(size_t di=0;di<ndeps;di++)
      {
        size_t cell_j = ptask[di];
//        mat[ cell_i*n_cells+cell_j ] = true;
        mat[ cell_j*n_cells+cell_i ] = true;
        // ++ no;
      }
    }
//    cout << "N="<<no<<endl;
  }

  void GridDependencyGraph::closure_matrix(std::vector<bool>& mat)
  {
    adj_matrix( mat );
  
    size_t n_cells = m_cell.size();
    std::vector<bool> tmp;
    size_t n;
    do
    {
      n = 0;
      tmp.assign( mat.size() , false );
      for(size_t cell_j=0;cell_j<n_cells;cell_j++)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        bool p = false;
        for(size_t k=0;k<n_cells;k++)
        {
          p = p || ( mat[k*n_cells+cell_i] && mat[cell_j*n_cells+k] );
        }
        tmp[cell_j*n_cells+cell_i] = p;
      }
      for(size_t cell_j=0;cell_j<n_cells;cell_j++)
      for(size_t cell_i=0;cell_i<n_cells;cell_i++)
      {
        if( tmp[cell_j*n_cells+cell_i] && !mat[cell_j*n_cells+cell_i] )
        {
          mat[cell_j*n_cells+cell_i] = true;
          ++n;
        }
      }
//      cout << "N="<<n<<endl;
    }
    while( n > 0 );

    n=0;
    for(size_t cell_j=0;cell_j<n_cells;cell_j++)
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      if( !mat[cell_j*n_cells+cell_i] ) ++n;
    }
//    cout << "N0="<<n<<endl;    
  }

  bool GridDependencyGraph::check(IJK dims)
  {
    size_t n_cells = dims.i*dims.j*dims.k;
    assert( n_cells == m_cell.size() );

    std::vector<bool> mat;
    closure_matrix(mat);

    GRID_FOR_BEGIN(dims,cell_i,cell_loc)
    {
      for(int i=-1;i<=1;i++)
      for(int j=-1;j<=1;j++)
      for(int k=-1;k<=1;k++)
      {
        if(i!=0 || j!=0 || k!=0)
        {
          ssize_t nci = cell_loc.i+i;
          ssize_t ncj = cell_loc.j+j;
          ssize_t nck = cell_loc.k+k;
          if( nci>=0 && nci<dims.i && ncj>=0 && ncj<dims.j && nck>=0 && nck<dims.k )
          {
            IJK nbh_loc = {nci,ncj,nck};
            size_t cell_j = grid_ijk_to_index(dims,nbh_loc);
            bool f = mat[cell_j*n_cells+cell_i]; 
            bool b = mat[cell_i*n_cells+cell_j]; 
            assert( f || b );
            if( ! (f || b) ) return false;
          }
        }
      }
    }
    GRID_FOR_END

    return true;
  }

  void GridDependencyGraph::build(IJK dims)
  {
    size_t n_cells = dims.i*dims.j*dims.k;

//    auto T0 = std::chrono::high_resolution_clock::now();

    std::vector<int> max_dist( n_cells , 0 );
    std::vector< std::unordered_set<int> > dep_out( n_cells );
    GRID_FOR_BEGIN(dims,_,cell_loc)
    {
      size_t cell_i = grid_ijk_to_index(dims,cell_loc);
      int ci=cell_loc.i%2;
      int cj=cell_loc.j%2;
      int ck=cell_loc.k%2;
      for(int i=-1;i<=1;i++)
      for(int j=-1;j<=1;j++)
      for(int k=-1;k<=1;k++)
      {
        if(i!=0 || j!=0 || k!=0)
        {
          ssize_t nci = cell_loc.i+i;
          ssize_t ncj = cell_loc.j+j;
          ssize_t nck = cell_loc.k+k;
          if( nci>=0 && nci<dims.i && ncj>=0 && ncj<dims.j && nck>=0 && nck<dims.k )
          {
            if( s_grid_pattern.pattern[ci][cj][ck].dep[i+1][j+1][k+1] )
            {
              size_t nbh_cell = grid_ijk_to_index(dims,IJK{nci,ncj,nck});
              dep_out[nbh_cell].insert( cell_i );
              max_dist[cell_i] = std::max( max_dist[cell_i] , max_dist[nbh_cell]+1 );
            }
          }
        }
      }
    }
    GRID_FOR_END

    std::vector< std::unordered_set<int> > dep_in( n_cells );

    size_t max_dep_count = 0;
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      for(auto d:dep_out[cell_i])
      {
        if( max_dist[d] == (max_dist[cell_i]+1) )
        {
          dep_in[d].insert(cell_i);
        }
      }
    }

    size_t total_deps = 0;
    int dmin = n_cells;
    int dmax = 0;
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      max_dep_count = std::max( max_dep_count , dep_in[cell_i].size() );
      total_deps += dep_in[cell_i].size();
      dmin = std::min( dmin, max_dist[cell_i] );
      dmax = std::max( dmax, max_dist[cell_i] );
    }

    m_cell.resize(n_cells);
    m_start.resize(n_cells+1);
    m_deps.resize(total_deps);
    total_deps=0;
    for(size_t cell_i=0;cell_i<n_cells;cell_i++)
    {
      m_cell[cell_i] = cell_i;
      m_start[cell_i] = total_deps;
      for(int d:dep_in[cell_i]) { m_deps[total_deps++]=d; }
    }
    m_start[n_cells] = total_deps;
      
    std::stable_sort( m_cell.begin() , m_cell.end() ,
      [&max_dist](int a,int b)->bool
      {
        return max_dist[a] < max_dist[b];
      });

//    std::chrono::duration<double,std::milli> elapsed = std::chrono::high_resolution_clock::now() - T0;
//    cout <<"max deps="<<max_dep_count<< ", total="<<total_deps<<", n_cells="<<n_cells <<", max_dist=["<<dmin<<" ; "<<dmax<<"] , time = "<<elapsed.count()<<" ms" <<endl;
    
    assert( check( dims ) );
    m_grid_dims = dims;
  }


}

