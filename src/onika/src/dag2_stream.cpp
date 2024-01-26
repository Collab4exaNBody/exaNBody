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
#include <onika/dag/dag2_stream.h>
#include <onika/oarray_stream.h>
#include <onika/stream_utils.h>
#include <onika/macro_utils.h>

#include <onika/oarray.h>
#include <onika/grid_grain.h>
#include <onika/dag/dot.h>
#include <onika/color_scale.h>

#include <cassert>
#include <random>
#include <iostream>
#include <sstream>
#include <functional>

//#include <map>

namespace onika
{
  namespace dag
  {

    template<size_t Nd>
    std::ostream& dag_to_dot(
      const WorkShareDAG2<Nd>& dag ,
      const oarray_t<size_t,Nd>& domain ,
      std::ostream& out ,
      Dag2DotConfig<Nd> && config
      )
    {    
      static const double depth_shift = 0.25;
      static const double gmargin = -0.5;
      static const double posscaling = 1.5;
      static const double gscaling = 1.0;
      static const double gsize = 0.75;
      static const double wave_margin = 0.1;
      static const bool viscorners = false;

      assert( size_t(domain_size(domain)) == dag.number_of_items() );

      onika_bind_vars_move(config,coord_func, mask_func, gw, bbenlarge, urenlarge, grainsize, fdp, add_legend, add_bounds_corner, movie_bounds, wave_group);

      if( coord_func == nullptr )
      {
        coord_func = [&dag](size_t i) -> oarray_t<size_t,Nd> { return dag.item_coord(i); };
      }
      if( mask_func == nullptr )
      {
        mask_func = [](size_t) -> bool { return true; };
      }
      //auto item_index = [&domain]( const oarray_t<size_t,Nd>& coord ) -> size_t { return coord_to_index(coord,domain); };

      // compute graph nodes dependency depth
      const size_t n_cells = dag.number_of_items();
      std::vector<int> node_dist( n_cells , -1 );
      bool conv;
      int all_md = -1;
      do
      {
        conv = true;
        for(size_t i=0;i<n_cells;i++)
        {
          int md = 0;
          //auto c = coord_func(i);
          for(const auto& d:dag.item_deps(i))
          {
            assert( d != i );
            md = std::max( md , node_dist[d] + 1 );
          }
          if( md > node_dist[i] ) { conv = false; node_dist[i] = md; }
          all_md = std::max( all_md , md );
        }
      }while( ! conv ); 


      std::ostringstream wout;

      // *************** compute node placement *****************
      std::vector< std::pair<double,double> > node_coords( n_cells , {0.0,-1.0} );
      const double side_spring = 10.0;
      const double dep_spring = 1.0;
      const double left_spring = 1.5; // original is 0.0
      const double height_scale = 0.85;
      const double height_offset = 0.0;
      const double dt = 0.01;
      std::vector<size_t> idx;
      std::vector<double> dep_pos;
      std::vector<double> position;
      std::vector<double> force;
      for(int dd=0;dd<=all_md;dd++)
      {
        idx.clear();
        position.clear();
        dep_pos.clear();
        size_t j=0;
        for(size_t i=0;i<n_cells;i++) 
        {
          //auto c = coord_func(i);
          //auto ci = item_index(c);
          if(node_dist[i]==dd)
          {
            idx.push_back(i);
            position.push_back(j);
            size_t n = 0;
            double dpos = 0.0;
            for(const auto& d:dag.item_deps(i))
            {
              //size_t di = item_index(d);
              assert( node_coords[d].second != -1.0 );
              dpos += node_coords[d].first;
              ++n;
            }
            if( n>0 ) dep_pos.push_back( dpos / n );
            else dep_pos.push_back(j);
            ++j;
          }
        }
        assert( j == idx.size() );
        assert( j == dep_pos.size() );
        assert( j == position.size() );
        size_t n = idx.size();
        force.resize(n);
        double tf = std::numeric_limits<double>::max();
        int k;
        for(k=0;k<10000 && tf>1.e-9;k++)
        {
          tf = 0.0;
          for(size_t i=0;i<n;i++)
          {
            double left_position = i;
            force[i] = 0.0;
            force[i] += ( dep_pos[i] - position[i] ) * dep_spring;
            force[i] += ( left_position - position[i] ) * left_spring;
            if( i>0 && std::fabs(position[i]-position[i-1]) < 1.0 )
            {
              force[i] += ( (position[i-1]+1.0) - position[i] ) * side_spring;
            }
            if( i<(n-1) && std::fabs(position[i]-position[i+1]) < 1.0 )
            {
              force[i] += ( (position[i+1]-1.0) - position[i] ) * side_spring;
            }
            tf += std::abs(force[i]*dt);
          }
          for(size_t i=0;i<n;i++) position[i] += force[i]*dt;
        }
        //std::cout<<"dd="<<dd<<", n="<<n<<", niter="<<k<<", force = "<<tf<<"\n";
        double wave_height = ( all_md - dd ) * height_scale + height_offset;
        for(size_t i=0;i<n;i++) { node_coords[idx[i]] = { position[i] , wave_height }; }

        if( wave_group && n>=1 )
        {
          double left_border = node_coords[idx[0]].first * grainsize * posscaling - gsize*0.5 - wave_margin;
          double right_border = node_coords[idx[n-1]].first * grainsize * posscaling + gsize*0.5 + wave_margin;
          double center = ( left_border + right_border ) * 0.5;
          double width = right_border - left_border;          
          wout << "wave_"<<dd << " [label=\"\",shape=box,style=\"rounded,dashed\",fillcolor=\"#000000ff\",color=\"black\",fillcolor=\"white\","
              << node_position( center, wave_height* grainsize * posscaling - wave_margin,
                                ZeroArray<size_t,Nd>::zero , gscaling , 0. , width , gsize + wave_margin*2 ) << "] ;\n";
        }
      }
      double posavg = 0.0;
      for(const auto& p:node_coords) posavg += p.first;
      posavg /= n_cells;
      // ************************************************************************

      auto compute_bb = [&](int dd, double bb[4]) -> void
      {
        for(size_t i=0;i<n_cells;i++) if(dd==-1 || node_dist[i]==dd)
        {     
          if( movie_bounds || gw>0.0 )
          {
            auto c = coord_func(i);
            double x = c[0];
            double y = 0.0;
            if constexpr (Nd>=2) y = c[1];
            x = x * grainsize * posscaling;
            y = y * grainsize * posscaling;
            bb[0] = std::min( bb[0] , x+gmargin - gsize*0.5 );
            bb[1] = std::min( bb[1] , y+gmargin - gsize*0.5 );
            bb[2] = std::max( bb[2] , x+gmargin + gsize*0.5 );
            bb[3] = std::max( bb[3] , y+gmargin + gsize*0.5 );
          }
          if( movie_bounds || gw<1.0 )
          {
            double x = node_coords[i].first;
            double y = node_coords[i].second;
            x = x * grainsize * posscaling;
            y = y * grainsize * posscaling;
            bb[0] = std::min( bb[0] , x+gmargin - gsize*0.5 );
            bb[1] = std::min( bb[1] , y+gmargin - gsize*0.5 );
            bb[2] = std::max( bb[2] , x+gmargin + gsize*0.5 );
            bb[3] = std::max( bb[3] , y+gmargin + gsize*0.5 );
          }
        }
      };

      // compute bounding box
      double bb[4] = { std::numeric_limits<double>::max() , std::numeric_limits<double>::max() , std::numeric_limits<double>::lowest() , std::numeric_limits<double>::lowest() };
      compute_bb(-1,bb);
      
      // bounding box enlargment
      bb[0] -= bbenlarge.first;
      bb[1] -= bbenlarge.second;
      bb[2] += bbenlarge.first + urenlarge.first;
      bb[3] += bbenlarge.second + urenlarge.second;

      if( wave_group )
      {
        std::string wn = wout.str(); wout.str("");
        wout << "digraph overlaywavegroup {\noverlap=\"true\"\n";
        wout << wn << "\n";
        for(int dd=0;dd<all_md;dd++)
        {
          wout << "wave_"<<dd <<" -> wave_"<< (dd+1) << "\n";
        }
        /*if( add_bounds_corner )
        {
          wout << bb_corners(bb,viscorners,"wave_");
        }*/
        wout << "}\n";
      }

      // draw dependences from shadowed previous task
      bool draw_prev_task_nodes = ( all_md == 1 );

      GridGrain<Nd> ggrid{ size_t(grainsize) };

      // graph global attributes     
      out << "digraph G\n{\n";
      out << "outputorder=\"nodesfirst\"\n";
      out << "overlap=\"true\"\n";
      out << "splines=\"true\"\n";
      //out << "bb=\""<<bb[0]<<","<<bb[1]<<","<<bb[2]<<","<<bb[3]<<"\"\n";
      
      // draw legend
      if( add_legend )
      {
        double legendx = ( bb[0] + bb[2] ) / 2.0 ;
        double legendy = 1.0;
        if constexpr ( Nd>=2 ) legendy = fdp ? domain[1] : (all_md+1) ;
        out<<"legend [label="<< ColorMapLegend{all_md,false} <<",shape=\"none\",margin=0,pos=\""<<legendx*grainsize*posscaling<<","<< legendy*grainsize*posscaling <<"!\""
            << (add_legend?"":",style=invis") <<"] ;\n";
      }

      std::ostringstream gout;
      if( grainsize > 1 )
      {
        gout << "digraph overlaygraph {\noverlap=\"true\"\n";
      }
            
      // 1. describe nodes
      for(size_t i=0;i<n_cells;i++)
      {
        auto c = coord_func(i);
                
        // cluster / single node position :
        double x = c[0];
        double y = 0.0;
        if constexpr (Nd>=2) y = c[1];
        x = x * gw + node_coords[i].first  * (1.-gw);
        y = y * gw + node_coords[i].second * (1.-gw);
        x = x * grainsize * posscaling;
        y = y * grainsize * posscaling;
        
        auto md = node_dist[i];

        if( draw_prev_task_nodes && mask_func(i) )
        {
          out << prev_node_name(c) << " [label=\""<<format_array(c)<<"\",shape=box,style=\"rounded,filled\",fillcolor=\"grey\","
              << node_position( x,y , ZeroArray<size_t,Nd>::zero  , 0.0 , depth_shift*posscaling , grainsize ) <<"] ;\n";      
        }

        // surronding group
        if( grainsize > 1 )
        {
          out << cluster_name(c) << " [label=\"\",shape=box,style=\"rounded,dashed,filled\",color=\"black\",fillcolor=\"white\","
              << node_position( x,y, ZeroArray<size_t,Nd>::zero , 0.0 , 0.0 , grainsize ) << "] ;\n";
          //gout << cluster_name(c) << " [label=\"\",style=invis,"<< node_position( x,y, ZeroArray<size_t,Nd>::zero , 0.0 , 0.0 , grainsize ) << "] ;\n";
        }

        // sub grid nodes
        grid_grain_apply( ggrid ,
          [&](const oarray_t<size_t,Nd>& gc) -> void
          {
            auto cs = subgridcoord(ggrid,c,gc);
            double text_shading = mask_func(i) ? 1.0 : 0.25;
            double cell_shading = mask_func(i) ? 1.0 : 0.15;
            const char* eattr = mask_func(i) ? "" : ",dashed";
            auto text_color = format_color_www( RGBAColord{0.,0.,0.,text_shading} );
            auto cell_color = format_color_www( cielab_discreet_colormap(md,all_md,cell_shading) );
            //std::cout<<"c="<<format_array(c)<<" gc="<<format_array(gc)<<" gg="<<ggrid.GrainSize<<" cs="<<format_array(cs)<<" nodename="<<node_name(cs)<< "\n";
            gout << node_name(cs) << " [label= <<font color='"<<text_color<<"'>"<< format_array(cs) << "</font>>,shape=box,style=\"rounded"<<eattr<<",filled\",color=\"black\",fillcolor=\""<< cell_color <<"\","
                 << node_position( x,y, gc , gscaling , gmargin , gsize ) << "] ;\n";
          });
      }

      if( grainsize > 1 )
      {
        //gout << "bb=\""<<bb[0]<<","<<bb[1]<<","<<bb[2]<<","<<bb[3]<<"\"\n";
        if( add_bounds_corner )
        {
          gout << bb_corners(bb,viscorners,"o_");
        }
        gout << "}\n";
      }
      else
      {
        out << gout.str();
        if( add_bounds_corner )
        {
          out << bb_corners(bb,viscorners);
        }
        gout.str("");
      }

      // 2. describe edges
      for(size_t i=0;i<n_cells;i++)
      {
        //auto ci = i; //item_index(c);
        auto c = coord_func(i);        
        auto md = node_dist[i];

        std::string target;        
        if( grainsize > 1 ) OStringStream(target) << cluster_name(c);
        else OStringStream(target) << node_name(c);

        // enforce legend positioning if not in fdp mode
        if( add_legend )
        {
          if( md == 0 && !fdp ) out << "legend -> "<< target << " [style=invis];\n";
        }

        for(const auto& d:dag.item_deps(i))
        {
          std::string origin;         
          if( draw_prev_task_nodes )
          {
            OStringStream(origin) << prev_node_name( coord_func(d) );
          }
          else
          {
            if( grainsize > 1 ) OStringStream(origin) << cluster_name( coord_func(d) );
            else OStringStream(origin) << node_name( coord_func(d) );
          }

          DotAttributes attribs;
          if( fdp ) attribs.add("constraint=false");
          if( wave_group ) attribs.add("style=invis");
          //else if( ! mask_func(i) ) attribs.add("style=dashed");
          out << origin <<" -> "<< target << " " << attribs << ";\n" ;
        }
      }

      out << "}\n" << gout.str() << "\n" << wout.str() << "\n";
      return out;
    }

    template std::ostream& dag_to_dot( const WorkShareDAG2<1> & , const oarray_t<size_t,1> & , std::ostream& , Dag2DotConfig<1> && );
    template std::ostream& dag_to_dot( const WorkShareDAG2<2> & , const oarray_t<size_t,2> & , std::ostream& , Dag2DotConfig<2> && );
    template std::ostream& dag_to_dot( const WorkShareDAG2<3> & , const oarray_t<size_t,3> & , std::ostream& , Dag2DotConfig<3> && );
  }
}
 
