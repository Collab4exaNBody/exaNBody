#include <onika/dag/dag_stream.h>
#include <onika/oarray_stream.h>
#include <onika/stream_utils.h>

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
      const WorkShareDAG<Nd>& dag ,
      const oarray_t<size_t,Nd>& domain ,
      std::ostream& out ,
      double position_scramble,
      int grainsize,
      bool fdp,
      std::function< oarray_t<size_t,Nd>(size_t) > coord_func,
      std::function< bool(const oarray_t<size_t,Nd>& c) > mask_func )
    {    
      static const double depth_shift = 0.25;
      static const double gmargin = -0.5;
      static const double posscaling = 1.5;
      static const double gscaling = 1.0;
      static const double gsize = 0.75;

      assert( size_t(domain_size(domain)) == dag.number_of_items() );

      if( coord_func == nullptr )
      {
        coord_func = [&dag](size_t i) -> oarray_t<size_t,Nd> { return dag.item_coord(i); };
      }
      if( mask_func == nullptr )
      {
        mask_func = [](const oarray_t<size_t,Nd>&) -> bool { return true; };
      }
      auto item_index = [&domain]( const oarray_t<size_t,Nd>& coord ) -> size_t { return coord_to_index(coord,domain); };

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
          auto c = coord_func(i);
          for(const auto& d:dag.item_deps(i)) if( d != c ) md = std::max( md , node_dist[item_index(d)] + 1 );
          auto& cmd = node_dist[ item_index( coord_func(i) ) ];
          if(md>cmd) { conv = false; cmd = md; }
          all_md = std::max( all_md , md );
        }
      }while( ! conv ); 



      // *************** compute node placement *****************
      std::vector< std::pair<double,double> > node_coords( n_cells , {0.0,-1.0} );
      const double side_spring = 10.0;
      const double dep_spring = 1.0;
      const double left_spring = 0.0;
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
          auto c = coord_func(i);
          auto ci = item_index(c);
          if(node_dist[ci]==dd)
          {
            idx.push_back(ci);
            position.push_back(j);
            size_t n = 0;
            double dpos = 0.0;
            for(const auto& d:dag.item_deps(i))
            {
              size_t di = item_index(d);
              assert( node_coords[di].second != -1.0 );
              dpos += node_coords[di].first;
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
        for(size_t i=0;i<n;i++) { node_coords[idx[i]] = { position[i] , all_md - dd }; }
      }
      double posavg = 0.0;
      for(const auto& p:node_coords) posavg += p.first;
      posavg /= n_cells;
      // ************************************************************************

      // draw dependences from shadowed previous task
      bool draw_prev_task_nodes = ( all_md == 1 );

      std::mt19937_64 re(1976);
      std::uniform_real_distribution<> rndpos(0.0,position_scramble);

      GridGrain<Nd> ggrid{ size_t(grainsize) };

      // graph global attributes     
      out << "digraph G\n{\n";
      out << "outputorder=\"nodesfirst\"\n";
      /*if( fdp )*/ out << "overlap=\"true\"\n";
      out << "splines=\"true\"\n";
      //out << "bb=\""<<bb[0]<<","<<bb[1]<<","<<bb[2]<<","<<bb[3]<<"\"\n";
            
      // draw legend
      double legendx = fdp ? (domain[0] * 0.5) : posavg ;
      double legendy = 1.0;
      if constexpr ( Nd>=2 ) legendy = fdp ? domain[1] : (all_md+1) ;
      out<<"legend [label="<< ColorMapLegend{all_md,false} <<",shape=\"none\",margin=0,pos=\""<<legendx*grainsize*posscaling<<","<< legendy*grainsize*posscaling <<"!\"] ;\n";

      std::ostringstream gout;
      if( grainsize > 1 )
      {
        gout << "digraph overlaygraph {\noverlap=\"true\"\n";
      }
      
      double bb[4] = { std::numeric_limits<double>::max() , std::numeric_limits<double>::max() , std::numeric_limits<double>::lowest() , std::numeric_limits<double>::lowest() };
      
      // 1. describe nodes
      for(size_t i=0;i<n_cells;i++)
      {
        auto c = coord_func(i);
        auto ci = item_index(c);
        auto md = node_dist[ci];
                
        // cluster / single node position :
        double x = c[0];
        double y = 0.0;
        if constexpr (Nd>=2) y = c[1];
        if( ! fdp ){ x=node_coords[ci].first; y=node_coords[ci].second; }
        x = x * grainsize * posscaling;
        y = y * grainsize * posscaling;
        
        bb[0] = std::min( bb[0] , x-grainsize*0.5 );
        bb[1] = std::min( bb[1] , y-grainsize*0.5 );
        bb[2] = std::max( bb[2] , x+grainsize*0.5 );
        bb[3] = std::max( bb[3] , y+grainsize*0.5 );
        
        if( draw_prev_task_nodes && mask_func(c) )
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
            //std::cout<<"c="<<format_array(c)<<" gc="<<format_array(gc)<<" gg="<<ggrid.GrainSize<<" cs="<<format_array(cs)<<" nodename="<<node_name(cs)<< "\n";
            gout << node_name(cs) << " [label=\""<< format_array(cs) << "\",shape=box,style=\"rounded,filled\",color=\""<< format_color_www(cielab_discreet_colormap(md,all_md)) <<"\","
                 << node_position( x,y, gc , gscaling , gmargin , gsize ) << "] ;\n";
          });
      }

      if( grainsize > 1 )
      {
        //gout << "bb=\""<<bb[0]<<","<<bb[1]<<","<<bb[2]<<","<<bb[3]<<"\"\n"; 
        gout << "llcorner [label=\"\",fixedsize=\"true\",width=\"0.01\",height=\"0.01\",style=invis,pos=\""<<bb[0] <<","<<bb[1] <<"!\"];\n";
        gout << "urcorner [label=\"\",fixedsize=\"true\",width=\"0.01\",height=\"0.01\",style=invis,pos=\""<<bb[2] <<","<<bb[3] <<"!\"];\n";
        gout << "}\n";
      }
      else
      {
        out << gout.str();
        gout.str("");
      }

      // 2. describe edges
      for(size_t i=0;i<n_cells;i++)
      {
        auto c = coord_func(i);        
        auto ci = item_index(c);
        auto md = node_dist[ci];

        std::string target;        
        if( grainsize > 1 ) OStringStream(target) << cluster_name(c);
        else OStringStream(target) << node_name(c);

        // enforce legend positioning if not in fdp mode
        if( md == 0 && !fdp ) out << "legend -> "<< target << " [style=invis];\n";

        for(const auto& d:dag.item_deps(i))
        {
          std::string origin;         
          if( draw_prev_task_nodes )
          {
            OStringStream(origin) << prev_node_name(d);
          }
          else
          {
            if( grainsize > 1 ) OStringStream(origin) << cluster_name(d);
            else OStringStream(origin) << node_name(d);
          }

          DotAttributes attribs;
          if( fdp ) attribs.add("constraint=false");
          out << origin <<" -> "<< target << " " << attribs << ";\n" ;
        }
      }

      out << "}\n" << gout.str();
      return out;
    }

    template std::ostream& dag_to_dot( const WorkShareDAG<1>& ,
                                       const oarray_t<size_t,1>& ,
                                       std::ostream& out ,
                                       double, int , bool ,
                                       std::function< oarray_t<size_t,1>(size_t) > ,
                                       std::function< bool(const oarray_t<size_t,1>& c) > );

    template std::ostream& dag_to_dot( const WorkShareDAG<2>& ,
                                       const oarray_t<size_t,2>& ,
                                       std::ostream& out ,
                                       double, int , bool ,
                                       std::function< oarray_t<size_t,2>(size_t) > ,
                                       std::function< bool(const oarray_t<size_t,2>& c) > );

    template std::ostream& dag_to_dot( const WorkShareDAG<3>& ,
                                       const oarray_t<size_t,3>& ,
                                       std::ostream& out ,
                                       double, int , bool ,
                                       std::function< oarray_t<size_t,3>(size_t) > ,
                                       std::function< bool(const oarray_t<size_t,3>& c) > );
  }
}
 
