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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot_base.h>
#include <onika/log.h>

#include "dot_sim_graph.h"

#include <fstream>
#include <sstream>

using namespace exanb;

void DotGraphOutput::aggregate_subgraph( 
      OperatorNode* op
    , std::map< OperatorNode* , std::vector<OperatorSlotBase*> >& clusters
    , std::map< OperatorNode* , std::vector<OperatorNode*> >& graph
    , std::map< OperatorSlotBase* , OperatorNode* >& node_container
    , const std::set<OperatorNode*>& shrunk_nodes
    , OperatorNode* container_op
    , bool collapse_subgraph )
{
  if( op == nullptr ) return ;
  if( op->name() == "nop" || clusters[op].empty() ) return;
  
  collapse_subgraph = collapse_subgraph || ( shrunk_nodes.find(op) != shrunk_nodes.end() );
  
  for(auto n:clusters[op])
  {
    node_container[ n ] = container_op;
//    fout << nodes[n] << " [label=\""<<n->name()<<"\"];" << std::endl;
  }
    
  for(auto subop:graph[op])
  {
    OperatorNode* next_container = nullptr;
    if( collapse_subgraph ) { next_container = container_op; }
    else { next_container = subop; }
    
    aggregate_subgraph( subop, clusters, graph, node_container, shrunk_nodes, next_container , collapse_subgraph );
  }

}



int DotGraphOutput::dot_subgraph(
      std::ofstream& fout
    , OperatorNode* op
    , std::map< OperatorNode* , std::vector<OperatorSlotBase*> >& clusters
    , std::map< OperatorNode* , std::vector<OperatorNode*> >& graph
    , std::map<OperatorSlotBase*,std::string>& nodes
    , const std::set<OperatorNode*>& shrunk_nodes
    , const std::set< OperatorSlotBase* >& crossing_nodes
    , const std::set<OperatorNode*>& source_nodes
    , const std::set<OperatorNode*>& sink_nodes
    , bool collapse_subgraph
    , int traversal_rank )
{
  if( op == nullptr ) return traversal_rank;
  if( op->name() == "nop" || clusters[op].empty() ) return traversal_rank;
 
  bool is_sink = sink_nodes.find(op) != sink_nodes.end();
  bool is_source = source_nodes.find(op) != source_nodes.end();
  
  if( ! collapse_subgraph )
  {
    fout << "subgraph cluster_" << op->name()<<(void*)op<<" {"<<std::endl;
    fout << "label = \""<<op->name()<<"\" ;" << std::endl;
    if( is_sink )
    {
      if( is_source ) fout << "bgcolor=lightgreen;" << std::endl;
      else fout << "bgcolor=cyan;" << std::endl;
    }
    else if( is_source )
    {
       fout << "bgcolor=yellow;" << std::endl;
    }
  }

  bool curlevelcollapse = collapse_subgraph;
  collapse_subgraph = collapse_subgraph || ( shrunk_nodes.find(op) != shrunk_nodes.end() );
  
  if( op->is_terminal() || !resource_graph_mode )
  {
    fout << "rank = same;" << std::endl;
    for(auto n:clusters[op])
    {
      if( crossing_nodes.find(n) != crossing_nodes.end() || n->is_conditional_input() )
      {
        fout << nodes[n] << " [";
        if( !n->owner()->is_terminal() && invisible_batch_slots && !n->is_conditional_input() )
        {
          fout<<"style=\"invisible\", shape=point, fixedsize=true";
        }
        else
        {
          fout <<"label=\""<<n->name()<<"\"";
          if( n->is_conditional_input() ) { fout << ", color=orange, style=filled"; }
        }
        fout << "];" << std::endl;
      }
    }
  }
    
  ++ traversal_rank;
  for(auto subop:graph[op])
  {
    traversal_rank = dot_subgraph( fout, subop, clusters, graph, nodes, shrunk_nodes, crossing_nodes, source_nodes, sink_nodes, collapse_subgraph, traversal_rank );
  }
  
  collapse_subgraph = curlevelcollapse;
  if( ! collapse_subgraph )
  {
    fout <<"}"<<std::endl;
  }
  
  return traversal_rank;
}


void DotGraphOutput::apply_graph_path(
    std::unordered_map<OperatorSlotBase*,std::vector<OperatorSlotBase*> >& graph
  , OperatorSlotBase* n
  , const std::vector<OperatorSlotBase*>& path
  , std::function<bool(const std::vector<OperatorSlotBase*>&,OperatorSlotBase*)> sf
  , std::function<bool(const std::vector<OperatorSlotBase*>&,OperatorSlotBase*)> ef
  , std::function<void(const std::vector<OperatorSlotBase*>&)> pf)
{

  bool path_start = sf(path,n);
  bool path_end = ef(path,n);
  std::vector<OperatorSlotBase*> spath = path;
  
  if( path_start || path_end )
  {
    spath.push_back(n);
  }

  if( path_end )
  {
    pf( spath );    
  }
  
  if( path_start )
  {
    sf = [](const std::vector<OperatorSlotBase*>&,OperatorSlotBase*) -> bool {return true;} ;
  }

  for(auto sn:graph[n])
  {
    apply_graph_path(
        graph
      , sn
      , spath
      , sf
      , ef
      , pf );
  }
}

void DotGraphOutput::dot_sim_graph(exanb::OperatorNode* simulation_graph, const std::string& filename, bool show_unconnected_slots, const std::set<OperatorNode*>& shrunk_nodes)
{
  std::map< OperatorNode* , std::vector<OperatorNode*> > graph;
  OperatorNode* root = nullptr;

  std::map< OperatorNode* , std::vector<OperatorSlotBase*> > clusters;
  
  std::map<OperatorSlotBase*,std::string> nodes;
  std::set< std::pair<OperatorSlotBase*,OperatorSlotBase*> > edges;
  std::set<OperatorNode*> source_nodes;
  std::set<OperatorNode*> sink_nodes;

  simulation_graph->apply_graph(
    [&source_nodes,&sink_nodes](OperatorNode* op)
    {
      source_nodes.insert(op);
      sink_nodes.insert(op);
    });
    
  std::ofstream fout( filename );
  fout << "digraph sim {" << std::endl;
  //fout << "newrank=true;"<<std::endl;

  simulation_graph->apply_graph(
    [/*&fout,*/&nodes,&edges,&graph,&root,&clusters,&source_nodes,&sink_nodes,show_unconnected_slots,this](OperatorNode* op)
    {
      if( op->parent() == nullptr )
      {
        if( root != nullptr )
        {
          lerr << "internal error, multiple root operator nodes" << std::endl;
          std::abort();
        }
        root = op;
      }
      else
      {
        graph[op->parent()].push_back(op);
      }
                
      for( const auto& s : op->named_slots() )
      {
        if( s.second->owner() != op )
        {
          lerr << "bad slot ownership" << std::endl;
          std::abort();
        }
        bool has_connections = s.second->input()!=nullptr || ! s.second->outputs().empty() ;
        if( show_unconnected_slots || has_connections )
        {
          std::ostringstream oss;
          oss << s.second->name() << "_" << (void*) s.second;
          nodes[s.second] = oss.str();
          clusters[op].push_back( s.second );
          
          if( s.second->input() != nullptr && (op->is_terminal() || expose_batch_slots) )
          {
            auto i = s.second->input();
            if( ! expose_batch_slots )
            {
              while(i!=nullptr && ! i->owner()->is_terminal() )
              {
                sink_nodes.erase( i->owner() );
                i = i->input();
              }
            }
            if( i==nullptr )
            {
              if( s.second->resource()->is_null() && s.second->is_required() )
              {
                lerr << s.second->pathname() << " has no terminal node input and has no initializer and is required to have a value" << std::endl;
                std::abort();
              }
            }
            if( i != nullptr )
            {
              source_nodes.erase( s.second->owner() );
              sink_nodes.erase( i->owner() );
              //if( s.second->is_conditional_input() ) std::cout<<"conditional input : "<<s.second->pathname()<<std::endl;
              edges.insert( std::pair<OperatorSlotBase*,OperatorSlotBase*>( i /*s.second->input()*/ , s.second ) );
            }
          }
        }
      }
    }); 

  if( root == nullptr )
  {
    lerr << "internal error : no root operator node found" << std::endl;
    std::abort();
  }

  // add edges for conditional slots
  for(const auto& n : nodes)
  {
    if( n.first->is_conditional_input() )
    {
      if( n.first->input() != nullptr )
      {
        auto i = n.first->input();
        if( ! expose_batch_slots )
        {
          while(i!=nullptr && ! i->owner()->is_terminal() )
          {
            sink_nodes.erase( i->owner() );
            i = i->input();
          }
        }
        sink_nodes.erase( i->owner() );
        source_nodes.erase( n.first->owner() );
        edges.insert( std::pair<OperatorSlotBase*,OperatorSlotBase*>( i , n.first ) );
        //fout << nodes[ n.first->input() ] << " -> " << nodes[ n.first ] << " ;" << std::endl;
      }
    }
  }
  
  std::cout << "show_unconnected_slots = "<< show_unconnected_slots <<std::endl;
  std::map< OperatorSlotBase* , OperatorNode* > node_container;
  aggregate_subgraph(root,clusters,graph,node_container,shrunk_nodes,root);
  std::set< OperatorSlotBase* > crossing_nodes;

  std::set<OperatorNode*> clusters_without_edges;
  simulation_graph->apply_graph( [&clusters_without_edges](OperatorNode* op) { clusters_without_edges.insert(op); });
  
  if( remove_batch_ended_path )
  {
    std::unordered_map<OperatorSlotBase*,std::vector<OperatorSlotBase*> > slot_graph;
    std::unordered_set<OperatorSlotBase*> not_source_slots;
    std::unordered_set<OperatorSlotBase*> to_keep;
    for(const auto& e:edges)
    {
      slot_graph[ e.first ].push_back(e.second);
      not_source_slots.insert(e.second);
    }
    
    int resource_id = 1;
    for(const auto& e:edges)
    {
      if( not_source_slots.find(e.first)==not_source_slots.end() )
      {
        apply_graph_path(
            slot_graph
          , e.first
          , {}
          , [](const std::vector<OperatorSlotBase*>& p,OperatorSlotBase* n)->bool{ return p.empty() && n->owner()->is_terminal(); }
          , [](const std::vector<OperatorSlotBase*>& p,OperatorSlotBase* n)->bool{ return !p.empty() && (n->owner()->is_terminal() || n->is_conditional_input()); }
          , [&to_keep](const std::vector<OperatorSlotBase*>& p) { for(auto n:p) to_keep.insert(n); }
          );
        if( resource_graph_mode )
        {
          apply_graph_path(
              slot_graph
            , e.first
            , {}
            , [](const std::vector<OperatorSlotBase*>& p,OperatorSlotBase* n)->bool{ return true; }
            , [](const std::vector<OperatorSlotBase*>& p,OperatorSlotBase* n)->bool{ return n->outputs().empty(); }
            , [resource_id,this](const std::vector<OperatorSlotBase*>& p) { for(auto n:p) pseudo_resource_id_map[n] = resource_id; }
            );
          ++ resource_id;
        }
      }
    }
    for(const auto& e:edges)
    {
      if( to_keep.find(e.first)==to_keep.end() ) { node_container[e.first]=nullptr; }
      if( to_keep.find(e.second)==to_keep.end() ) { node_container[e.second]=nullptr; }
    }    
  }
  
  for(const auto& e:edges)
  {
    if( node_container[e.first] != node_container[e.second] && node_container[e.first]!=nullptr && node_container[e.second]!=nullptr )
    {
      clusters_without_edges.erase( node_container[e.first] );
      clusters_without_edges.erase( node_container[e.second] );
      crossing_nodes.insert( e.first );
      crossing_nodes.insert( e.second );
    }
  }
  for(const auto& op: clusters_without_edges)
  {
    source_nodes.erase(op);
    sink_nodes.erase(op);
  }
  
  dot_subgraph(fout,root,clusters,graph,nodes, shrunk_nodes, crossing_nodes, source_nodes, sink_nodes );

  if( resource_graph_mode )
  {
    //fout << "subgraph cluster_AllResources {"<<std::endl;
    //fout << "label = \"resources\" ;" << std::endl;
    std::unordered_set<int> used_resources;
    std::unordered_map<int,std::string> resource_name;
    std::unordered_set<OperatorSlotBase*> slot_nodes;
    for(const auto& e:edges)
    {
      if( node_container[e.first] != node_container[e.second] && node_container[e.first]!=nullptr && node_container[e.second]!=nullptr )
      {
        if( e.first->owner()->is_terminal() )
        {
          int resource_id = pseudo_resource_id_map[ e.first ];
          used_resources.insert( resource_id );
          resource_name[ resource_id ] = e.first->name();
          slot_nodes.insert(e.first);
        }
        if( e.second->owner()->is_terminal() )
        {        
          int resource_id = pseudo_resource_id_map[ e.second ];
          used_resources.insert( resource_id );
          resource_name[ resource_id ] = e.second->name();
          slot_nodes.insert(e.second);
        }
      }
    }
    for(auto r:used_resources)
    {
      fout << "resource_"<<r<< " [label=\""<<resource_name[r]<<"\", color=lightgreen] ;" << std::endl;
    }
    //fout << "}"  << std::endl;
    for(auto s:slot_nodes)
    {
      fout << nodes[s] << " -> resource_" << pseudo_resource_id_map[s] << " ;" << std::endl;
    }
  }
  else
  {
    for(const auto& e:edges)
    {
      if( node_container[e.first] != node_container[e.second] && node_container[e.first]!=nullptr && node_container[e.second]!=nullptr )
      {
        bool invisible_end = ( !e.second->owner()->is_terminal() && !e.second->is_conditional_input() ) && invisible_batch_slots;
        bool scattering_node = false;
        if( invisible_end )
        {
          int n_secondary_arcs = 0;
          for(auto sn:e.second->outputs())
          {
            auto se_it = edges.find( { e.second , sn } );
            if(se_it!=edges.end())
            {
              auto se = *se_it;
              assert( se.first == e.second );
              if( node_container[e.second] != node_container[se.second] && node_container[e.second]!=nullptr && node_container[se.second]!=nullptr )
              {
                ++ n_secondary_arcs;
              }
            }
          }
          scattering_node = ( n_secondary_arcs > 1 );
        }
        bool arrow_end = !invisible_end || scattering_node;
        fout << nodes[ e.first ] << " -> " << nodes[ e.second ];
        if( !arrow_end ) { fout << " [arrowhead=none]"; }
        fout << " ;" << std::endl;
      }
    }
  }

  fout << "}" << std::endl;
}



