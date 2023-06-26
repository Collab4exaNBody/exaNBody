#pragma once

#include <exanb/core/operator.h>

#include <set>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

struct DotGraphOutput
{
  bool expose_batch_slots = true;
  bool remove_batch_ended_path = true;
  bool invisible_batch_slots = true;
  bool resource_graph_mode = false;

  std::unordered_map< exanb::OperatorSlotBase* , int > pseudo_resource_id_map;

  void aggregate_subgraph( 
        exanb::OperatorNode* op
      , std::map< exanb::OperatorNode* , std::vector<exanb::OperatorSlotBase*> >& clusters
      , std::map< exanb::OperatorNode* , std::vector<exanb::OperatorNode*> >& graph
      , std::map< exanb::OperatorSlotBase* , exanb::OperatorNode* >& node_container
      , const std::set<exanb::OperatorNode*>& shrunk_nodes
      , exanb::OperatorNode* container_op
      , bool collapse_subgraph = false );

  int dot_subgraph(
        std::ofstream& fout
      , exanb::OperatorNode* op
      , std::map< exanb::OperatorNode* , std::vector<exanb::OperatorSlotBase*> >& clusters
      , std::map< exanb::OperatorNode* , std::vector<exanb::OperatorNode*> >& graph
      , std::map<exanb::OperatorSlotBase*,std::string>& nodes
      , const std::set<exanb::OperatorNode*>& shrunk_nodes
      , const std::set<exanb::OperatorSlotBase*>& crossing_nodes
      , const std::set<exanb::OperatorNode*>& source_nodes
      , const std::set<exanb::OperatorNode*>& sink_nodes
      , bool collapse_subgraph = false
      , int traversal_rank = 0 );

  void apply_graph_path(
      std::unordered_map<exanb::OperatorSlotBase*,std::vector<exanb::OperatorSlotBase*> >& graph
    , exanb::OperatorSlotBase* n
    , const std::vector<exanb::OperatorSlotBase*>& path
    , std::function<bool(const std::vector<exanb::OperatorSlotBase*>&,exanb::OperatorSlotBase*)> sf
    , std::function<bool(const std::vector<exanb::OperatorSlotBase*>&,exanb::OperatorSlotBase*)> ef
    , std::function<void(const std::vector<exanb::OperatorSlotBase*>&)> pf);

  void dot_sim_graph(
      exanb::OperatorNode* simulation_graph
    , const std::string& filename = "sim_graph.dot"
    , bool show_unconnected_slots = false
    , const std::set<exanb::OperatorNode*>& shrunk_nodes = std::set<exanb::OperatorNode*>() );
};

