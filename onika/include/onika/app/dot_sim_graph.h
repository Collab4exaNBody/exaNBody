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
#pragma once

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot_base.h>

#include <set>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace onika
{
  namespace app
  {

    struct DotGraphOutput
    {
      bool expose_batch_slots = true;
      bool remove_batch_ended_path = true;
      bool invisible_batch_slots = true;
      bool resource_graph_mode = false;

      using OperatorNode = onika::scg::OperatorNode;
      using OperatorSlotBase = onika::scg::OperatorSlotBase;

      std::unordered_map< OperatorSlotBase* , int > pseudo_resource_id_map;

      void aggregate_subgraph( 
            OperatorNode* op
          , std::map< OperatorNode* , std::vector<OperatorSlotBase*> >& clusters
          , std::map< OperatorNode* , std::vector<OperatorNode*> >& graph
          , std::map< OperatorSlotBase* , OperatorNode* >& node_container
          , const std::set<OperatorNode*>& shrunk_nodes
          , OperatorNode* container_op
          , bool collapse_subgraph = false );

      int dot_subgraph(
            std::ofstream& fout
          , OperatorNode* op
          , std::map< OperatorNode* , std::vector<OperatorSlotBase*> >& clusters
          , std::map< OperatorNode* , std::vector<OperatorNode*> >& graph
          , std::map<OperatorSlotBase*,std::string>& nodes
          , const std::set<OperatorNode*>& shrunk_nodes
          , const std::set<OperatorSlotBase*>& crossing_nodes
          , const std::set<OperatorNode*>& source_nodes
          , const std::set<OperatorNode*>& sink_nodes
          , bool collapse_subgraph = false
          , int traversal_rank = 0 );

      void apply_graph_path(
          std::unordered_map<OperatorSlotBase*,std::vector<OperatorSlotBase*> >& graph
        , OperatorSlotBase* n
        , const std::vector<OperatorSlotBase*>& path
        , std::function<bool(const std::vector<OperatorSlotBase*>&,OperatorSlotBase*)> sf
        , std::function<bool(const std::vector<OperatorSlotBase*>&,OperatorSlotBase*)> ef
        , std::function<void(const std::vector<OperatorSlotBase*>&)> pf);

      void dot_sim_graph(
          OperatorNode* simulation_graph
        , const std::string& filename = "sim_graph.dot"
        , bool show_unconnected_slots = false
        , const std::set<OperatorNode*>& shrunk_nodes = std::set<OperatorNode*>() );
    };

  }
}


