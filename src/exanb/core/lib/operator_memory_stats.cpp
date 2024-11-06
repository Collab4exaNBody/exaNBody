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
#include <onika/string_utils.h>
#include <exanb/core/operator_memory_stats.h>

#include <string>
#include <fstream>
#include <map>
#include <set>
#include <cstdlib>
#include <mpi.h>

namespace exanb
{

  struct OperatorSlotBaseInfo
  {
    std::string m_typename;
    std::string m_name;
    std::set< std::string > m_pathnames;
    const void* m_ptr = nullptr;
    unsigned long long m_musage = 0;
    unsigned long long m_musage_min = -1;
    unsigned long long m_musage_avg = 0;
    unsigned long long m_musage_max = 0;
    unsigned long long m_repeat_same_name = 0;
  };

  using SlotInfoMap = std::map< const void* , OperatorSlotBaseInfo >;

  void build_slot_mem_info( SlotInfoMap& info, OperatorNode* op)
  {
    std::map<std::string,OperatorSlotBase*> named_slots;
    for( auto slot : op->slots() )
    {
      named_slots[slot->name()] = slot;
    }    
  
    for( auto& named_slot : named_slots )
    {
      auto slot = named_slot.second;
      // if( slot == nullptr ) { return; }
      void* ptr = slot->resource()->memory_ptr();
      if( ptr != nullptr )
      {
        if( info[ptr].m_typename.empty() )
        {
          info[ptr].m_ptr = ptr;
          info[ptr].m_typename = exanb::pretty_short_type( slot->value_type() );
          info[ptr].m_name = slot->name();
          info[ptr].m_musage = slot->memory_bytes();
          info[ptr].m_musage_min = 0;
          info[ptr].m_musage_avg = 0;
          info[ptr].m_musage_max = 0;
          info[nullptr].m_musage += info[ptr].m_musage;
        }
        info[ptr].m_pathnames.insert( slot->pathname() );
      }
    }
  }

  void print_operator_memory_stats(exanb::OperatorNode* simulation_graph , MPI_Comm comm , size_t musage_threshold )
  {
    int rank=0, np=1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    SlotInfoMap mem_info;    
    simulation_graph->apply_graph(
      [&mem_info](OperatorNode* o)
      {
        build_slot_mem_info(mem_info,o);
      }
    );

    std::map< std::string , OperatorSlotBaseInfo > m_ordered_info;
    for(const auto& p:mem_info)
    {
      if( ! p.second.m_pathnames.empty() )
      {
        m_ordered_info[ * p.second.m_pathnames.begin() ] = p.second;
      }
    }

    long n_values = m_ordered_info.size();    
    long n_values_min = -1;
    long n_values_max = -1;
    MPI_Allreduce(&n_values,&n_values_min,1,MPI_LONG,MPI_MIN,comm);
    MPI_Allreduce(&n_values,&n_values_max,1,MPI_LONG,MPI_MAX,comm);
    if( n_values_max == n_values_min )
    {
      //std::cout << "number of values match"<<std::endl;
      std::vector<long> all_values( n_values , -1 );
      std::vector<long> all_values_min( n_values , -1 );
      std::vector<long> all_values_max( n_values , -1 );
      int i=0;
      for(const auto& p:m_ordered_info)
      {
        assert( i < n_values );
        all_values[i] = std::hash<std::string>{}(p.first);
        ++i;
      }
      MPI_Allreduce(all_values.data(),all_values_min.data(),n_values,MPI_LONG,MPI_MIN,comm);
      MPI_Allreduce(all_values.data(),all_values_max.data(),n_values,MPI_LONG,MPI_MAX,comm);
      bool all_hash_match = true;
      for(i=0;i<n_values;i++)
      {
        all_hash_match = all_hash_match && all_values_min[i]==all_values_max[i] ;
      }
      if( all_hash_match )
      {
        //std::cout << "slot names match"<<std::endl;
        std::vector<long> all_values_sum( n_values , -1 );
        i=0;
        for(const auto& p:m_ordered_info)
        {
          assert( i < n_values );
          all_values[i] = (p.second.m_musage);
          ++i;
        }       
        MPI_Allreduce(all_values.data(),all_values_min.data(),n_values,MPI_LONG,MPI_MIN,comm);
        MPI_Allreduce(all_values.data(),all_values_max.data(),n_values,MPI_LONG,MPI_MAX,comm);
        MPI_Allreduce(all_values.data(),all_values_sum.data(),n_values,MPI_LONG,MPI_SUM,comm);
        mem_info[nullptr].m_musage = 0;
        i=0;
        for(auto& p:m_ordered_info)
        {
          assert( i < n_values );
          assert( mem_info[p.second.m_ptr].m_musage == p.second.m_musage );
          mem_info[p.second.m_ptr].m_musage = all_values_sum[i];
          mem_info[p.second.m_ptr].m_musage_min = all_values_min[i];
          mem_info[p.second.m_ptr].m_musage_max = all_values_max[i];
          mem_info[p.second.m_ptr].m_musage_avg = all_values_sum[i] / np;
          mem_info[nullptr].m_musage += all_values_sum[i];
          mem_info[nullptr].m_musage_min += all_values_min[i];
          mem_info[nullptr].m_musage_max += all_values_max[i];
          mem_info[nullptr].m_musage_avg += all_values_sum[i] / np;
          ++i;
        }
      }
      else
      {
        lerr <<"P"<<rank<<" : memory stats : slot names hash doesn't match across processors"<<std::endl;
      }
    }
    else
    {
      lerr <<"P"<<rank<<" : memory stats : number of values mismatch local="<<n_values<<", min="<<n_values_max<<", max="<<n_values_max<<std::endl;
    }

    // header    
    lout << format_string( "%-40s %-10s %-10s %-10s %-10s %-80s"," Slot name","Mem. tot","Mem. min","Mem. avg","Mem. max","data type") << std::endl;

/*
    std::vector<OperatorSlotBaseInfo> mem_info_vec;
    mem_info_vec.reserve( mem_info.size() );        
    for(auto p : mem_info )
    {
      if(p.first!=nullptr)
      {
        mem_info_vec.push_back( p.second );
      }
    }
*/

    std::map< std::tuple<ssize_t,std::string,std::string> , OperatorSlotBaseInfo > repeats;
    for(auto p : mem_info )
    {
      if(p.first!=nullptr)
      {
        assert( ! p.second.m_name.empty() );
        auto& info = repeats[ { - p.second.m_musage , p.second.m_name , p.second.m_typename } ];
        size_t n = info.m_repeat_same_name + 1;
        info = p.second;
        info.m_repeat_same_name = n;
      }
    }
  
//    std::sort( mem_info_vec.begin(), mem_info_vec.end() , [](const OperatorSlotBaseInfo& a , const OperatorSlotBaseInfo& b)->bool { return a.m_musage > b.m_musage; } );

    for( const auto& kv : repeats )
    {
      const auto& p = kv.second;
      assert( ! p.m_name.empty() );
      std::string name;
      if( p.m_repeat_same_name>1 ) name = format_string("%s x%d",p.m_name,p.m_repeat_same_name);
      else name = p.m_name;
      
      if( p.m_musage >= musage_threshold )
      {
        lout << format_string( "%-40s %-10s %-10s %-10s %-10s %-80s"
                              , name
                              , memory_bytes_string( p.m_musage ) 
                              , memory_bytes_string( p.m_musage_min ) 
                              , memory_bytes_string( p.m_musage_avg ) 
                              , memory_bytes_string( p.m_musage_max )
                              , p.m_typename ) << std::endl;
      }

      /*
      if( p.m_pathnames.size() > 1 )
      {
        for(const auto& pn : p.m_pathnames)
        {
          lout << format_string( "%-30s" , pn ) << std::endl;
        }
      }
      */
    }
    
    lout << format_string( "%-40s %-10s %-10s %-10s %-10s %-80s"
                          , "total"
                          , memory_bytes_string( mem_info[nullptr].m_musage ) 
                          , memory_bytes_string( mem_info[nullptr].m_musage_min ) 
                          , memory_bytes_string( mem_info[nullptr].m_musage_avg ) 
                          , memory_bytes_string( mem_info[nullptr].m_musage_max )
                          , " - " ) << std::endl;
    
  }

}
