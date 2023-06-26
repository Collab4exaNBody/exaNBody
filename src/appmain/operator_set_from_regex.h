#pragma once

#include <exanb/core/operator.h>
#include <unordered_set>
#include <vector>
#include <string>
#include <regex>

std::unordered_set<size_t> operator_set_from_regex(
  std::shared_ptr<exanb::OperatorNode> graph,
  const std::vector<std::string> & re_strings,
  const std::vector< std::pair<std::string,std::size_t> >& special_values = {} ,
  const std::string& message = "")
{
  using namespace exanb;

  std::unordered_set<size_t> hashes;
  for(const std::string& f : re_strings)
  {
    const std::regex re(f);
    for(const auto& sv:special_values)
    {
      if( std::regex_match(sv.first,re) )
      {
        if( hashes.find(sv.second) == hashes.end() )
        {
          if( ! message.empty() ) ldbg << message << sv.first << std::endl;
          hashes.insert(sv.second);
        }
      }
    }
    graph->apply_graph(
      [&re,&hashes,&message](OperatorNode* op)
      {
        if( std::regex_match(op->pathname(),re) )
        {
          auto H = op->hash();
          if( hashes.find( H ) == hashes.end() )
          {
            if( ! message.empty() ) ldbg << message << op->pathname() << std::endl;
            hashes.insert( H );
          }
        }
      });
  }
  return hashes;
}
