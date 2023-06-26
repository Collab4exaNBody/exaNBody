#include "vite_operator_functions.h"
#include <exanb/core/operator.h>
#include <random>


struct ViteOperatorRandomColoring
{
  std::unordered_map<std::string, std::unordered_map<std::string,ViteEventColor> > event_color;
  std::mt19937_64 re {0};
  inline ViteEventColor operator () (const ViteTraceElement& e)
  {
    using namespace exanb;
    std::uniform_real_distribution<> rndcol(0.3,1.0);
    const OperatorNode* op = reinterpret_cast<const OperatorNode*>( e.app_ctx );
    if( op == nullptr ) return { 1.0, 0.0, 0.0 };
    const std::string& opname = op->name();
    std::string tag = ( (e.tag==nullptr) ? "null" : e.tag );
    auto op_it = event_color.find(opname);
    if( op_it == event_color.end() )
    {
      double r = rndcol(re);
      double g = rndcol(re);
      double b = rndcol(re);
      event_color[opname][tag] = {r,g,b};
    }
    else
    {
      assert( ! op_it->second.empty() );
      auto col = op_it->second.begin()->second;
      event_color[opname][tag] = col;
    }
    return event_color[opname][tag];
  }
};
ViteColoringFunction g_vite_operator_rnd_color = ViteOperatorRandomColoring{};

ViteLabelFunction g_vite_operator_label = [](const ViteTraceElement& e) -> std::string
  {
    using namespace exanb;
    const OperatorNode* op = reinterpret_cast<const OperatorNode*>( e.app_ctx );
    std::string s;
    if( op != nullptr ) { s = op->name() + "@"; }
    else { s = "null@"; }
    if( e.tag != nullptr )
    {
      const char* s2 = std::strrchr(e.tag,'/');
      s2 = ( (s2!=nullptr) ? (s2+1) : e.tag );
      s += s2;
    }
    return s;
  };

