#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/yaml_utils.h>
#include <memory>

namespace exanb
{

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class BeforeTimeStepNode : public OperatorNode
  {
  public:
  
    ADD_SLOT( long , timestep , INPUT);
    ADD_SLOT( long , end_at   , INPUT);
    ADD_SLOT( bool , result   , INPUT_OUTPUT);
    
    void execute() override final
    {
      long timestep = *(this->timestep);
      long end_at = *(this->end_at);
      bool& out = *result;
      
      out = ( timestep <= end_at );
      ldbg << "BeforeTimeStepNode: timestep="<<timestep<<", result="<<std::boolalpha<<out<<std::endl;
    }    
  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "before_timestep",
      []( const YAML::Node& node, const OperatorNodeFlavor& flavor) -> std::shared_ptr<OperatorNode> 
      {
        if( node.IsMap() || node.IsNull() )
        {
          return make_compatible_operator< BeforeTimeStepNode >(node,flavor);
        }
        else if( node.IsScalar() )
        {
          YAML::Node tmp;
          tmp["end_at"] = node;
          return make_compatible_operator< BeforeTimeStepNode >(tmp,flavor);
        }
        else
        {
          lerr << "before_timestep must have the form before_timestep: value or before_timestep: { end_at: value }" << std::endl;
          lerr << "--- YAML node content ---" << std::endl;
	  dump_node_to_stream( lerr , node );
          lerr << "-------------------------" << std::endl;
          std::abort();
        }
      });
  }

}

