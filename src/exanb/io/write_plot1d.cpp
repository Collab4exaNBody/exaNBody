#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/string_utils.h>
#include <onika/plot1d.h>

#include <vector>
#include <string>

namespace exanb
{

  class WritePlot1D : public OperatorNode
  {      
    using StringList = std::vector<std::string>;
    using Plot1DSet = onika::Plot1DSet;

    ADD_SLOT( Plot1DSet  , plots     , INPUT );
    ADD_SLOT( StringList , plot_names, INPUT , StringList{} , DocString{"name of plot to print"} );
    ADD_SLOT( std::string, separator , INPUT , std::string(" ; ") , DocString{"filename. if mutliple plots are selected, plot name will be appended to filename"} );
    ADD_SLOT( std::string, filename  , INPUT , std::string("plot.csv") , DocString{"filename. if mutliple plots are selected, plot name will be appended to filename"} );

  public:
    inline void execute () override final
    { 
      for(const auto& p : plots->m_plots)
      {
        ldbg << "plot "<<p.first << " : size="<<p.second.size() << std::endl;
      }

      const bool multiple = ( plot_names->size() > 1 ); 
      for(const auto& name : *plot_names)
      {
        const auto & plot = plots->m_plots[ name ];
        size_t n = plot.size();
        if( n > 0 )
        {
          std::string fname = *filename;
          if( multiple ) fname = fname + "." + name;
          std::ofstream fout( fname ); 
          for(size_t i=0;i<n;i++)
          {
            fout << plot[i].first << (*separator) << plot[i].second << std::endl;
          }
        }
      }
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsSequence() )
      {
        tmp["plot_names"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize(tmp);
    }
    
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "write_plot1d" , make_compatible_operator< WritePlot1D > );
  }

}

