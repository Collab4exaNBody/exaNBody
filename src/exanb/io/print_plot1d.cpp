#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/string_utils.h>
#include <onika/plot1d.h>

#include <vector>
#include <string>

namespace exanb
{

  class PrintPlot1D : public OperatorNode
  {      
    using StringList = std::vector<std::string>;
    using Plot1DSet = onika::Plot1DSet;

    ADD_SLOT( long       , nmarks    , INPUT , 10 );
    ADD_SLOT( double     , scaling   , INPUT , 1.0 );
    ADD_SLOT( Plot1DSet  , plots     , INPUT );
    ADD_SLOT( StringList , plot_names, INPUT , std::string("") , DocString{"name of plot to print"} );

  public:
    inline void execute () override final
    {
      const long nm = *nmarks;
      
      for(const auto& p : plots->m_plots)
      {
        ldbg << "plot "<<p.first << " : size="<<p.second.size() << std::endl;
      }
      
      for(const auto& name : *plot_names)
      {
        const auto & plot = plots->m_plots[ name ];
        size_t n = plot.size();

        lout << "==================== "<<format_string("%-12s",plots->m_captions[ name ]) <<" ===============" << std::endl;

        if( n > 0 )
        {
          double y_min = plot[0].second;
          double y_max = plot[0].second;
          for(size_t i=1;i<n;i++)
          {
            y_min = std::min( y_min , plot[i].second );
            y_max = std::max( y_max , plot[i].second );
          }

          lout << "           ";
          for(int i=0;i<=(nm+1);i++) { lout << format_string("%.1e   ", y_min + ( i * (y_max-y_min) ) / static_cast<double>(nm) ); }
          lout << std::endl << "            ";
          for(int i=0;i<=nm;i++) { lout << "|_________"; }
          lout << '|' << std::endl;

          for(size_t i=0;i<n;i++)
          {
            //double raw_count = data[i];
            long count = ( (plot[i].second-y_min) * 10 * nm * (*scaling) ) / (y_max-y_min);
            bool overflow = false;
            if( count < 0 ) count=0;
            if( count > (10*(nm+1)) )
            {
              count = 10*(nm+1) - 1;
              overflow = true;
            }
    //        lout << format_string("% .3e : %.1e : ",histogram->m_min_val + ( i * (histogram->m_max_val - histogram->m_min_val) ) / n, raw_count ) ;
            lout << format_string("% .4e ", plot[i].first ) ;
            if( count == 0 )
            {
              if( plot[i].second > y_min ) lout << '.';
            }
            else
            {
              for(int j=0;j<(count-1);j++) { lout << ' '; }
              if( ! overflow ) { lout << '*'; }
              else { lout << ">"; }
            }
            lout << std::endl;
          }
        }
        else
        {
          lout << "<empty>" << std::endl;
        }
        lout << "==============================================" << std::endl;
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
   OperatorNodeFactory::instance()->register_factory( "print_plot1d" , make_compatible_operator< PrintPlot1D > );
  }

}

