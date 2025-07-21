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
    ADD_SLOT( StringList , plot_names, INPUT , StringList{} , DocString{"(YAML: list) Names of plots to write to file"} );
    ADD_SLOT( std::string, separator , INPUT , std::string(" ; ") , DocString{"(YAML: string) Separator to be used in csv file"} );
    ADD_SLOT( std::string, filename  , INPUT , std::string("plot.csv") , DocString{"(YAML: string) Filename. if mutliple plots are selected, each plot will be written to a separate file with an extension named after it except if boolean multicolumns is set to true."} );
    ADD_SLOT( bool,        multicolumns , INPUT , false, DocString{"(YAML: bool) If set to true and multiple plots are required, a single .csv file is created where the first column is the position along the slice direction and other columns are the value of the required fields."});

  public:
    inline void execute () override final
    { 
      for(const auto& p : plots->m_plots)
      {
        ldbg << "plot "<<p.first << " : size="<<p.second.size() << std::endl;
      }

      const bool multiple = ( plot_names->size() > 1 );
      if (not multiple) *multicolumns = false;
      
      if (*multicolumns && multiple)
        {
          const auto firstname = (*plot_names)[0];
          const auto & firstplot = plots->m_plots[ firstname ];
          size_t resolution = firstplot.size();
          
          // Gathering bins positions along slice direction
          std::vector<double> points;
          points.resize(resolution);
          for (size_t i=0;i<resolution;i++)
            {
              points[i] = firstplot[i].first;
            }
          
          std::string fname = *filename;
          fname = fname + ".csv";
          std::ofstream fout( fname ); 
          
          // Writing header to multicolumns .csv file
          fout << "# pos" << (*separator);
          for(const auto& name : *plot_names)
            {
              fout << name << (*separator);
            }
          fout << std::endl;
          
          if (resolution>0)
            {
              for (size_t i=0;i<resolution;i++)
                {
                  fout << points[i] << (*separator);
                  for (const auto& name : *plot_names)
                    {
                      const auto & plot = plots->m_plots[ name ];
                      fout << plot[i].second << (*separator);
                    }
                  fout << std::endl;
                }
            }
        } else {
        
        for(const auto& name : *plot_names)
          {
            const auto & plot = plots->m_plots[ name ];
            size_t n = plot.size();
            if( n > 0 )
              {
                std::string fname = *filename;
                if( multiple ) {
                  fname = fname + "_" + name + ".csv";
                } else {
                  fname = fname + ".csv";
                }
                  
                std::ofstream fout( fname ); 
                for(size_t i=0;i<n;i++)
                  {
                    fout << plot[i].first << (*separator) << plot[i].second << std::endl;
                  }
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

    inline std::string documentation() const override final
    {
      return R"EOF(
Write to file 1D Plots generated with grid_particle_slicing operator.

Usage example:

dump_data:
  - grid_particle_slicing:
      fields: [ vx, vy, vz ]
      thickness: 3.3 ang
      direction: [1,0,0]
      caption:
        "vx": "Velocity X"
        "vy": "Velocity Y"
        "vx": "Velocity X"
      average: [ "vx", "vy", "vz" ]
  - write_plot1d:
      plot_names: [ "vx", "vy", "vz" ]
      separator: " "
      filename: "plot_test"
      multicolumns: true
  - write_plot1d:
      plot_names: [ "vx", "vy", "vz" ]
      separator: " "
      filename: "plot_test"
      multicolumns: false

)EOF";
    }    
  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(write_plot1d)
  {
   OperatorNodeFactory::instance()->register_factory( "write_plot1d" , make_compatible_operator< WritePlot1D > );
  }

}

