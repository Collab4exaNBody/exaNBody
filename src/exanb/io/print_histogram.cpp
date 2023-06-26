#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/histogram.h>

#include <vector>
#include <string>

namespace exanb
{

  struct PrintHistogramOperator : public OperatorNode
  {      
    ADD_SLOT( std::string , message   , INPUT , std::string("histogram") );
    ADD_SLOT( long        , nmarks    , INPUT , 10 );
    ADD_SLOT( double      , scaling   , INPUT , 1.0 );
    ADD_SLOT( Histogram<> , histogram , INPUT , REQUIRED );

    inline void execute () override final
    {
      size_t n = histogram->m_data.size();
      double* data = histogram->m_data.data();

      double min_count = data[0];
      double max_count = data[0];
      for(size_t i=1;i<n;i++)
      {
        min_count = std::min( min_count , data[i] );
        max_count = std::max( max_count , data[i] );
      }

      long nm = *nmarks;
      double sfactor = *scaling;

      double median_value = 0.0;
      size_t median_bin = 0;
      double median_count = 0.0;
      double total_count = 0.0;
      double total_value = 0.0;
      {
        for(size_t i=0;i<n;i++)
        {
          double value = histogram->m_min_val + ( i * (histogram->m_max_val - histogram->m_min_val) ) / n ;
          double count = data[i];
          median_value += value * count;
        }
        median_value /= 2.0;
        double min_dist = median_value;
        for(size_t i=0;i<n;i++)
        {
          double value = histogram->m_min_val + ( i * (histogram->m_max_val - histogram->m_min_val) ) / n ;
          double count = data[i];
          if( std::abs(total_value-median_value) < min_dist )
          {
            median_bin = i;
            min_dist = std::abs(total_value-median_value);
            median_count = total_count;
          }
          total_count += count;
          total_value += count * value;
        }
      }

      lout << std::endl
           << "==================== Histogram ===============" << std::endl
           << "=== "<< format_string("%-39s",*message) << "===" << std::endl
           << "=== "<< format_string("count : %.2e / %.2e / %.2e",median_count,total_count-median_count,total_count) << " ===" << std::endl
           << "=== "<< format_string("total : %.3e %24s",total_value,"===") << std::endl
           << "==============================================" << std::endl
           << "           ";

      for(int i=0;i<=(nm+1);i++) { lout << format_string("%.1e   ", min_count + ( i * (max_count-min_count) ) / sfactor / static_cast<double>(nm) ); }
      lout << std::endl << "            ";
      for(int i=0;i<=nm;i++) { lout << "|_________"; }
      lout << '|' << std::endl;

      for(size_t i=0;i<n;i++)
      {
        //double raw_count = data[i];
        long count = ( (data[i]-min_count) * 10 * nm * sfactor ) / (max_count-min_count);
        bool overflow = false;
        if( count < 0 ) count=0;
        if( count > (10*(nm+1)) )
        {
          count = 10*(nm+1) - 1;
          overflow = true;
        }
//        lout << format_string("% .3e : %.1e : ",histogram->m_min_val + ( i * (histogram->m_max_val - histogram->m_min_val) ) / n, raw_count ) ;
        lout << format_string("%c% .3e ", ( (i==median_bin) ? '*' : ' ' ) , histogram->m_min_val + ( i * (histogram->m_max_val - histogram->m_min_val) ) / n ) ;
        if( count == 0 )
        {
          if( data[i] != min_count ) { lout << '.'; }
        }
        else
        {
          for(int j=0;j<count;j++) { lout << '#'; }
          if( overflow ) { lout << ">>"; }
        }
        lout << std::endl;
      }
    }
    
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "print_histogram" , make_compatible_operator< PrintHistogramOperator > );
  }

}

