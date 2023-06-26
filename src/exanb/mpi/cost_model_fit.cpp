#include <exanb/mpi/simple_cost_model.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/polynomial_fit.h>
#include <exanb/core/histogram.h>
#include <cmath>
#include <mpi.h>

namespace exanb
{

  // simple cost model where the cost of a cell is the number of particles in it
  // 
  template< class GridT >
  class CostModelFit : public OperatorNode
  {
    using DoubleVector = std::vector<double>;
  
    ADD_SLOT( MPI_Comm , mpi , INPUT , REQUIRED );
    ADD_SLOT( GridT , grid ,INPUT_OUTPUT , REQUIRED);
    
    ADD_SLOT( DoubleVector , cost_model_coefs , INPUT_OUTPUT, DoubleVector({0.0,0.0,1.0,1.0}) , DocString{"Polynomial coefs for cost function. input is particle density per volume unit"} ); 

    ADD_SLOT( long , samples , INPUT , 256 );
    ADD_SLOT( long , order , INPUT , 3 );

    ADD_SLOT( Histogram<> , histogram , OUTPUT );

  public:

    inline void execute () override final
    {
      if( ! grid->cell_profiling() )
      {
        lout << "Cell profiling data not available" << std::endl;
        return;
      }

      std::vector<double> & coefs = *cost_model_coefs;

      const int ghost_layers = grid->ghost_layers();
      const double cell_size = grid->cell_size();
      const double cell_volume = cell_size * cell_size * cell_size;
      const IJK dims = grid->dimension() - 2 * ghost_layers;
      const auto cells = grid->cells();

      coefs.resize(4,0.0);
      ldbg << "CostModelFit: A="<<std::setprecision(3)<< coefs[0] << ", B="<< coefs[1] <<", C="<< coefs[2] <<", D="<< coefs[3] <<", vol="<<cell_volume<< ", order="<<*order<<std::endl;

      // Warning: for correctness we only account for inner cells (not ghost cells)

      double cost_model_max = 0;
      double measured_max = 0;
      double fit_error = 0;
      int n_samples = 0;

      std::vector< std::pair<double,double> > fit_data( grid->number_of_cells() , { 0.0 , 0.0 } );
      //std::vector<double> Y( grid->number_of_cells() , 0.0 );

      auto cell_profiler = grid->cell_profiler();

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(static) reduction(max:cost_model_max,measured_max) reduction(+:n_samples) )
        {
          const size_t cell_i = grid_ijk_to_index( dims , loc + ghost_layers );
          assert( cell_i >= 0 && cell_i < grid->number_of_cells() );
          const size_t N = cells[cell_i].size();
          double np = N;
          const double x = np / cell_volume;
          const double x2 = x*x;
          const double x3 = x2*x;
          const double model_cost = coefs[0]*x3 + coefs[1]*x2 + coefs[2]*x + coefs[3];
          const double measured_cost = cell_profiler.get_cell_time(cell_i) ;
          cost_model_max = std::max( cost_model_max , model_cost );          
          measured_max = std::max( measured_max , measured_cost );
          ++ n_samples;
        }
        GRID_OMP_FOR_END
      }

      MPI_Allreduce(MPI_IN_PLACE, &cost_model_max, 1, MPI_DOUBLE, MPI_MAX, *mpi);      
      MPI_Allreduce(MPI_IN_PLACE, &measured_max, 1, MPI_DOUBLE, MPI_MAX, *mpi);      

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(static) reduction(+:fit_error) )
        {
          const size_t cell_i = grid_ijk_to_index( dims , loc + ghost_layers );
          assert( cell_i >= 0 && cell_i < grid->number_of_cells() );
          const size_t N = cells[cell_i].size();
          double np = N;
          const double x = np / cell_volume;
          const double x2 = x*x;
          const double x3 = x2*x;
          const double model_cost = coefs[0]*x3 + coefs[1]*x2 + coefs[2]*x + coefs[3];
          const double measured_cost = cell_profiler.get_cell_time(cell_i) ;
          fit_data[ cell_i ] = { x , measured_cost };
          //Y[ cell_i ] = measured_cost;
          const double e = (model_cost/cost_model_max) - (measured_cost/measured_max);
          fit_error += e*e;
        }
        GRID_OMP_FOR_END
      }

      grid->set_cell_profiling( false );

      MPI_Allreduce(MPI_IN_PLACE, &fit_error, 1, MPI_DOUBLE, MPI_SUM, *mpi);      

      unsigned long long total_n_samples = n_samples;
      MPI_Allreduce(MPI_IN_PLACE, &total_n_samples, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi);      
      
      double min_x = fit_data[0].first;
      double max_x = min_x;
      for(const auto& p:fit_data)
      {
        min_x = std::min( min_x , p.first );
        max_x = std::max( max_x , p.first );
      }
      MPI_Allreduce(MPI_IN_PLACE, &min_x, 1, MPI_DOUBLE, MPI_MIN, *mpi);      
      MPI_Allreduce(MPI_IN_PLACE, &max_x, 1, MPI_DOUBLE, MPI_MAX, *mpi);

      ldbg << "total_n_samples = "<< total_n_samples << std::endl;
      ldbg << "X range = ["<< min_x<<";"<<max_x<<"] , measure_max = "<<measured_max<<" , model_max = "<<cost_model_max << std::endl;
      lout << "CostModelFit: Sum E2="<<std::setprecision(3)<<fit_error<<" , mean E2="<< fit_error / total_n_samples << std::endl;
      histogram->m_min_val = min_x;
      histogram->m_max_val = max_x;

      const unsigned int nbins = *samples;
      std::vector<unsigned long long> bincount( nbins , 0 );
      histogram->m_data.assign( nbins , 0.0 );
      for(const auto& p:fit_data)
      {
        int bin = static_cast<int>( std::floor( ( p.first - min_x ) * nbins / ( max_x - min_x ) ) );
        if(bin<0) bin = 0;
        else if( bin >= static_cast<int>(nbins) ) bin = nbins-1;
        ++ bincount[bin];
        histogram->m_data[bin] += p.second;
      }

      MPI_Allreduce(MPI_IN_PLACE, histogram->m_data.data(), nbins, MPI_DOUBLE, MPI_SUM, *mpi);      
      MPI_Allreduce(MPI_IN_PLACE, bincount.data(), nbins, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi);      

      std::vector<double> all_X( nbins , 0.0 );
      for(unsigned int i=0;i<nbins;i++)
      {
        all_X[i] = min_x + ( (i+0.5) / nbins * ( max_x - min_x ) );
        if( bincount[i] > 0 ) histogram->m_data[i] /= bincount[i];
      }

      int deg = *order;
      if(deg<1) deg=1;
      else if( deg>3 ) deg=2;

      coefs.assign( 4 , 0.0 );
      if( ! polynomial_regression_fit( all_X , histogram->m_data , deg , coefs ) )
      {
        lerr << "Could not fit cost model data"<< std::endl;
      }
      coefs.resize( 4 , 0.0 );
      std::swap( coefs[0] , coefs[3] );
      std::swap( coefs[1] , coefs[2] );

      lout << "CostModelFit: best fit parameters : A="<<std::setprecision(3)<< coefs[0] << ", B="<< coefs[1] <<", C="<< coefs[2] <<", D="<< coefs[3] << std::endl;
    }

  };
  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "cost_model_fit",
      make_grid_variant_operator< CostModelFit > );
  }

}

