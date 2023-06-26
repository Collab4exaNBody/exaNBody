#pragma once

#include <fstream>
#include <iostream>
#include <functional>
#include <memory>
#include <unordered_set>
#include <string>
#include <sstream>

#include <exanb/core/basic_types_stream.h>

#include <onika/stream_utils.h>

namespace exanb
{

  static inline std::ostream& null_stream_manip(std::ostream& os) { return os; }
	  //static inline std::ostream& default_end_line(std::ostream& os) { return os << std::endl ; }

  struct LogStreamWrapper
  {
    int m_fd = -1;
    std::function<std::ostream&(void)> m_out = nullptr;
    std::function<std::ostream&(std::ostream&)> m_begin_line = null_stream_manip;
    std::function<std::ostream&(std::ostream&)> m_end_line = std::endl<char,std::char_traits<char>>;

    std::unordered_set<size_t> m_filters; // set of label hash for which log is enabled
    bool m_filter_enable = false;
    bool m_line_start = true;
            
    // utility function to open a file and use it as output stream
    void open( const std::string& file_name );
    void set_filters( const std::unordered_set<size_t>& filters );
    bool is_a_tty() const;
    void progress_bar( const std::string mesg, double ratio );
    
    // return a stream with respect to label filtering
    LogStreamWrapper& filter(size_t label);
  };

  // add a mechanism so that this is available only if ostream << T is overloaded
  template<typename T>
  static inline LogStreamWrapper& operator << ( LogStreamWrapper& log , const T& x )
  {
    if( log.m_out )
    {
#     pragma omp critical(exanb_log_manip)
      {
        if(log.m_line_start) { log.m_begin_line( log.m_out() ); }
        log.m_line_start = false;
        log.m_out() << x;
      }
    }
    return log;
  }

  template<class T>
  inline LogStreamWrapper& operator << ( LogStreamWrapper& log , const onika::PrintableFormattedObject<T>& a )
  {
    if( log.m_out )
    {
#     pragma omp critical(exanb_log_manip)
      {
        if(log.m_line_start) { log.m_begin_line( log.m_out() ); }
        log.m_line_start = false;
        a.to_stream( log.m_out() );
      }
    }
    return log;
  }

  // LogStream specific IO manip
  inline LogStreamWrapper& cret( LogStreamWrapper& log )
  {
    if( log.m_out )
    {
#     pragma omp critical(exanb_log_manip)
      {
        log.m_out() << '\r' << std::flush; 
        log.m_line_start = true;
      }
    }
    return log;
  }

  inline LogStreamWrapper& operator << ( LogStreamWrapper& log , LogStreamWrapper& (*manip)(LogStreamWrapper&) )
  {
    return manip(log);
  }

  // std manipulators compatibility
  inline LogStreamWrapper& operator << ( LogStreamWrapper& log , std::ostream& (*manip)(std::ostream&) )
  {
    static std::ostream& (*endl_value)(std::ostream&) = std::endl<char,std::char_traits<char> >;
    
    if( log.m_out )
    {
#     pragma omp critical(exanb_log_manip)
      {
        if( manip == endl_value )
        {
          if(log.m_line_start)
          {
            log.m_begin_line( log.m_out() );
          }
          log.m_line_start = true;
        }
        manip( log.m_out() );
      }
    }
    return log;
  }

  void configure_logging(bool debug, bool parallel_log,
                         std::string out_file_name,
                         std::string err_file_name,
                         std::string dbg_file_name,
                         int mpi_rank=0, int mpi_size=1);

  extern LogStreamWrapper lout;
  extern LogStreamWrapper lerr;
  extern LogStreamWrapper ldbg_raw;
    
  std::ostream& default_stream_format(std::ostream& out);

  // helper that replaces ldbg with a filtered version of the log stream
  struct LogStreamFilterHelper
  {
    LogStreamWrapper& m_log;
    uint64_t m_filter_hash;
    template<typename T>
    inline LogStreamWrapper& operator << (const T& x)
    {
      return m_log.filter(m_filter_hash) << x;
    }
    
    inline LogStreamWrapper& operator << ( std::ostream& (*manip)(std::ostream&) )
    {
      return m_log.filter(m_filter_hash) << manip ;
    }

    inline bool is_a_tty() const { return m_log.is_a_tty(); }
  };

  // create a log filter for debug messages out of operator scope
  extern LogStreamFilterHelper ldbg;


  // helper methods to avoid global variable capture in lambdas or OpenMP structured blocks
  inline auto& lout_stream() { return lout; }
  inline auto& lerr_stream() { return lerr; }
  inline auto& ldbg_stream() { return ldbg; }

  struct FatalErrorLogStream
  {
    std::ostringstream m_oss;
    
    inline FatalErrorLogStream() {}
    
    template<class T> inline FatalErrorLogStream& operator << (const T& x)
    {
      m_oss << x;
      return *this;
    }
    inline FatalErrorLogStream& operator << ( std::ostream& (*manip)(std::ostream&) )
    {
       m_oss << manip ;
       return *this;
    }
    ~FatalErrorLogStream();
  };
  
  inline FatalErrorLogStream fatal_error() { return FatalErrorLogStream(); }
}

