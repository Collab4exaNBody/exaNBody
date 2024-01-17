#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/core/log.h>

namespace exanb
{

  struct BoolExprNode
  {
    enum Token { CONST_FALSE=-1, OP_AND=-2, OP_OR=-3, OP_NOT=-4, PAR_OPEN=-5, PAR_CLOSE=-6 };
    int op = CONST_FALSE;
    int l = -1;
    int r = -1;
    bool neg_l = false;
    bool neg_r = false;
  };

  static inline bool nand_binary_tree_is_leaf( const std::vector< BoolExprNode >& nodes , int i )
  {
    if( i==-1 ) return true;
    else if( nodes[i].op == BoolExprNode::CONST_FALSE || nodes[i].op>=0 ) return true;
    else return false;
  }

  static inline void nand_binary_tree_operands(const std::vector< BoolExprNode >& nodes, std::vector<int>& operands, int i)
  {
    if( i==-1 )
    {
      operands.push_back(-1);
    }
    else if( nodes[i].op==BoolExprNode::OP_AND )
    {
      nand_binary_tree_operands(nodes,operands,nodes[i].l);
      nand_binary_tree_operands(nodes,operands,nodes[i].r);
    }
    else
    {
      if( nodes[i].op<0 )
      {
        fatal_error() << "nand_binary_tree_operands: unexpected operator #"<<nodes[i].op<<" at "<<i<<std::endl;
      }
      operands.push_back(nodes[i].op);
    }
  }

  static inline void nand_binary_tree_operators(const std::vector< BoolExprNode >& nodes, uint64_t& opbits, int i, int level)
  {
    if( i==-1 || nodes[i].op != BoolExprNode::OP_AND )
    {
      fatal_error() << "nand_binary_tree_operators: unexpected token at "<<i<<" , level="<<level<<std::endl;
    }
    if(level==0)
    {
       opbits = opbits << 2;
       if( nodes[i].neg_l ) opbits |= 1ull;
       if( nodes[i].neg_r ) opbits |= 2ull;
    }
    else
    {
      nand_binary_tree_operators(nodes,opbits,nodes[i].r,level-1);
      nand_binary_tree_operators(nodes,opbits,nodes[i].l,level-1);
    }
  }

  /*
  static inline void nand_binary_tree_print( const std::vector< BoolExprNode >& nodes , int i , int depth=0)
  {
    for(int s=0;s<depth;s++) lout<<"   ";
    if(depth>0) lout<<"|_ ";

    if( nand_binary_tree_is_leaf(nodes,i) )
    {
      int value = -1;
      if( i != -1 ) value = nodes[i].op;
      if(value==-1) lout<<"False ("<<i<<")";
      else          lout<<"R"<<value<<" ("<<i<<")";
      lout<<std::endl;
    }
    else
    {
      switch(nodes[i].op)
      {
        case BoolExprNode::OP_AND : lout<<"AND-"; break;
        case BoolExprNode::OP_OR  : lout<<"OR-"; break;
        case BoolExprNode::OP_NOT : lout<<"NOT-"; break;
        case BoolExprNode::PAR_OPEN :
        case BoolExprNode::PAR_CLOSE : lout<<"()"; break;
        default:
          fatal_error()<<"unexpected operator #"<<nodes[i].op<<std::endl;
          break;
      }
      lout << ( nodes[i].neg_l ? 'N' : 'x' ) ;
      if(nodes[i].op==BoolExprNode::OP_AND || nodes[i].op==BoolExprNode::OP_OR)
      {
        lout << ( nodes[i].neg_r ? 'N' : 'x' ) << " ("<<i<<" -> "<<nodes[i].l<<","<<nodes[i].r<< ")" <<std::endl;
        nand_binary_tree_print(nodes,nodes[i].l,depth+1);
        nand_binary_tree_print(nodes,nodes[i].r,depth+1);
      }
      else
      {
        int s = ( nodes[i].l != -1 ) ? nodes[i].l : nodes[i].r;
        lout << " ("<<i<<" -> "<< s << ")" <<std::endl;
        nand_binary_tree_print(nodes,s,depth+1);
      }
    }
  }
*/

  static inline int nand_binary_tree_depth( const std::vector< BoolExprNode >& nodes , int i )
  {
    if( nand_binary_tree_is_leaf(nodes,i) )
    {
      return 0;
    }
    else
    {
      return 1 + std::max( nand_binary_tree_depth(nodes,nodes[i].l) , nand_binary_tree_depth(nodes,nodes[i].r) );
    }
  }

  static inline int nand_binary_tree_fill( std::vector< BoolExprNode >& nodes , int i , int depth )
  {
    if( depth>0 )
    {
      if( nand_binary_tree_is_leaf(nodes,i) )
      {
        nodes.push_back( { BoolExprNode::OP_AND , nand_binary_tree_fill(nodes,i,depth-1) , nand_binary_tree_fill(nodes,-1,depth-1) , false , true } );
        return nodes.size()-1;
      }
      else
      {
        nodes[i].l = nand_binary_tree_fill(nodes,nodes[i].l,depth-1);
        nodes[i].r = nand_binary_tree_fill(nodes,nodes[i].r,depth-1);
        return i;
      }
    }
    else if( ! nand_binary_tree_is_leaf(nodes,i) )
    {
      fatal_error() << "Internal error : nand binary tree too deep" << std::endl;
    }
    return i;
  }

  static inline bool make_nand_binary_tree( std::vector< BoolExprNode >& nodes, int & i )
  {
    if(i==-1)
    {
      fatal_error() << "Unexpected empty expression" << std::endl;
      return false;
    }
    else if( nodes[i].op == BoolExprNode::OP_NOT )
    {
      bool neg_op = make_nand_binary_tree( nodes, nodes[i].r );
      //lout << "supress NOT at "<<i<<", new expr root = "<<nodes[i].r<< ", result neg = "<< (! neg_op) <<std::endl;
      int r = nodes[i].r;
      nodes[i] = BoolExprNode{};
      i = r;
      return ! neg_op;
    }
    else if( nodes[i].op == BoolExprNode::OP_OR )
    {
      nodes[i].op = BoolExprNode::OP_AND;
      nodes[i].neg_l = ! make_nand_binary_tree( nodes, nodes[i].l );
      nodes[i].neg_r = ! make_nand_binary_tree( nodes, nodes[i].r );
      //lout << "convert OR at "<<i<<" to AND , neg_l="<<nodes[i].neg_l<<" ,neg_r="<<nodes[i].neg_r<<" ,l="<<nodes[i].l<<" ,r="<<nodes[i].r<<std::endl;
      return true;
    }
    else if( nodes[i].op == BoolExprNode::OP_AND )
    {
      nodes[i].neg_l = make_nand_binary_tree( nodes, nodes[i].l );
      nodes[i].neg_r = make_nand_binary_tree( nodes, nodes[i].r );      
    }
    else if( nodes[i].op < 0 )
    {
      fatal_error() << "unexepected operator #"<<nodes[i].op<<" at "<< i<<std::endl;
      return false;
    }
    return false;
  }

  static inline int fetch_expression( std::vector< BoolExprNode >& nodes, int& begin, int end );
  
  static inline int build_expression_tree( std::vector< BoolExprNode >& nodes, int begin, int end )
  {
    /*        
    lout << "build_expression_tree( "<<begin<<" , "<<end<<")" <<std::endl;
    for(unsigned int i=0;i<nodes.size();i++)
    {
      lout << "TOKEN "<<i<<" : OP="<<nodes[i].op<<", L="<<nodes[i].l<<", R="<<nodes[i].r
           <<", NegL="<<std::boolalpha<<nodes[i].neg_l<<", NegR="<<std::boolalpha<<nodes[i].neg_r<<std::endl;
    }
    lout << std::endl;
    */
    int e1 = fetch_expression( nodes , begin , end );
    while( begin < end )
    {
      int o = begin ++;
      //lout << "e1="<<e1<< " , o="<<o<< std::endl;
      if( nodes[o].op == BoolExprNode::OP_AND )
      {
        int e2 = fetch_expression( nodes , begin , end );
        //lout << "AND: e2="<<e2<< ", begin="<<begin<<", end="<<end<<std::endl;
        nodes[o].l = e1;
        nodes[o].r = e2;
      }
      else if( nodes[o].op == BoolExprNode::OP_OR )
      {
        int e2 = build_expression_tree( nodes , begin , end );
        begin = end;
        //lout << "OR: e2="<<e2<< ", begin="<<begin<<", end="<<end<< std::endl;
        nodes[o].l = e1;
        nodes[o].r = e2;
      }
      else
      {
        fatal_error() << "expected binary operator at "<<o<<" but found token #"<<nodes[o].op<<" instead" << std::endl;
      }
      e1 = o;
    }
    //lout << "final e1="<<e1<< std::endl;
    return e1;
  }

  // pickup first complete expression in token list
  static inline int fetch_expression( std::vector< BoolExprNode >& nodes, int& begin, int end )
  {
    /* 
    lout << "fetch_expression( "<<begin<<" , "<<end<<")" <<std::endl;
    for(unsigned int i=0;i<nodes.size();i++)
    {
      lout << "TOKEN "<<i<<" : OP="<<nodes[i].op<<", L="<<nodes[i].l<<", R="<<nodes[i].r
           <<", NegL="<<std::boolalpha<<nodes[i].neg_l<<", NegR="<<std::boolalpha<<nodes[i].neg_r<<std::endl;
    }
    lout << std::endl;
    */

    if( begin<0 || begin >= static_cast<ssize_t>(nodes.size()) || (end-1) >= static_cast<ssize_t>(nodes.size()) )
    {
      fatal_error() << "Internal error: boolean expression cannot be constructed" << std::endl;
      return -1;
    }

    if( begin >= end )
    {
      fatal_error() << "Empty expression: begin="<<begin<<" , end="<<end;
      return -1;
    }

    if( nodes[begin].op == BoolExprNode::OP_NOT )
    {
      if( begin+1 >= end )
      {
        fatal_error() << "'not' left alone in remaining expression at "<<begin<<std::endl;
      }
      int e = begin;
      ++ begin;
      nodes[e].r = fetch_expression( nodes , begin , end );
      return e;
    }
    
    if( nodes[begin].op == BoolExprNode::PAR_OPEN )
    {
      int par_end = begin;
      int c = 1;
      while( c>0 && par_end<end )
      {
        ++ par_end;
        if( par_end >= end )
        {
          fatal_error() << "')' expected before end of expression at token #"<<begin<<std::endl;
        }
        if( nodes[par_end].op == BoolExprNode::PAR_OPEN ) ++c;
        else if( nodes[par_end].op == BoolExprNode::PAR_CLOSE ) --c;
      }
      if( c != 0 )
      {
        fatal_error() << "mismatched '(' in expression at position "<<begin<<std::endl;
      }
      if( nodes[par_end].op != BoolExprNode::PAR_CLOSE )
      {
        fatal_error() << "internal error : expected ')' but found #"<<nodes[par_end].op<<" instead at " <<par_end << std::endl;
      }
      // sub expression construction
      int e = begin;
      nodes[e].r = build_expression_tree( nodes, begin+1 , par_end );
      begin = par_end+1; // remiaing expression after last perenthesis    
      return e;
    }

    return begin ++;
  }

  static inline int cleanup_expression_tree( std::vector< BoolExprNode >& nodes, int i )
  {
    if( i == -1 ) return -1;
    nodes[i].l = cleanup_expression_tree( nodes , nodes[i].l );
    nodes[i].r = cleanup_expression_tree( nodes , nodes[i].r );
    if( nodes[i].op == BoolExprNode::PAR_OPEN )
    {
      int x = nodes[i].r;
      nodes[i] = BoolExprNode{};
      return x;
    }
    else if( nodes[i].op == BoolExprNode::PAR_CLOSE )
    {
      int x = nodes[i].l;
      nodes[i] = BoolExprNode{};
      return x;
    }
    else
    { 
      return i;
    }
  }

  void ParticleRegionCSG::build_from_expression_string(const ParticleRegion* regions , size_t nb_regions)
  {
    std::map< std::string , int > identMap;
    for(unsigned int i=0;i<nb_regions;i++)
    {
      identMap [ regions[i].name() ] = i;
    }

    std::vector< BoolExprNode > tokens;
    const std::string expr = m_user_expr + " "; // to end with a separator
    std::string ident;
    for(size_t i=0;i<expr.length();i++)
    {
      if( expr[i] == '(' ) tokens.push_back( { BoolExprNode::PAR_OPEN } );
      else if( expr[i] == ')' ) tokens.push_back( { BoolExprNode::PAR_CLOSE } );
      else if( expr[i]==' ' || expr[i]=='\t' || expr[i]=='\n' )
      {
        if( ident == "and" ) tokens.push_back( { BoolExprNode::OP_AND } );
        else if( ident == "or" ) tokens.push_back( { BoolExprNode::OP_OR } );
        else if( ident == "not" ) tokens.push_back( { BoolExprNode::OP_NOT } );
        else if( ! ident.empty() )
        {
          auto it = identMap.find(ident);
          if( it == identMap.end() )
          {
            fatal_error() << "region '"<<ident<<"' not found" << std::endl;
          }
          tokens.push_back( { it->second } );
        }
        ident.clear();
      }
      else
      {
        ident.push_back( expr[i] );
      }
    }
    
    // convert operand indices to local copy indices
    {
      std::map<int,int> region_idx_map;
      m_regions.clear();
      m_regions.reserve( 32 );
      for(auto& tok:tokens)
      {
        if( tok.op >= 0 )
        {
          int id = region_idx_map.size();
          assert( id == int(m_regions.size()) );
          if( region_idx_map.find( tok.op ) == region_idx_map.end() )
          {
            region_idx_map[tok.op] = id;
            m_regions.push_back( regions[tok.op] );
          }
          tok.op = region_idx_map[tok.op];
        }
      }
    }
    if( m_regions.size() > MAX_REGION_OPERANDS )
    {
      fatal_error() << "maximum number of regions used in expression is "<<MAX_REGION_OPERANDS<<std::endl;
    }

//    lout << "build initial tree, expr = "<< m_user_expr << std::endl;    
    int root = build_expression_tree( tokens , 0 , tokens.size() );
//    nand_binary_tree_print(tokens,root);

//    lout << "remove ()" << std::endl;    
    root = cleanup_expression_tree( tokens , root );
//    nand_binary_tree_print(tokens,root);

//    lout << "convert to NAND binary tree" << std::endl;    
    bool neg_result = make_nand_binary_tree( tokens , root );
    if( neg_result )
    {
//      lout << "add final negation root " << root << " -> "<< tokens.size() << std::endl;
      tokens.push_back( { BoolExprNode::OP_AND , root , -1 , true , true } );
      root = tokens.size()-1;
    }
//    nand_binary_tree_print(tokens,root);
    
    int depth = nand_binary_tree_depth( tokens , root );
//    lout << "complete NAND binary tree, depth="<<depth << std::endl;    
    root = nand_binary_tree_fill( tokens , root , depth );
//    nand_binary_tree_print(tokens,root);
    
    // build operand list and copy table
    {
      std::vector<int> operands;
      nand_binary_tree_operands(tokens,operands,root);
      if( operands.size() > MAX_REGION_OPERANDS )
      {
        fatal_error() << "expression exceeds maximum number of operands ("<<operands.size()<<">"<<MAX_REGION_OPERANDS<<")"<<std::endl;
      }
      m_nb_operands = operands.size();
      for(unsigned int i=0;i<m_nb_operands;i++)
      {
        if( operands[i]==-1 ) m_operand_places[i] = m_regions.size(); // region mask is 0 padded, so the first place after the last region is a false constant
        else m_operand_places[i] = operands[i];
      }
/*
      lout << "operands =";
      for(unsigned int i=0;i<m_nb_operands;i++) lout <<" "<<m_operand_places[i];
      lout<<std::endl;
*/
    }
    
    // build operator parameters bitmask
    for(int l=0;l<depth;l++) nand_binary_tree_operators(tokens,m_expr,root,l);
/*
    uint64_t opbits = m_expr;
    int l=depth-1;
    while(l>=0)
    {
      lout << "round "<<depth-1-l<<" : ";
      int nb = 2<<l;
      for(int b=0;b<nb;b++) { lout<<" "<< (opbits&1); opbits = opbits>>1; }
      lout <<std::endl;
      --l;
    }
*/
  }
}


// =========== Unit tests ====================
#include <exanb/core/unit_test.h>
#include <random>

XSTAMP_UNIT_TEST(particle_regions)
{
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -3 , 3 );

  // Sphere test with translation
  const auto R1 = YAML::Load( R"EOF(
R1:
  bounds: [ [ -2 , -2 , -2 ] , [ 2 , 2 , 2 ] ] 
  quadric:
    shape: sphere
)EOF").as<exanb::ParticleRegion>();

  const auto R2 = YAML::Load( R"EOF(
R2:
  bounds: [ [ -2 , -2 , -2 ] , [ 2 , 2 , 2 ] ] 
  quadric:
    shape: sphere
    transform:
      - scale: [ 1.5 , 0.9 , 0.5 ]
      - translate: [ 0.5 , 1.1 , 1.5 ]
)EOF").as<exanb::ParticleRegion>();

  const auto M = YAML::Load( R"EOF(
- scale: [ 1.5 , 0.9 , 0.5 ]
- translate: [ 0.5 , 1.1 , 1.5 ]
)EOF").as<exanb::Mat4d>();
  
  const auto M_inv = inverse(M);

/*
  std::cout << R1.m_name << std::endl;
  std::cout << "\tQuadric = "<< R1.m_quadric << std::endl;
  std::cout << "\tBounds = "<< R1.m_bounds << std::endl;

  std::cout <<"M="<<M<<std::endl;

  std::cout << R2.m_name << std::endl;
  std::cout << "\tQuadric = "<< R2.m_quadric << std::endl;
  std::cout << "\tBounds = "<< R2.m_bounds << std::endl;
*/

  //const Vec3d center = {1,1,1};
  std::vector<Vec3d> points = { {0,0,0} , {1,0,0} , {1.5,0,0} , {2,0,0} , {2.5,0,0} , {3,0,0} };

  for(unsigned int i=0;i<1000;i++)
  {
    Vec3d p3 = { ud(gen) , ud(gen) , ud(gen) };
    if(i<points.size()) p3=points[i];
    const auto P = make_vec4d(p3);
    const auto P_inv = M_inv * P;
    const double l = norm( make_vec3d(P_inv) );
    double f1 = quadric_eval( R1.m_quadric , P_inv ); 
    double f2 = quadric_eval( R2.m_quadric , P );
    const bool inside = (l<=1.0) && is_inside(R1.m_bounds,p3);
    //std::cout << "P="<<P<<", L="<<l<<", in="<<inside<<", R1(M^-1.P)="<<f1<<", R2(P)="<<f2<<std::endl;
    XSTAMP_TEST_ASSERT( std::fabs(f2-f1) <= 1.e-10 );
    XSTAMP_TEST_ASSERT( inside == R2.contains(p3,123) );
  }
}

XSTAMP_UNIT_TEST(particle_region_csg)
{
  using namespace exanb;
  
  auto regions = YAML::Load( R"EOF(
      - ZONE1:  
          quadric:
            shape: conex
            transform:
              - scale: [ 3 , 2.4 , 2.4 ]
              - translate: [ 1 , 0 , 0 ]
      - ZONE2:
          quadric:
            shape: sphere
            transform:
              - scale: [ 1.5 , 1.0 , 0.5 ]
              - yrot: pi*0.25
      - ZONE3:
          bounds: [ [ 1 , 1 , 1 ] , [ 2 , 2 , 2 ] ]
)EOF").as<ParticleRegions>();

  struct Particle { Vec3d r; uint64_t id; };
  std::vector<Particle> particles = {
    { {0,0,0} , 1 } ,
    { {1,0,0} , 2 } ,
    { {0,1,0} , 3 } ,
    { {0,0,1} , 4 } ,
    { {1,0,1} , 5 } ,
    { {0,1,1} , 6 } ,
    { {1,1,1} , 7 } ,
    { {1,1,0} , 8 } 
    };

  std::map<std::string,int> regionMap;
  for(const auto& R:regions)
  {
//    lout << "\t"<< R.m_name << std::endl;
//    lout << "\t\tQuadric = "<< R.m_quadric << std::endl;
//    lout << "\t\tBounds  = "<< R.m_bounds << std::endl;
    int rid = regionMap.size();
//    lout << "\t\tRid     = "<< rid<<std::endl;
    regionMap[R.name()] = rid;
  }
//  lout << "expression = "<<expr<<std::endl;

  ParticleRegionCSG prcsg;
  prcsg.m_user_expr = "ZONE2 and ( ZONE3 or not ZONE1 )" ;
  prcsg.build_from_expression_string( regions.data() , regions.size() );
  ParticleRegionCSGShallowCopy prcsg_sc = prcsg;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud(-3,3);

  for(unsigned int i=0;i<1000;i++)
  {
    Vec3d p = { ud(gen) , ud(gen) , ud(gen) };
    uint64_t id = static_cast<uint64_t>( (ud(gen)+3)*1000 );
    if( i<particles.size() ) { p = particles[i].r; id = particles[i].id; }
//    lout << "particle "<<particles[i].id<<" : r = "<< particles[i].r << std::endl;
    bool ZONE1 = regions[0].contains( p , id );
    bool ZONE2 = regions[1].contains( p , id );
    bool ZONE3 = regions[2].contains( p , id );
    bool expected = ZONE2 && ( ZONE3 || ! ZONE1 );
    bool result = prcsg_sc.contains( p , id );
    XSTAMP_TEST_ASSERT( result == expected );
  }
}

