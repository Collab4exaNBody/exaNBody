
    //using onika::cuda::clamp ;
    //using onika::cuda::max ;
    //using onika::cuda::min ;
    using onika::FixedCapacityVector;

    //auto cells = grid.cells();
    const unsigned int n_particles_a = cells[cell_a].size();
    if( n_particles_a == 0 ) return true;

    //IJK dims = grid.dimension();
    //const double cell_size = grid.cell_size();

    const size_t* sub_grid_start = amr.m_sub_grid_start;
    const uint32_t* sub_grid_cells = amr.m_sub_grid_cells;

    //const double max_dist = *nbh_dist;
    const double max_dist2 = max_dist*max_dist;

    const unsigned int loc_max_gap = static_cast<size_t>( std::ceil( max_dist / cell_size ) );
    // ldbg << "cell max gap = "<<loc_max_gap<<", cslog2="<<cs_log2<<std::endl;

    auto & p_a_nbh_status = tmp.p_a_nbh_status;
    auto & cell_a_particle_nbh = tmp.cell_a_particle_nbh;

    const auto* __restrict__ rx_a = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_a);
    const auto* __restrict__ ry_a = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_a);
    const auto* __restrict__ rz_a = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_a);

    ssize_t sgstart_a = sub_grid_start[cell_a];
    ssize_t sgsize_a = sub_grid_start[cell_a+1] - sgstart_a;
    ssize_t n_sub_cells_a = sgsize_a+1;
    ssize_t sgside_a = icbrt( n_sub_cells_a );
    assert( sgside_a == icbrt64( n_sub_cells_a ) );
    
    assert( sgside_a <= static_cast<ssize_t>(GRID_CHUNK_NBH_MAX_AMR_RES) );
    const double subcell_size_a = cell_size / sgside_a;

    // neighbor cells area to scan (ranges for neighbor cell B)
    ssize_t bstarti = onika::cuda::max( loc_a.i-loc_max_gap , 0l ); 
    ssize_t bendi = onika::cuda::min( loc_a.i+loc_max_gap , dims.i-1 );
    ssize_t bstartj = onika::cuda::max( loc_a.j-loc_max_gap , 0l );
    ssize_t bendj = onika::cuda::min( loc_a.j+loc_max_gap , dims.j-1 );
    ssize_t bstartk = onika::cuda::max( loc_a.k-loc_max_gap , 0l );
    ssize_t bendk = onika::cuda::min( loc_a.k+loc_max_gap , dims.k-1 );
      
    // ---------
    
    //for(int ka=0;ka<sgside_a;ka++)
    //for(int ja=0;ja<sgside_a;ja++)
    //for(int ia=0;ia<sgside_a;ia++)
    for(unsigned int sgindex_a=0; sgindex_a<n_sub_cells_a; sgindex_a++)
    {
      //const unsigned int sgindex_a = sg_cell_index( sgside_a , IJK{ia,ja,ka} ); // grid_ijk_to_index( IJK{sgside_a,sgside_a,sgside_a} , IJK{sca_i,sca_j,sca_k} );
      auto [ia,ja,ka] = grid_index_to_ijk( IJK{sgside_a,sgside_a,sgside_a} , sgindex_a );
      
      unsigned int p_start_a = 0;
      unsigned int p_end_a = n_particles_a;
      if( sgindex_a > 0 ) { p_start_a = sub_grid_cells[sgstart_a+sgindex_a-1]; }
      if( sgindex_a < sgsize_a ) { p_end_a = sub_grid_cells[sgstart_a+sgindex_a]; }
      
      if( p_start_a < p_end_a )
      {
        const AABB sc_a_bounds = {
          { ia   *subcell_size_a -max_dist , ja   *subcell_size_a -max_dist , ka   *subcell_size_a -max_dist } ,
          {(ia+1)*subcell_size_a +max_dist ,(ja+1)*subcell_size_a +max_dist ,(ka+1)*subcell_size_a +max_dist } };

        // reset uncompressed scratch buffer status for current sub cell
        tmp.begin_sub_cell( p_start_a, p_end_a );
        // ---------
        
        // TODO: here, neighboring cell B's block can be computed differently for each sub cell of cell A
        // such that number of traversed neighboring cells is reduced (for large cells, most of the time, all neighboring sub cells wil be in cell A itself)
        
        for(ssize_t loc_bk=bstartk;loc_bk<=bendk;loc_bk++)
        for(ssize_t loc_bj=bstartj;loc_bj<=bendj;loc_bj++)
        for(ssize_t loc_bi=bstarti;loc_bi<=bendi;loc_bi++)
        {
          IJK loc_b { loc_bi, loc_bj, loc_bk };
          ssize_t cell_b = grid_ijk_to_index( dims, loc_b );
          size_t n_particles_b = cells[cell_b].size();

          ssize_t sgstart_b = sub_grid_start[cell_b];
          ssize_t sgsize_b = sub_grid_start[cell_b+1] - sgstart_b;
          ssize_t n_sub_cells_b = sgsize_b+1;
          ssize_t sgside_b = icbrt( n_sub_cells_b );
          assert( sgside_b == icbrt64(n_sub_cells_b) ) ;
          
          assert( sgside_b <= static_cast<ssize_t>(GRID_CHUNK_NBH_MAX_AMR_RES) );
          const double subcell_size_b = cell_size / sgside_b;

          const auto* __restrict__ rx_b = cells[cell_b][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_b);
          const auto* __restrict__ ry_b = cells[cell_b][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_b);
          const auto* __restrict__ rz_b = cells[cell_b][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_b);

          const Vec3d rel_pos_a = ( loc_a - loc_b ) * cell_size; // relative position of cell a from cell b

          const IJK rloc_b = loc_b - loc_a; // relative (to cell b) position of cell a in grid
          const uint16_t cell_b_enc = encode_cell_index( rloc_b );
          assert( cell_b_enc != 0 ); // for safety

          // -----

          // bounds of sub-cell of cell A, relative to cell B's origin
          const AABB sca_relb = { sc_a_bounds.bmin + rel_pos_a , sc_a_bounds.bmax + rel_pos_a };

          const int ibs = onika::cuda::clamp( static_cast<int>( std::floor( sca_relb.bmin.x / subcell_size_b ) ) , int(0) , int(sgside_b-1) );
          const int ibe = onika::cuda::clamp( static_cast<int>( std::ceil ( sca_relb.bmax.x / subcell_size_b ) ) , int(0) , int(sgside_b-1) );

          const int jbs = onika::cuda::clamp( static_cast<int>( std::floor( sca_relb.bmin.y / subcell_size_b ) ) , int(0) , int(sgside_b-1) );
          const int jbe = onika::cuda::clamp( static_cast<int>( std::ceil ( sca_relb.bmax.y / subcell_size_b ) ) , int(0) , int(sgside_b-1) );

          const int kbs = onika::cuda::clamp( static_cast<int>( std::floor( sca_relb.bmin.z / subcell_size_b ) ) , int(0) , int(sgside_b-1) );
          const int kbe = onika::cuda::clamp( static_cast<int>( std::ceil ( sca_relb.bmax.z / subcell_size_b ) ) , int(0) , int(sgside_b-1) );

          for(int kb=kbs;kb<=kbe;kb++)
          for(int jb=jbs;jb<=jbe;jb++)
          for(int ib=ibs;ib<=ibe;ib++)
          //for(unsigned int sgindex_b=0; sgindex_b<n_sub_cells_b; sgindex_b++)
          {
            const unsigned int sgindex_b = sg_cell_index( sgside_b , IJK{ib,jb,kb} ); // grid_ijk_to_index( IJK{sgside_b,sgside_b,sgside_b} , IJK{scb_i,scb_j,scb_k} );
            //auto [ib,jb,kb] = grid_index_to_ijk( IJK{sgside_b,sgside_b,sgside_b} , sgindex_b );
            
            unsigned int p_start_b = 0;
            unsigned int p_end_b = n_particles_b;
            if( sgindex_b > 0 ) { p_start_b = sub_grid_cells[sgstart_b+sgindex_b-1]; }
            if( sgindex_b < sgsize_b ) { p_end_b = sub_grid_cells[sgstart_b+sgindex_b]; }

            if( p_end_b>p_start_b )
            {
              const AABB sc_b_bounds = {
                { ib   *subcell_size_b , jb   *subcell_size_b , kb   *subcell_size_b } ,
                {(ib+1)*subcell_size_b ,(jb+1)*subcell_size_b ,(kb+1)*subcell_size_b } };

              if( min_distance2_between( sca_relb , sc_b_bounds ) <= max_dist2 )
              {
                if( max_distance2_between( sca_relb , sc_b_bounds ) <= max_dist2 )
                {
                  // ==> BLOCK PARALLEL FOR HERE <==
                  for(unsigned int p_a=p_start_a+ONIKA_CU_THREAD_IDX; p_a<p_end_a; p_a+=ONIKA_CU_BLOCK_SIZE)
                  {
                    for(unsigned int chunk_b=(p_start_b>>cs_log2); chunk_b <= ((p_end_b-1)>>cs_log2) ; chunk_b++)
                    {
                      onika::cuda::pair<uint16_t,uint16_t> nbh = { cell_b_enc , static_cast<uint16_t>(chunk_b) };
                      assert( nbh >= p_a_nbh_status[p_a-p_start_a].last_chunk );
                      if( nbh != p_a_nbh_status[p_a-p_start_a].last_chunk )
                      {
                        bool cell_switch = ( nbh.first != p_a_nbh_status[p_a-p_start_a].last_chunk.first );
                        if( tmp.avail_stream_space(p_a) >= ( 1u + cell_switch * 2u ) )
                        {
                          assert( p_a_nbh_status[p_a-p_start_a].chunk_count_idx != 0 );
                          if( cell_switch ) // switched to a different neighbor cell
                          {
                            ++ cell_a_particle_nbh[p_a][0];
                            cell_a_particle_nbh[p_a].push_back( nbh.first );
                            p_a_nbh_status[p_a-p_start_a].chunk_count_idx = cell_a_particle_nbh[p_a].size();
                            cell_a_particle_nbh[p_a].push_back(0);                
                          }
                          assert( cell_a_particle_nbh[p_a][p_a_nbh_status[p_a-p_start_a].chunk_count_idx] < onika::cuda::numeric_limits<uint16_t>::max );
                          ++ cell_a_particle_nbh[p_a][p_a_nbh_status[p_a-p_start_a].chunk_count_idx];
                          cell_a_particle_nbh[p_a].push_back( nbh.second );
                          p_a_nbh_status[p_a-p_start_a].last_chunk = nbh;
                        }
                        else { p_a_nbh_status[p_a-p_start_a].chunk_count_idx = 0; }
                      }
                    }
                  }
                  ONIKA_CU_BLOCK_FENCE();
                  ONIKA_CU_BLOCK_SYNC();
                }
                else
                {
                  const Vec3d cell_b_position = grid_origin+((grid_offset+loc_b)*cell_size); //grid.cell_position( loc_b ); // => m_origin+((m_offset+loc)*m_cell_size);

                  // ==> BLOCK PARALLEL FOR HERE <==
                  for(unsigned int p_a=p_start_a+ONIKA_CU_THREAD_IDX; p_a<p_end_a; p_a+=ONIKA_CU_BLOCK_SIZE)
                  {
                    assert( /* p_a>=0 && */ p_a<n_particles_a && p_a<cell_a_particle_nbh.size() );
                    const double pa_mind2 = min_distance2_between( Vec3d{rx_a[p_a],ry_a[p_a],rz_a[p_a]} - cell_b_position , sc_b_bounds );
                    if( pa_mind2 <= max_dist2 )
                    {
                      for(unsigned int p_b=p_start_b; p_b<p_end_b; )
                      {
                        const double d2 = norm2( Vec3d{rx_b[p_b],ry_b[p_b],rz_b[p_b]} - Vec3d{rx_a[p_a],ry_a[p_a],rz_a[p_a]} );
                        if( d2 <= max_dist2 )
                        {
                          const unsigned int chunk_b = p_b >> cs_log2;
                          onika::cuda::pair<uint16_t,uint16_t> nbh = { cell_b_enc , static_cast<uint16_t>(chunk_b) };
                          assert( nbh >= p_a_nbh_status[p_a-p_start_a].last_chunk );
                          if( nbh != p_a_nbh_status[p_a-p_start_a].last_chunk )
                          {
                            bool cell_switch = ( nbh.first != p_a_nbh_status[p_a-p_start_a].last_chunk.first );
                            if( tmp.avail_stream_space(p_a) >= ( 1u + cell_switch * 2u ) )
                            {
                              assert( p_a_nbh_status[p_a-p_start_a].chunk_count_idx != 0 );
                              if( cell_switch ) // switched to a different neighbor cell
                              {
                                ++ cell_a_particle_nbh[p_a][0];
                                cell_a_particle_nbh[p_a].push_back( nbh.first );
                                p_a_nbh_status[p_a-p_start_a].chunk_count_idx = cell_a_particle_nbh[p_a].size();
                                cell_a_particle_nbh[p_a].push_back(0);                
                              }
                              assert( cell_a_particle_nbh[p_a][p_a_nbh_status[p_a-p_start_a].chunk_count_idx] < onika::cuda::numeric_limits<uint16_t>::max );
                              ++ cell_a_particle_nbh[p_a][p_a_nbh_status[p_a-p_start_a].chunk_count_idx];
                              cell_a_particle_nbh[p_a].push_back( nbh.second );
                              p_a_nbh_status[p_a-p_start_a].last_chunk = nbh;
                            }
                            else { p_a_nbh_status[p_a-p_start_a].chunk_count_idx = 0; }
                          }
                          p_b = (chunk_b+1) << cs_log2;
                        }
                        else { ++ p_b; }
                      }
                      
                    } // if pa_mind2 <= max_dist2

                  } // for p_a ...
                  ONIKA_CU_BLOCK_FENCE();
                  ONIKA_CU_BLOCK_SYNC();
                
                } // else ( if max_distance2_between <= max_dist2 )
              
              } // if min_distance2_between <= max_dist2
              
            } // if mind2 <= max_dist2
            
          } // for ib ...
                        
        } // for cell B loc

        // detect out of temporary memory and return failure code
        ONIKA_CU_BLOCK_SHARED unsigned int encoding_failed;
        if( ONIKA_CU_THREAD_IDX == 0 ) { encoding_failed = false; }
        ONIKA_CU_BLOCK_SYNC();
        for(unsigned int p_a=p_start_a+ONIKA_CU_THREAD_IDX; p_a<p_end_a; p_a+=ONIKA_CU_BLOCK_SIZE)
        {
          if( p_a_nbh_status[p_a-p_start_a].chunk_count_idx == 0 ) { encoding_failed = true; }
        }
        ONIKA_CU_BLOCK_SYNC();
        
        // on temporary scratch buffer overflow, return failure code
        if( encoding_failed ) return false;

        tmp.end_sub_cell( p_start_a, p_end_a );
      } // if p_a_start < p_a_end
        
    } // for ia 


    /************ final step, copy streams to global output ***********/
    
    ONIKA_CU_BLOCK_SHARED size_t cell_particles_nbh_stream_size;
    ONIKA_CU_BLOCK_SHARED size_t stream_index_size;
    ONIKA_CU_BLOCK_SHARED uint16_t * target_ccnbh;
    if( ONIKA_CU_THREAD_IDX == 0 )
    {
      cell_particles_nbh_stream_size = 0;
      for(size_t p_a=0;p_a<n_particles_a;p_a++)
      {
        cell_particles_nbh_stream_size += cell_a_particle_nbh[p_a].size();
      }
      stream_index_size = config.build_particle_offset ? ( n_particles_a * 2 ) : 0;
      target_ccnbh = chunk_neighbors.allocate( cell_a , stream_index_size + cell_particles_nbh_stream_size );

      if( target_ccnbh!=nullptr && config.build_particle_offset )
      {
        //assert( target_ccnbh.size() >= n_particles_a * 2 );
        size_t stream_pos = stream_index_size;
//        for(size_t p_a=ONIKA_CU_THREAD_IDX;p_a<n_particles_a;p_a+=ONIKA_CU_BLOCK_SIZE) // FALSE because inside " if(ONIKA_CU_THREAD_IDX==0){ ... } "
        for(size_t p_a=0;p_a<n_particles_a;++p_a)
        {
          //assert( stream_pos == p_a_nbh_status[p_a].chunk_count_idx + stream_index_size );
          uint32_t offset = stream_pos - (n_particles_a*2) + 1;
          target_ccnbh[p_a*2+0] = offset ;
          target_ccnbh[p_a*2+1] = offset >> 16 ;
          assert( reinterpret_cast<const uint32_t*>( target_ccnbh /*.data()*/ )[p_a] == offset );
          assert( p_a!=0 || ( target_ccnbh[0]==1 && target_ccnbh[1]==0 ) );
          stream_pos += cell_a_particle_nbh[p_a].size();
        }
        assert( stream_pos == stream_index_size + cell_particles_nbh_stream_size );
      }
    }
    ONIKA_CU_BLOCK_FENCE();
    ONIKA_CU_BLOCK_SYNC();

    // return failure code if output stream allocation failed
    if( target_ccnbh == nullptr ) return false;

    // copy partial encoded stream for particle p_a
    const uint16_t* cpy_src = nullptr;
    uint16_t* cpy_dst = target_ccnbh + stream_index_size;
    size_t cpy_len = 0;
    for(size_t p_a=0;p_a<n_particles_a;p_a++)
    {
      const uint16_t* cur_src = cell_a_particle_nbh[p_a].data();
      const size_t cur_len = cell_a_particle_nbh[p_a].size();
      if( cur_src == cpy_src+cpy_len )
      {
        cpy_len += cur_len;
      }
      else
      {
        //if(cpy_len>0) onika::cuda::cu_block_memcpy( cpy_dst , cpy_src , cpy_len * sizeof(uint16_t) );
        cpy_dst += cpy_len;
        cpy_src = cur_src;
        cpy_len = cur_len;
      }
    }
    //if(cpy_len>0) onika::cuda::cu_block_memcpy( cpy_dst , cpy_src , cpy_len * sizeof(uint16_t) );
    
    // for(size_t p_a=0;p_a<n_particles_a;p_a++)
    for(size_t p_a=ONIKA_CU_THREAD_IDX;p_a<n_particles_a;p_a+=ONIKA_CU_BLOCK_SIZE)
    {
      cell_a_particle_nbh[p_a].clear();
    }
    ONIKA_CU_BLOCK_FENCE();
    ONIKA_CU_BLOCK_SYNC();
    // **********************************************************

    assert( ( cpy_dst + cpy_len ) == ( target_ccnbh + stream_index_size + cell_particles_nbh_stream_size ) );
    
    return true;

