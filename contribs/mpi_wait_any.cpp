/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  int nprocs = 1;
  int rank = 0;
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&rank);

  std::cout << "P"<<rank<<" nprocs="<<nprocs<<std::endl;
  
  int buffer_size[nprocs];
  for(int p=0;p<nprocs;p++)
  {
    buffer_size[p] = 4709 + ( p*523 ) % 92;
  }

  MPI_Request requests[ 2 ] = { MPI_REQUEST_NULL , MPI_REQUEST_NULL };

  int recv_partner = (rank+1)%nprocs;
  int send_partner = (rank+nprocs-1)%nprocs;

  int send_size = buffer_size[rank];
  int recv_size = buffer_size[recv_partner];

  char* send_buffer = new char[ send_size ] ;
  char* recv_buffer = new char[ recv_size ] ;

  for(int i=0;i<send_size;i++) send_buffer[i]='X';
  for(int i=0;i<recv_size;i++) recv_buffer[i]='\0';
  
  std::cout << "P"<<rank<<" async recv from "<<recv_partner<<" size="<<recv_size<<std::endl;  
  MPI_Irecv( recv_buffer , recv_size , MPI_CHAR, recv_partner , 0, comm, & requests[0] );
  
  std::cout << "P"<<rank<<" async send to "<<send_partner<<" size="<<send_size<<std::endl;  
  MPI_Isend( send_buffer , send_size , MPI_CHAR, send_partner , 0, comm, & requests[1] );

  int reqidx = -1;
  MPI_Waitany( 2 , requests , &reqidx , MPI_STATUS_IGNORE );
  std::cout<<"P"<<rank<<" 1st completed req is "<<reqidx<<std::endl;

  reqidx = -1;
  MPI_Waitany( 2 , requests , &reqidx , MPI_STATUS_IGNORE );
  std::cout<<"P"<<rank<<" 2nd completed req is "<<reqidx<<std::endl;
  
  MPI_Finalize();
}

