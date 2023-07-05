# ExaNBody : The documentation file !!!

Welcome to the exaNBody documentation page, your comprehensive guide to mastering the art of N-Body simulation. This documentation provides a detailed overview of exaNBody, a powerful code built to simulate the interactions and dynamics of multiple celestial bodies. Whether you're an physician enthusiast or a seasoned researcher, this documentation will equip you with the knowledge and tools to harness the full potential of exaNBody in accurately modeling complex N-Body systems.


## How to cite it :


To properly cite exaNBody in your research or documentation, please follow the provided citation guidelines.

```
TODO : Add link and bibtex here
```

## General purpose of exaNBody

ExaNBody is a software framework developed at CEA for building numerical simulation codes based on the N-Body problem in various physics domains.
Originally designed as part of exaStamp, a molecular dynamics simulation code for atomic-scale material's behavior,
exaNBody has achieved nearly linear speed-ups on thousands of cores (100,000+) for simulations involving highly heterogeneous density scenarios,
such as droplet splashing or micro-jetting, with half a trillion atoms.

To achieve such scalability, high-performance data structures and algorithms, including compact and fast neighbor lists as well as optimized, memory-efficient MPI communications, have been developed.
These advancements have made the data structures and algorithms increasingly generic and customizable to accommodate a growing number of physical models,
while also becoming \gpu-aware and compatible with supercomputers featuring accelerators.
As a result, the authors decided to extract the N-Body core, exaNBody, as a standalone project, making it available for other N-Body simulation codes.

Today, exaNBody continues to evolve alongside exaStamp, focusing on flexibility, performance, and portability,
using industry standards as its software stack basis and providing application-level customizability.

ExaNBody encompasses various aspects of constructing N-Body simulation codes.
Firstly, it provides a high level of flexibility through a component-based programming model.
These components serve as building blocks and are assembled in YAML format files, with added features such as document inclusion and merging support.
Secondly, it offers portable performance by providing developers with a collection of algorithms and programming interfaces specifically designed for common N-Body compute kernels.
Additionally, the programming tools mentioned above are natively compatible with different CPU architectures and GPU devices, thanks to the Onika execution support library.

## How to install it

To correctly install exaNBody using CMake, please follow the provided set of installation commands.

```
git clone pathto/exaNBody.git
mkdir build-exaNBody/ && cd build-exaNBody/
cmake ../exaNBody/ -DCMAKE_INSTALL_PREFIX=path_to_install
make install
export exaNBody_DIR=path_to_install
```

To successfully build your application using CMake, please refer to the following set of commands for the installation process.

```
export exaNBody_DIR=path_to_install
git clone pathto/myApp.git
mkdir build-myApp && cd build-myApp
cmake ../myApp/
make -j 16
```

Packages required: 
```
spack load yaml-cpp
```

## Data structure storages

Particle data layout and auxiliary data structures are essential to reach maximum performance at the NUMA node level.
Particle data, organized independently for each cell, are stored in a container, designed to match the targeted hardware characteristics (such as NUMA node setup or vectorization units) and to allow for a suitable memory access pattern in calculation kernels.

To this end, particle fields are stored using Onika's Structure Of Array template library to support the use of CPU's SIMD instructions or GPU thread blocks.
It features proper alignment and vector friendly padding for all fields, depending on underlying hardware (CPU or GPU).
Additionally, it has a low memory footprint, taking only 16 bytes overhead per cell regardless of the number of particle fields, making it possible to store very large and sparse domains.
Access to particle fields is tightly coupled with parallelization templates, adding no burden to the developer to handle it.

In N-Body simulation, another essential feature is the search for particles' neighbors. Given that, the computational domain is organized as a cartesian grid of cells, each of which contains a set of particles, the neighbors search on each process usually takes advantage of this grid structure to speedup search by mixing the linked cell method and the Verlet method.
However, depending on the type of N-body simulation, particles may either move extremely rapidly or remain close to their initial positions, and their distribution can be heterogeneously dense or merely homogeneous.
Those two factors heavily impact neighbors search performance: neighbor lists are refreshed more frequently when particles move rapidly and they have larger memory footprint as particle density increases.
On the one hand, exaNBody takes advantage of an Adaptative Mesh Refinement (AMR grid using dynamic cell refinement algorithms (cells refinement is updated as particles move around) to accelerate (frequent) neighbor list reconstructions.

On the other hand, a novel compressed neighbor list data structure can be up to 9x more compact than a naive storage (one cell index and one particle index per neighbor).
The AMR and neighbor list data structures have a design that can be tuned for hardware, but also ensures fast access from both the CPU and the GPU.

### Define your grid of particles:

Example, the field of a sphere: 

```
XSTAMP_DECLARE_FIELD(uint64_t ,id      ,"particle id");
XSTAMP_DECLARE_FIELD(double   ,rx      ,"particle position X");
XSTAMP_DECLARE_FIELD(double   ,ry      ,"particle position Y");
XSTAMP_DECLARE_FIELD(double   ,rz      ,"particle position Z");
XSTAMP_DECLARE_FIELD(double   ,radius  ,"radius");
XSTAMP_DECLARE_FIELD(double   ,vx      ,"particle velocity X");
XSTAMP_DECLARE_FIELD(double   ,vy      ,"particle velocity Y");
XSTAMP_DECLARE_FIELD(double   ,vz      ,"particle velocity Z");
XSTAMP_DECLARE_FIELD(double   ,ax      ,"particle acceleration X");
XSTAMP_DECLARE_FIELD(double   ,ay      ,"particle acceleration Y");
XSTAMP_DECLARE_FIELD(double   ,az      ,"particle acceleration Z");
// aliases
XSTAMP_DECLARE_ALIAS( fx, ax )
XSTAMP_DECLARE_ALIAS( fy, ay )
XSTAMP_DECLARE_ALIAS( fz, az )
```

Example, define your grid of spheres:

```
namespace exanb
{
  // DEM model field set
  using SpheresFieldSet= FieldSet<
	field::_radius, field::_vx,field::_vy,field::_vz, field::_fx,field::_fy,field::_fz>;

  // the standard set of FieldSet
  // use FieldSetsWith<fields...> (at the bottom of this file) to select a subset depending on required fields
  using StandardFieldSets = FieldSets< SpheresFieldSet >;

}
```

Please note that the fields rx, ry, and rz are implicitly added to the simulation.

## Adding Developer Documentation: A Step-by-Step Guid

Every operator should provide two essential documentation components:
- A general description can provided by overriding ` void documentation() final` function int the operator class.
- Additionally, a description per slot can be added using the `ADD_SLOT(...,DocString{"your_text"});`

To display a list of operators from the command line, use the following command: ./myApplication --help plugins. Similarly, individual operator information can be accessed using the command line: ./myApplication --plugin operator name.


Example of a general operator description:

```
inline std::string documentation() const override final
{
	return R"EOF(
		This operator does something.
		)EOF";
}
```

Example of a documented slot:
```
ADD_SLOT( TYPE , variable , INPUT , default_value  , DocString{"Description of my slot"} );
```

Example of an output: `./exaDEM --help gravity_force` 

```
==============================
============ help ============
==============================

+ exadem_force_fieldPlugin
Operator    : gravity_force

Description : 
        This operator computes forces related to the gravity.
        
Slots       : 
  gravity
    flow IN
    type Vec3d
    desc define the gravity constant in function of the gravity axis, default value are x axis = 0, y axis = 0 and z axis = -9.807
  grid
    flow IN/OUT
    type Grid<onika::soatl::FieldIds<vx,vy,vz,ax,ay,az,mass,homothety,radius,orient,mom,vrot,arot,inertia,id,shape,friction,type>>
    desc 
```

## Description of Onika : Low level hardware interface

Onika is the low-level hardware interface exaNBody that powers software building blocks exposed to developers.
It features a hybrid execution abstraction layer, Structure Of Array templates (used for particle data) and unified memory management tools and allocators.
Its hybrid execution layer brings functions and building blocks compatible both with CPU and GPU worlds, making it easier to re-use the same code portions in either OpenMP or CUDA execution contexts.
Its generic Structure Of Array Template Library (soatl) containers are powering memory efficient particle fields at exaNBody level.
It allows a very low memory footprint per cell storage of particle attributes while guaranteeing alignment and vector friendly padding for both CPU and GPU.
Low memory footprint means that an empty cell never uses more than 16 bytes of memory, independently of the fieldset used.
Finally, the memory management layer offers C++'s STL compatible allocators to ease the management of data structures that are transferred back and forth between the CPU and the GPU.

## Parallelization strategies

### General description:

_[TODO REWRITE IT]_

#### Message Passing Interface (MPI) or inter-nodes parallelization

Spatial domain decomposition and inter-process communications are critical to ensure scalability at large scales. Indeed, the coarsest parallelization level can become the main bottleneck due to network latencies and load imbalance issues.
To take advantage of this first level of parallelization, the simulation domain is divided into subdomains using an \rcb (Recursive Coordinate Bisection), assigning one subdomain to each \mpi process.
Those features are powered by different software components among which: cell cost estimator, \rcb domain decomposition, and particle migration algorithm.

Particle migration algorithm can be used as-is by any N-Body application, thanks to the underlying generic particle data storage.It has been designed to handle heavily multi-threaded large scale simulations and specifically adjusted to limit memory usage peaks during this communication phase. Additionally, the migration algorithm is also customizable to adapt to specific application needs, keeping the core implementation unchanged. Finally, ghost particle updates are also handled and available to any N-Body application, via a set of readily available components.
 
#### OpenMP / Cuda or intra-node parallelization

Intra-node parallelization API is available in exaNBody to help developers express parallel computations within a MPI process.

This API is offered as a set of parallelization templates associated with three types of computation kernels:
- Local calculations on a particle
- Calculations coupled with reduction across all particles
- Calculations involving one particle and its neighbors.

When a developer injects a compute function into these templates, computation may be routed to CPU or GPU. 
Thread parallelization on the CPU is powered by OpenMP, assuming that each cell represents a work unit.
In case of GPU accelerators are present, the same provided function is re-used in generic CUDA kernels.

The main difference betwwen the GPU and thread parallelization is that each cell is a base work unit for blocks of threads, not an individual thread,
and particles of one cell are processed concurrently by the different threads of a CUDA block. Those two parallelization levels (multi-core and GPU) are easily accessible to developers thanks to the execution support layer of Onika.


### Tune your run with OpenMP

Liste of command lines :

| Type of tools | Command line | Description | Default |
|---------------|--------------|-------------|---------|
| Pine OMP Threads | --pinethreads true | This option pines your OMP threads with a posix thread affinity | false |
| Set the number of threads | --omp_num_threads 10 | Set the number of threads desired | By default it takes the maximum number of threads available |
| TODO | --omp_max_nesting 1 | TODO |  -1|
| TODO | --omp_nested true | TODO | false |

## Apply a functor on your grid

- The common approach to build an operator involves applying your functor over cells or particles by using the `compute_cell_particle` or `compute_reduce features`. These features require defining the fields to be used in the computation.
- Alternatively, you can apply your functor to a subset of cells using the `compute_cellist_particle` feature. This feature is particularly useful for packing and unpacking data in a buffer on the GPU during the `update_ghost operator`.

Please ensure that you appropriately define the required fields and tailor the documentation to your specific implementation and use case.

Example with the functor gravity: 
```
using ComputeFields = FieldSet< field::_mass, field::_fx ,field::_fy ,field::_fz >;
inline void execute () override final
{
  GravityForceFunctor func {*gravity}; // gravity is a slot
  compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
}
```

## Build neighbor lists

- To construct "blocks of Verlet lists," you need to define the chunk size. Please specify the desired chunk size to create these blocks.
- Warning: When using the Intel compiler, a chunk size of 1 may lead to failure. Please avoid setting the chunk size to 1 when using the Intel compiler.
- Warning [GPU]: Please note that neighbor lists are built exclusively on the CPU, not on the GPU.

Please ensure that you set an appropriate chunk size based on your requirements, and take into account the provided warnings regarding the Intel compiler and GPU implementation limitations.

### Display details

Example of output : 

```
===== Chunk Neighbors stats =====
Chunk size             = 2
Particles              = 1000
Nbh cells (tot./avg)   = 1000 / 1.00
Neighbors (chunk/<d)   = 1000 / 1000
<d / chunk ratio       = 1.00 , storage eff. = 2.00
Avg nbh (chunk/<d)     = 1.00 / 1.00
min [chunk;<d] / Max [chunk;<d] = [1;1] / [1;1]
distance : min / max   = 1.05 / 1.05
=================================
```

## Performance tools in exaNBody

ExaNBody offers a range of performance tools designed to facilitate the profiling of your parallel application. These tools are specifically developed to help you analyze and optimize the performance of your ExaNBody-based simulations.

### Summary

| Type of tools | Command line | Operator | Description |
|---------------|--------------|----------| ------------|
| Timers        | --profiling-summary true  | profiling : {summary: true }| This tool Displays timer informtaions for every operators. |
| VITE Trace    | --profilingtrace-file true |----------| This tool generates a VITE trace on CPU.  |
| Memory footprint | --TODO    |----------| This tool displays the memory footprint of every data storage used during the execution.  |
| nvtx instructions  | default | default | Instructions nvtxtoolpush and nvtxtoolpop are included around every operator->execute() |
| performance adviser | X | - performance_adviser: { verbose: true } | This tool displays some tips according to your simulation (fit cell size, your number of MPI processes ...) |


### Descriptions

#### Timers

This tool provides a hierarchical list of timers for each operator, offering valuable insights into the performance of your application. The provided information includes:

- Number of calls made to each operator
- CPU time consumed by each operator
- GPU time consumed by each operator
- Imbalance time between MPI processes, including average and maximum values
- Execution time ratio, allowing you to assess the distribution of execution time across operators

By utilizing this tool, you can gain a comprehensive understanding of the timing characteristics of your operators and make informed optimizations to enhance the overall performance of your ExaNBody simulation.


The imbalance value is calculated using the following formula:
```
I = (T_max - T_ave)/T_ave - 1 
```


Here are the variables involved in the calculation:

- `T_max` represents the execution time of the slowest MPI process.
- `T_ave` corresponds to the average time spent across all MPI processes.
- `I` denotes the computed imbalance value.

By evaluating the imbalance value, you can quantify the performance discrepancy among MPI processes, allowing you to identify and address potential load imbalances within your ExaNBody simulation.

Please note that if you forcefully stop your simulation, the timers will be automatically printed in your terminal. This feature provides convenient access to the timing information, allowing you to analyze the execution times of different components even if the simulation is prematurely halted.

Output with OpenMP: 

```
Profiling .........................................  tot. time  ( GPU )   avginb  maxinb     count  percent
sim ...............................................  2.967e+04            0.000   0.000         1  100.00%
... // logs
  loop ............................................  2.964e+04            0.000   0.000         1  99.88%
    scheme ........................................  2.881e+04            0.000   0.000    100000  97.09%
      combined_compute_prolog .....................  2.300e+03            0.000   0.000    100000   7.75%
      check_and_update_particles ..................  1.016e+04            0.000   0.000    100000  34.25%
        particle_displ_over .......................  2.154e+03            0.000   0.000    100000   7.26%
        update_particles_full .....................  6.482e+03            0.000   0.000      5961  21.84%
          update_particles_full_body ..............  6.474e+03            0.000   0.000      5961  21.82%
            compact_neighbor_friction .............  1.621e+02            0.000   0.000      5961   0.55%
            move_particles_friction ...............  6.347e+02            0.000   0.000      5961   2.14%
            trigger_load_balance ..................  2.591e+02            0.000   0.000      5961   0.87%
              trigger_lb_tmp ......................  6.095e+00            0.000   0.000      5961   0.02%
                nth_timestep ......................  3.342e+00            0.000   0.000      5961   0.01%
              extend_domain .......................  2.389e+02            0.000   0.000      5961   0.80%
...
```

Output with MPI:
```
Profiling .........................................  tot. time  ( GPU )   avginb  maxinb     count  percent
sim ...............................................  2.376e+04            0.000   0.000         1  100.00%
...
  loop ............................................  2.372e+04            0.000   0.000         1  99.82%
    scheme ........................................  2.308e+04            0.086   2.249    100000  97.13%
      combined_compute_prolog .....................  5.779e+02            0.280   2.937    100000   2.43%
      check_and_update_particles ..................  1.687e+04            0.454   2.770    100000  70.97%
        particle_displ_over .......................  4.067e+03            0.687   2.643    100000  17.11%
        update_particles_full .....................  1.159e+04            0.167   0.812      6001  48.78%
          update_particles_full_body ..............  1.159e+04            0.167   0.813      6001  48.76%
            compact_neighbor_friction .............  7.170e+01            0.387   0.876      6001   0.30%
            move_particles_friction ...............  1.797e+02            0.254   0.853      6001   0.76%
            trigger_load_balance ..................  9.340e+01            0.674   1.787      6001   0.39%
              trigger_lb_tmp ......................  2.582e+00            0.187   2.836      6001   0.01%
                nth_timestep
              extend_domain .......................  8.655e+01            0.733   2.016      6001   0.36%
...
```


#### VITE trace generator

VITE description: 

```VITE is a simulation framework that aims to provide a high-performance and scalable environment for conducting large-scale scientific simulations. It offers a flexible and modular architecture that enables the simulation of diverse systems, such as molecular dynamics, astrophysics, and fluid dynamics. VITE leverages parallel computing capabilities to efficiently distribute the computational workload across multiple processors or computing nodes. With its focus on performance, scalability, and extensibility, VITE serves as a powerful tool for researchers and scientists in various domains to simulate and analyze complex phenomena and phenomena in a computationally efficient manner.```


- Warning: This operation should not be executed on the GPU.

#### Memory footprint

- TODO

## Debug features in exaNBody

ExaNBody provides several debug features to help developpers. This is an exhaustive list:

| Type of tools | Command line | Architecture | Description | 
|---------------|--------------|--------------|-------------|
| Cuda threads size | TODO | GPU | Set the number of cuda threads to 1 on GPU.|
| Output ldbg   | --logging-debug true | CPU | Print debug logs added in `ldbg <<` |

### Output ldbg

- Possiblity to active it only for one operator: 
	- Command line : `--logging-debug true --debug-filter[".*operator1",".*operator2",...]`
	- Operator name : logging and debug

Example in your input file (.msp):
```
configuration:
  logging: { debug: false , parallel: true }
  debug:
    filter: [ ".*init_neighbor_friction" , ".*move_particles_friction" , ".*check_nbh_friction" , ".*compact_neighbor_friction" , ".*extend_domain" ]
``` 
