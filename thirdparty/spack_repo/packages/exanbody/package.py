from spack import *

class Exanbody(CMakePackage):
    """ExaNBody is a N-Body framework.
    """

    homepage = "https://github.com/Collab4exaNBody/exaNBody"
    git = "https://github.com/Collab4exaNBody/exaNBody.git"


    version("main", branch="main")

    variant("cuda", default=False, description="Support for GPU")
    depends_on("cmake@3.27.9")
    depends_on("yaml-cpp@0.6.3")
    depends_on("openmpi")
    depends_on("cuda", when="+cuda")
#    build_system("cmake", "autotools", default="cmake")
    default_build_system = "cmake"
    build_system("cmake", default="cmake")

    variant(
        "build_type",
        default="Release", 
        values=("Release", "Debug", "RelWithDebInfo"),
        description="CMake build type",
        )

    def cmake_args(self):
      args = [ self.define_from_variant("-DXNB_BUILD_CUDA=ON", "cuda"), ]
      return args
