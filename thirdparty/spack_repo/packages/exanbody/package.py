from spack import *

class Exanbody(CMakePackage):
    """ExaNBody is a N-Body framework.
    """

    homepage = "https://github.com/Collab4exaNBody/exaNBody"
    git = "https://github.com/Collab4exaNBody/exaNBody.git"


    version("master", branch="main")
    version("main", commit="ae61aa74c5e5bce34a8124e7d46e0d6aa0d4b097")

    depends_on("cmake")
    variant("cuda", default=False, description="Support for GPU")
    depends_on("yaml-cpp")
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
