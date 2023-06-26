message("Analysing ${BINARY_FILE} ...")

execute_process(COMMAND ${SOATL_OBJDUMP} -D ${BINARY_FILE} OUTPUT_FILE ${BINARY_FILE}.asm OUTPUT_QUIET ERROR_QUIET)

file(STRINGS ${BINARY_FILE}.asm BINARRY_ASSEMBLY)

set(INSLIST vmov vmovapd vmovaps vmovupd vmovups sqrt vsqrtpd vrsqrtpd vsqrtps vrsqrtps vsqrtsd vrsqrtsd vsqrtss vrsqrtss vfmadd132sd vfmadd132pd vfmadd132ss vfmadd132ps)

set(mova vmovapd vmovaps)
set(movu vmovupd vmovups)
set(sqrtp vsqrtpd vrsqrtpd vsqrtps vrsqrtps)
set(sqrts vsqrtsd vrsqrtsd vsqrtss vrsqrtss)
set(fmas vfmadd132sd vfmadd132ss)
set(fmap vfmadd132pd vfmadd132ps)
set(fma vfmadd132sd vfmadd132ss vfmadd132pd vfmadd132ps)
set(INSSUM mova movu sqrtp sqrts fmas fmap)

set(INSREPORT mova movu sqrtp sqrts fmap fmas)

foreach(ins ${INSLIST})
  set(${ins} 0)
endforeach()

foreach(line ${BINARRY_ASSEMBLY})
  foreach(ins ${INSLIST})
    if(${line} MATCHES ".*${ins}.*")
      math(EXPR ${ins} ${${ins}}+1)
    endif()
  endforeach()
endforeach()

foreach(tosum ${INSSUM})
  set(tmplist ${${tosum}})
  set(${tosum} 0)
  foreach(ins ${tmplist})
    math(EXPR ${tosum} ${${tosum}}+${${ins}})
  endforeach()
endforeach()

foreach(ins ${INSREPORT})
  message("${ins} ${${ins}}")
endforeach()


