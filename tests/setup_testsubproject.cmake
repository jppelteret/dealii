FIND_PACKAGE(deal.II 8.0 REQUIRED HINTS ${DEAL_II_DIR})

SET(CMAKE_BUILD_TYPE DebugRelease CACHE STRING "" FORCE)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

#
# Silence warnings:
#
FOREACH(_var
  DIFF_DIR NUMDIFF_DIR TEST_DIFF TEST_PICKUP_REGEX TEST_TIME_LIMIT MPIEXEC
  MPIEXEC_NUMPROC_FLAG MPIEXEC_PREFLAGS MPIEXEC_POSTFLAGS
  )
  IF(DEFINED ${_var})
    SET(_bogus "${${_var}}")
  ENDIF()
ENDFOREACH()

#
# A custom target that does absolutely nothing. It is used in the main
# project to trigger a "make rebuild_cache" if necessary.
#
ADD_CUSTOM_TARGET(regenerate)
