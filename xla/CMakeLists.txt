set(XLA_SOURCES jfdctxla.c)
add_library(xla OBJECT ${XLA_SOURCES})

# Needed for logging in shared library.
if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
  set_target_properties(xla PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()
