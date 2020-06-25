set(XLA_SOURCES jfdctxla.c)
add_library(xla OBJECT ${XLA_SOURCES})

add_library(normal-dct OBJECT ../jfdctflt.c)
set_property(TARGET normal-dct PROPERTY COMPILE_FLAGS "-DDCT_FLOAT_SUPPORTED")

add_executable(dctunittest dctunittest.c $<TARGET_OBJECTS:normal-dct> $<TARGET_OBJECTS:xla>)
set_property(TARGET dctunittest PROPERTY COMPILE_FLAGS "-DREL_ERROR_MARGIN=0.001 -DTEST_COUNT=10")
add_test(basic_dctunittest ./dctunittest)

# Needed for logging in shared library.
if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
  set_target_properties(xla PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()
