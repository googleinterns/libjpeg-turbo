# External tensorflow library.
add_definitions(-DPB_BINARY_PATH=\"${PROJECT_SOURCE_DIR}/xla/tfdct.pb\")
add_library( tensorflow SHARED IMPORTED GLOBAL)
set_target_properties( tensorflow PROPERTIES IMPORTED_LOCATION /usr/local/lib/libtensorflow.so)
include_directories(${CMAKE_SOURCE_DIR}/xla)
add_library(inference SHARED Inference.c)
target_link_libraries(inference tensorflow)

add_library(xla OBJECT jfdctxla.c)
target_link_libraries(xla inference)

add_library(normal-dct OBJECT ../jfdctflt.c)
set_property(TARGET normal-dct PROPERTY COMPILE_FLAGS "-DDCT_FLOAT_SUPPORTED")

add_executable(dctunittest dctunittest.c $<TARGET_OBJECTS:normal-dct>)
target_link_libraries(dctunittest xla)
set_property(TARGET dctunittest PROPERTY COMPILE_FLAGS "-DREL_ERROR_MARGIN=0.001 -DTEST_COUNT=10 -DRAND_SEED=19191919")
add_test(basic_dctunittest ./dctunittest)

# Needed for logging in shared library.
if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
  set_target_properties(xla PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()
