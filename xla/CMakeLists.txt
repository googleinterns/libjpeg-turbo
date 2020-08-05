# Locate the pb file and define it as a macro for futher use.
set(TF_GRAPH ${PROJECT_SOURCE_DIR}/xla/tfdct.pb)
if(NOT EXISTS ${TF_GRAPH})
  message(FATAL_ERROR "Tensorflow graph definition file not found in ${TF_GRAPH}. Run the python script ${PROJECT_SOURCE_DIR}/xla/pbgenerator.py to generate it.")
endif()
add_definitions(-DPB_BINARY_PATH=\"${TF_GRAPH}\")

# External tensorflow library.
if(NOT TENSORFLOW_LIB)
  # Default location
  set(TENSORFLOW_LIB /usr/local/lib/libtensorflow.so)
endif()

if(NOT EXISTS ${TENSORFLOW_LIB})
  message(FATAL_ERROR "Can't find the tensorflow library in ${TENSORFLOW_LIB}. If it's installed in another directory, specify it as a flag called TENSORFLOW_LIB. If you haven't installed it, you should, otherwise it's impossible to run the dct with xla: https://www.tensorflow.org/install/lang_c.")
endif()

add_library( tensorflow SHARED IMPORTED GLOBAL)
set_target_properties( tensorflow PROPERTIES IMPORTED_LOCATION ${TENSORFLOW_LIB})
include_directories(${CMAKE_SOURCE_DIR}/xla)
add_library(inference SHARED Inference.c)
target_link_libraries(inference tensorflow)

add_library(xla OBJECT jfdctxla.c)
target_link_libraries(xla inference)

add_library(normal-dct OBJECT ../jfdctflt.c)
set_property(TARGET normal-dct PROPERTY COMPILE_FLAGS "-DDCT_FLOAT_SUPPORTED")

add_executable(dctunittest dctunittest.c $<TARGET_OBJECTS:normal-dct>)
target_link_libraries(dctunittest xla)
set_property(TARGET dctunittest PROPERTY COMPILE_FLAGS "-DREL_ERROR_MARGIN=0.001 -DRAND_SEED=19191919")
add_test(basic_dctunittest ./dctunittest)

# Needed for logging in shared library.
if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
  set_target_properties(xla PROPERTIES POSITION_INDEPENDENT_CODE 1)
endif()
