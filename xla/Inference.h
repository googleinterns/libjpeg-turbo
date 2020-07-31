#include <tensorflow/c/c_api.h>
typedef struct Inference {
  TF_Graph*   graph;
  TF_Session* session;
  TF_Operation* input_op;
  TF_Output* input;
  TF_Operation* output_op;
  TF_Output* output;
  TF_Status* status;
  TF_Tensor* output_tensor;
} Inference;

/**
 * Load a protobuf buffer from disk,
 * recreate the tensorflow graph and
 * provide it for inference.
 */
TF_Buffer* ReadBinaryProto(const char* fname);

/**
 * Tensorflow does not throw errors but manages runtime information
 *   in a _Status_ object containing error codes and a failure message.
 *
 * AssertOk throws a runtime_error if Tensorflow communicates an
 *   exceptional status.
 *
 */
void AssertOk(const TF_Status* status);

/**
 * binary_graphdef_protobuf_filename: only binary protobuffers
 *   seem to be supported via the tensorflow C api.
 * input_node_name: the name of the node that should be feed with the
 *   input tensor
 * output_node_name: the node from which the output tensor should be
 *   retrieved
 */
Inference* newInference(const char* binary_graphdef_protobuf_filename,
    const char* input_node_name,
    const char* output_node_name);

/**
 * Clean up all pointer-members using the dedicated tensorflor api functions
 */
void destroy(Inference* inf);

/**
 * Run the graph on some input data.
 *
 * Provide the input and output tensor.
 */
TF_Tensor* runGraph(const Inference* inf, TF_Tensor* input_tensor);

