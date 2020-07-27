#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"
#include "Inference.h"
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/**
 * Read a binary protobuf (.pb) buffer into a TF_Buffer object.
 *
 * Non-binary protobuffers are not supported by the C api.
 * The caller is responsible for freeing the returned TF_Buffer.
 */
TF_Buffer* ReadBinaryProto(const char* filename)
{
  int fd = open(filename, 0);
  if (fd < 0) {
    perror("failed to open file: ");
    return NULL;
  }
  struct stat stat;
  if (fstat(fd, &stat) != 0) {
    perror("failed to read file: ");
    return NULL;
  }
  char* data = (char*)malloc(stat.st_size);
  ssize_t nread = read(fd, data, stat.st_size);
  if (nread < 0) {
    perror("failed to read file: ");
    free(data);
    return NULL;
  }
  if (nread != stat.st_size) {
    fprintf(stderr, "read %zd bytes, expected to read %zd\n", nread,
    stat.st_size);
    free(data);
    return NULL;
  }
  TF_Buffer* ret = TF_NewBufferFromString(data, stat.st_size);
  free(data);
  return ret;
}

void AssertOk(const TF_Status* status)
{
  if(TF_GetCode(status) != TF_OK)
  {
    printf("Error: %s", TF_Message(status));
    exit(1);
  }
}

TF_Output* newOutput(TF_Operation* oper, int index) {
  TF_Output* obj = malloc(sizeof(TF_Output));
  obj->oper = oper;
  obj->index = index;
  return obj;
}

/**
 * Load a protobuf buffer from disk,
 * recreate the tensorflow graph and
 * provide it for inference.
 */
Inference* newInference(
  const char* binary_graphdef_protobuffer_filename,
  const char* input_node_name,
  const char* output_node_name)
{
  Inference* obj = malloc(sizeof(Inference));
  // init the 'trival' members
  TF_Status* status = TF_NewStatus();
  obj->graph = TF_NewGraph();

  // create a bunch of objects we need to init graph and session
  TF_Buffer* graph_def = ReadBinaryProto(binary_graphdef_protobuffer_filename);
  TF_ImportGraphDefOptions* opts  = TF_NewImportGraphDefOptions();
  TF_SessionOptions* session_opts = TF_NewSessionOptions();

  // import graph
  TF_GraphImportGraphDef(obj->graph, graph_def, opts, status);
  AssertOk(status);
  // and create session
  obj->session = TF_NewSession(obj->graph, session_opts, status);
  AssertOk(status);

  // prepare the constants for inference
  // input
  obj->input_op = TF_GraphOperationByName(obj->graph, input_node_name);
  obj->input = newOutput(obj->input_op, 0);

  // output
  obj->output_op = TF_GraphOperationByName(obj->graph, output_node_name);
  obj->output = newOutput(obj->output_op, 0);

  // Clean Up all temporary objects
  TF_DeleteBuffer(graph_def);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteSessionOptions(session_opts);

  TF_DeleteStatus(status);
  return obj;
}

void destroy(Inference* inf)
{
  TF_Status* status = TF_NewStatus();
  // Clean up all the members
  TF_CloseSession(inf->session, status);
  TF_DeleteGraph(inf->graph);
  TF_DeleteSession(inf->session, status);

  TF_DeleteStatus(status);
  // input_op & output_op are delete by deleting the graph
}

TF_Tensor* runGraph(const Inference* inf, TF_Tensor* input_tensor)
{
  TF_Status* status = TF_NewStatus();
  TF_Tensor* output_tensor;
  TF_SessionRun(inf->session, NULL,
                inf->input,  &input_tensor,  1,
                inf->output, &output_tensor, 1,
                &inf->output_op, 1,
                NULL, status);
  AssertOk(status);
  TF_DeleteStatus(status);

  return output_tensor;
}

