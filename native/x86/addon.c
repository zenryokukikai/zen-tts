// Native x86-64 build of native/tts-cpu.c for zen-tts/node: the same C source with the wasm SIMD128
// intrinsics mapped to SSE4.1 (wasm_simd128.h here), as a node addon. JS passes the wasm memory's ArrayBuffer and the
// same byte offsets as to the wasm kernels, so CpuKernels runs unchanged on either. Build: native/x86/build.sh.
#include <node_api.h>
#include <stdint.h>
#define export_name(n) used
#include "../tts-cpu.c"
#ifdef WIDE
#include "wide.c"
#endif

typedef float f;
static char *B;
#define P(i) ((void *)(B + (uint32_t)a[i]))
#define N(i) ((int)a[i])
#define F(i) ((float)a[i])

static napi_value call(napi_env env, napi_callback_info info) {
  size_t argc = 16;
  napi_value argv[16];
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  int32_t k; napi_get_value_int32(env, argv[0], &k);
  void *data; size_t len; napi_get_arraybuffer_info(env, argv[1], &data, &len);
  B = data;
  double a[14] = {0};
  for (size_t i = 2; i < argc && i < 16; i++) napi_get_value_double(env, argv[i], &a[i - 2]);
  switch (k) {
    case 0: lin_f32(P(0), N(1), N(2), P(3), N(4), P(5), P(6), P(7), N(8), N(9)); break;
    case 1: LIN_P16(P(0), N(1), N(2), P(3), N(4), P(5), P(6), P(7), N(8), N(9)); break;
    case 2: LIN_I8(P(0), N(1), N(2), P(3), N(4), P(5), P(6), P(7), N(8), N(9)); break;
    case 3: im2col(P(0), N(1), N(2), N(3), P(4)); break;
    case 4: dw_norm(P(0), N(1), N(2), N(3), N(4), P(5), P(6), F(7), P(8)); break;
    case 5: vattn(P(0), N(1), N(2), N(3), N(4), P(5), P(6), P(7), P(8)); break;
    case 6: frame_in(P(0), N(1), N(2), P(3), P(4), P(5)); break;
    case 7: nsf_exc(P(0), N(1), N(2), (uint32_t)a[3], P(4)); break;
    case 8: nsf_stft(P(0), N(1), N(2), N(3), P(4), N(5), N(6), F(7), P(8), P(9)); break;
    case 9: nsf_istft(P(0), N(1), N(2), P(3), N(4), P(5)); break;
    case 10: nsf_ola(P(0), N(1), N(2), N(3), N(4), P(5), P(6)); break;
  }
  return NULL;
}

NAPI_MODULE_INIT() {
  napi_value fn;
  napi_create_function(env, "call", NAPI_AUTO_LENGTH, call, NULL, &fn);
  napi_set_named_property(env, exports, "call", fn);
  return exports;
}
