#ifndef MUJOCO_EXPERIMENTAL_NELIF_ONNX_BACKEND_H_
#define MUJOCO_EXPERIMENTAL_NELIF_ONNX_BACKEND_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "experimental/nelif/gbuffer_pass.h"
#include "experimental/nelif/screen_space_light_pass.h"

namespace mujoco::nelif {

struct OnnxIndirectConfig {
  std::string model_path;
  std::string execution_provider = "cpu";
  bool use_gpu_staging = false;
  bool use_pbo_readback = false;
  int screen_width = 128;
  int screen_height = 128;
  int rsm_face_size = 64;
  float max_scale[3] = {1.0f, 1.0f, 1.0f};
};

using OnnxShadowConfig = OnnxIndirectConfig;

constexpr int kOnnxRuntimeInputCount = 14;

struct OnnxRuntimeInputs {
  std::array<std::vector<float>, kOnnxRuntimeInputCount> buffers;
  std::array<std::vector<int64_t>, kOnnxRuntimeInputCount> shapes;
};

struct OnnxRuntimeInputTiming {
  double gbuffer_read_ms = 0.0;
  double gbuffer_resize_ms = 0.0;
  double screen_read_ms = 0.0;
  double screen_resize_ms = 0.0;
  double light_read_ms = 0.0;
  double light_downsample_ms = 0.0;
  double light_mask_ms = 0.0;
  double finalize_ms = 0.0;
};

bool BuildOnnxRuntimeInputs(const GBufferPass& gbuffer,
                            const ScreenSpaceLightPass& screen_space_light,
                            const float camera_pos[3],
                            const OnnxIndirectConfig& config,
                            OnnxRuntimeInputs* out,
                            std::string* error,
                            OnnxRuntimeInputTiming* timing = nullptr);

class OnnxIndirectBackend {
 public:
  OnnxIndirectBackend();
  ~OnnxIndirectBackend();

  OnnxIndirectBackend(const OnnxIndirectBackend&) = delete;
  OnnxIndirectBackend& operator=(const OnnxIndirectBackend&) = delete;

  bool Init(const OnnxIndirectConfig& config, std::string* error);
  bool IsEnabled() const;
  const char* Status() const;
  void Shutdown();

  // Writes RGBA data in OpenGL row order, ready for glTexImage2D.
  bool Run(const GBufferPass& gbuffer, const ScreenSpaceLightPass& screen_space_light,
           const float camera_pos[3], std::vector<float>* indirect_rgba,
           std::string* error);
  bool Run(OnnxRuntimeInputs& inputs, std::vector<float>* indirect_rgba,
           std::string* error);

  int output_width() const;
  int output_height() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

class OnnxShadowBackend {
 public:
  OnnxShadowBackend();
  ~OnnxShadowBackend();

  OnnxShadowBackend(const OnnxShadowBackend&) = delete;
  OnnxShadowBackend& operator=(const OnnxShadowBackend&) = delete;

  bool Init(const OnnxShadowConfig& config, std::string* error);
  bool IsEnabled() const;
  const char* Status() const;
  void Shutdown();

  // Writes RGB visibility in OpenGL row order, ready for glTexImage2D.
  bool Run(const GBufferPass& gbuffer, const ScreenSpaceLightPass& screen_space_light,
           const float camera_pos[3], std::vector<float>* shadow_rgba,
           std::string* error);
  bool Run(OnnxRuntimeInputs& inputs, std::vector<float>* shadow_rgba,
           std::string* error);

  int output_width() const;
  int output_height() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace mujoco::nelif

#endif  // MUJOCO_EXPERIMENTAL_NELIF_ONNX_BACKEND_H_
