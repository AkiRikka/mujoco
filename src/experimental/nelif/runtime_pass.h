#ifndef MUJOCO_EXPERIMENTAL_NELIF_RUNTIME_PASS_H_
#define MUJOCO_EXPERIMENTAL_NELIF_RUNTIME_PASS_H_

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#define GL_SILENCE_DEPRECATION

#include <GLFW/glfw3.h>

#include <vector>

#include "experimental/nelif/gbuffer_pass.h"
#include "experimental/nelif/input_contract.h"
#include "experimental/nelif/screen_space_light_pass.h"

namespace mujoco::nelif {

enum class RuntimeAttachment : int {
  kDirectUnshadowed = 0,
  kDirectShadowed = 1,
  kIndirect = 2,
  kShadow = 3,
  kComposed = 4,
  kCount,
};

constexpr int kRuntimeAttachmentCount = static_cast<int>(RuntimeAttachment::kCount);

struct RuntimeConfig {
  float shadow_bias = 0.02f;
  float soft_shadow_radius = 0.0f;
  float diffuse_scale = 1.0f;
  float specular_scale = 1.0f;
  float display_exposure = 1.0f;
};

class RuntimePass {
 public:
  bool Init(int width, int height);
  void Resize(int width, int height);
  void Render(const GBufferPass& gbuffer, const ScreenSpaceLightPass& screen_space_light,
              const RuntimeConfig& config = RuntimeConfig(), GLuint indirect_texture = 0,
              GLuint shadow_texture = 0);
  void DrawDebug(RuntimeAttachment attachment, int viewport_width, int viewport_height,
                 float exposure = 1.0f) const;
  void Shutdown();

  GLuint Texture(RuntimeAttachment attachment) const;
  bool Readback(RuntimeAttachment attachment, std::vector<float>* rgba) const;
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  int width_ = 0;
  int height_ = 0;
  GLuint fbo_ = 0;
  GLuint textures_[kRuntimeAttachmentCount] = {};
  GLuint runtime_program_ = 0;
  GLuint debug_program_ = 0;
};

constexpr int AttachmentIndex(RuntimeAttachment attachment) {
  return static_cast<int>(attachment);
}

}  // namespace mujoco::nelif

#endif  // MUJOCO_EXPERIMENTAL_NELIF_RUNTIME_PASS_H_
