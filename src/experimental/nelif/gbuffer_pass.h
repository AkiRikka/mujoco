#ifndef MUJOCO_EXPERIMENTAL_NELIF_GBUFFER_PASS_H_
#define MUJOCO_EXPERIMENTAL_NELIF_GBUFFER_PASS_H_

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#define GL_SILENCE_DEPRECATION

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <vector>

#include "experimental/nelif/input_contract.h"

namespace mujoco::nelif {

enum DebugMode {
  kDebugMujoco = 0,
  kDebugPosition = 1,
  kDebugNormal = 2,
  kDebugDiffuse = 3,
  kDebugReflect = 4,
  kDebugGloss = 5,
  kDebugOutDir = 6,
  kDebugSSLightDepth = 7,
  kDebugSSLightVec = 8,
  kDebugSMPosition = 9,
  kDebugSMNormal = 10,
  kDebugSMFlux = 11,
};

class GBufferPass {
 public:
  bool Init(int width, int height);
  void Resize(int width, int height);
  void Render(const mjModel* model, const mjvScene* scene);
  void DrawDebug(int mode, int viewport_width, int viewport_height) const;
  void Shutdown();

  GLuint Texture(GBufferAttachment attachment) const;
  bool Readback(GBufferAttachment attachment, std::vector<float>* rgba) const;
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  void DrawGeom(const mjModel* model, const mjvGeom& geom) const;

  static void DrawPlane();
  static void DrawBox();
  static void DrawSphere();
  static void DrawMesh(const mjModel* model, const mjvGeom& geom);

  int width_ = 0;
  int height_ = 0;
  GLuint fbo_ = 0;
  GLuint textures_[kGBufferAttachmentCount] = {};
  GLuint depth_ = 0;
  GLuint gbuffer_program_ = 0;
  GLuint debug_program_ = 0;
};

}  // namespace mujoco::nelif

#endif  // MUJOCO_EXPERIMENTAL_NELIF_GBUFFER_PASS_H_
