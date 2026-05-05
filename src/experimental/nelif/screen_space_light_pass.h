#ifndef MUJOCO_EXPERIMENTAL_NELIF_SCREEN_SPACE_LIGHT_PASS_H_
#define MUJOCO_EXPERIMENTAL_NELIF_SCREEN_SPACE_LIGHT_PASS_H_

#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#define GL_SILENCE_DEPRECATION

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <vector>

#include "experimental/nelif/gbuffer_pass.h"
#include "experimental/nelif/input_contract.h"

namespace mujoco::nelif {

class ScreenSpaceLightPass {
 public:
  bool Init(int width, int height, int face_size = 512);
  void Resize(int width, int height);
  void Render(const mjModel* model, const mjvScene* scene, const GBufferPass& gbuffer);
  void DrawDebug(int mode, int viewport_width, int viewport_height) const;
  void Shutdown();

  GLuint Texture(ScreenSpaceLightAttachment attachment) const;
  GLuint Texture(LightSpaceAttachment attachment) const;
  bool Readback(ScreenSpaceLightAttachment attachment, std::vector<float>* rgba) const;
  bool Readback(LightSpaceAttachment attachment, std::vector<float>* rgba) const;
  bool HasActiveLight() const { return has_active_light_; }
  const LightDescriptor& light() const { return light_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int face_size() const { return face_size_; }
  int light_space_width() const { return face_size_; }
  int light_space_height() const { return face_size_ * 6; }

 private:
  bool UpdateActiveLight(const mjvScene* scene);
  void RenderLightAtlas(const mjModel* model, const mjvScene* scene);
  void DrawGeom(const mjModel* model, const mjvGeom& geom) const;

  static void DrawPlane();
  static void DrawBox();
  static void DrawSphere();
  static void DrawMesh(const mjModel* model, const mjvGeom& geom);

  int width_ = 0;
  int height_ = 0;
  int face_size_ = 512;

  bool has_active_light_ = false;
  LightDescriptor light_;

  GLuint output_fbo_ = 0;
  GLuint output_textures_[kScreenSpaceLightAttachmentCount] = {};
  GLuint output_depth_ = 0;

  GLuint light_space_fbo_ = 0;
  GLuint light_space_textures_[kLightSpaceAttachmentCount] = {};
  GLuint light_space_depth_ = 0;

  GLuint light_atlas_program_ = 0;
  GLuint screen_program_ = 0;
  GLuint debug_program_ = 0;
};

}  // namespace mujoco::nelif

#endif  // MUJOCO_EXPERIMENTAL_NELIF_SCREEN_SPACE_LIGHT_PASS_H_
