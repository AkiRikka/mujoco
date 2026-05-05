#include "experimental/nelif/screen_space_light_pass.h"

#include <cmath>
#include <vector>

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

namespace mujoco::nelif {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kNearPlane = 0.01f;
constexpr float kFiniteRadiusEps = 1e-6f;

struct Mat4 {
  float v[16];
};

struct Vec3 {
  float x;
  float y;
  float z;
};

Vec3 Cross(Vec3 a, Vec3 b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

float Dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Normalize(Vec3 v) {
  float len = std::sqrt(Dot(v, v));
  if (len <= 1e-8f) {
    return {0.0f, 0.0f, 0.0f};
  }
  return {v.x / len, v.y / len, v.z / len};
}

Mat4 Identity() {
  Mat4 out{};
  out.v[0] = out.v[5] = out.v[10] = out.v[15] = 1.0f;
  return out;
}

Mat4 Frustum(float left, float right, float bottom, float top, float znear, float zfar) {
  Mat4 out{};
  out.v[0] = 2.0f * znear / (right - left);
  out.v[5] = 2.0f * znear / (top - bottom);
  out.v[8] = (right + left) / (right - left);
  out.v[9] = (top + bottom) / (top - bottom);
  out.v[10] = -(zfar + znear) / (zfar - znear);
  out.v[11] = -1.0f;
  out.v[14] = -(2.0f * zfar * znear) / (zfar - znear);
  return out;
}

Mat4 Perspective90(float znear, float zfar) {
  return Frustum(-znear, znear, -znear, znear, znear, zfar);
}

Mat4 LookAt(Vec3 eye, Vec3 forward, Vec3 up) {
  Vec3 f = Normalize(forward);
  Vec3 s = Normalize(Cross(f, up));
  Vec3 u = Cross(s, f);

  Mat4 out = Identity();
  out.v[0] = s.x;
  out.v[4] = s.y;
  out.v[8] = s.z;
  out.v[1] = u.x;
  out.v[5] = u.y;
  out.v[9] = u.z;
  out.v[2] = -f.x;
  out.v[6] = -f.y;
  out.v[10] = -f.z;
  out.v[12] = -Dot(s, eye);
  out.v[13] = -Dot(u, eye);
  out.v[14] = Dot(f, eye);
  return out;
}

Mat4 ModelFromGeom(const mjvGeom& geom) {
  float sx = 1.0f;
  float sy = 1.0f;
  float sz = 1.0f;

  switch (geom.type) {
    case mjGEOM_PLANE:
      sx = geom.size[0] > 0 ? geom.size[0] : 1.0f;
      sy = geom.size[1] > 0 ? geom.size[1] : 1.0f;
      sz = 1.0f;
      break;
    case mjGEOM_SPHERE:
      sx = sy = sz = geom.size[0];
      break;
    case mjGEOM_MESH:
      sx = sy = sz = 1.0f;
      break;
    case mjGEOM_BOX:
    default:
      sx = geom.size[0];
      sy = geom.size[1];
      sz = geom.size[2];
      break;
  }

  Mat4 out{};
  out.v[0] = geom.mat[0] * sx;
  out.v[1] = geom.mat[3] * sx;
  out.v[2] = geom.mat[6] * sx;
  out.v[4] = geom.mat[1] * sy;
  out.v[5] = geom.mat[4] * sy;
  out.v[6] = geom.mat[7] * sy;
  out.v[8] = geom.mat[2] * sz;
  out.v[9] = geom.mat[5] * sz;
  out.v[10] = geom.mat[8] * sz;
  out.v[12] = geom.pos[0];
  out.v[13] = geom.pos[1];
  out.v[14] = geom.pos[2];
  out.v[15] = 1.0f;
  return out;
}

void NormalMatFromGeom(const mjvGeom& geom, float out[9]) {
  out[0] = geom.mat[0];
  out[1] = geom.mat[3];
  out[2] = geom.mat[6];
  out[3] = geom.mat[1];
  out[4] = geom.mat[4];
  out[5] = geom.mat[7];
  out[6] = geom.mat[2];
  out[7] = geom.mat[5];
  out[8] = geom.mat[8];
}

void CheckShader(GLuint shader, const char* label) {
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    mju_error("Failed to compile %s shader:\n%s", label, log);
  }
}

void CheckProgram(GLuint program, const char* label) {
  GLint ok = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetProgramInfoLog(program, sizeof(log), nullptr, log);
    mju_error("Failed to link %s program:\n%s", label, log);
  }
}

GLuint BuildProgram(const char* vs_source, const char* fs_source, const char* label) {
  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vs_source, nullptr);
  glCompileShader(vs);
  CheckShader(vs, label);

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fs_source, nullptr);
  glCompileShader(fs);
  CheckShader(fs, label);

  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  CheckProgram(program, label);

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

bool ReadTextureRGBA32F(GLuint texture, int width, int height,
                        std::vector<float>* rgba) {
  if (!texture || width <= 0 || height <= 0 || !rgba) {
    return false;
  }

  rgba->assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4, 0.0f);
  GLint previous = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
  glBindTexture(GL_TEXTURE_2D, texture);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, rgba->data());
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous));
  return glGetError() == GL_NO_ERROR;
}

Vec3 CubeFaceForward(int face) {
  switch (face) {
    case 0: return {1.0f, 0.0f, 0.0f};
    case 1: return {-1.0f, 0.0f, 0.0f};
    case 2: return {0.0f, 1.0f, 0.0f};
    case 3: return {0.0f, -1.0f, 0.0f};
    case 4: return {0.0f, 0.0f, 1.0f};
    default: return {0.0f, 0.0f, -1.0f};
  }
}

Vec3 CubeFaceUp(int face) {
  switch (face) {
    case 0:
    case 1:
    case 4:
    case 5:
      return {0.0f, 1.0f, 0.0f};
    case 2:
      return {0.0f, 0.0f, -1.0f};
    default:
      return {0.0f, 0.0f, 1.0f};
  }
}

}  // namespace

bool ScreenSpaceLightPass::Init(int width, int height, int face_size) {
  const char* atlas_vs = R"GLSL(
#version 120
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightPos;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying float vLightDistance;
void main() {
  vec4 world = uModel * gl_Vertex;
  vWorldPos = world.xyz;
  vWorldNormal = normalize(uNormalMat * gl_Normal);
  vLightDistance = length(vWorldPos - uLightPos);
  gl_Position = uProj * uView * world;
}
)GLSL";

  const char* atlas_fs = R"GLSL(
#version 120
#extension GL_ARB_draw_buffers : require
uniform vec3 uLightPos;
uniform vec3 uDiffuse;
uniform vec3 uLightColor;
uniform float uLightIntensity;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying float vLightDistance;
void main() {
  vec3 toLight = normalize(uLightPos - vWorldPos);
  float ndotl = max(dot(normalize(vWorldNormal), toLight), 0.0);
  float attenuation = uLightIntensity / max(vLightDistance * vLightDistance, 1e-4);
  vec3 flux = uDiffuse * uLightColor * attenuation * ndotl;
  gl_FragData[0] = vec4(vWorldPos, vLightDistance);
  gl_FragData[1] = vec4(normalize(vWorldNormal), 1.0);
  gl_FragData[2] = vec4(flux, 1.0);
}
)GLSL";

  const char* screen_vs = R"GLSL(
#version 120
varying vec2 vUv;
void main() {
  vUv = gl_MultiTexCoord0.xy;
  gl_Position = gl_Vertex;
}
)GLSL";

  const char* screen_fs = R"GLSL(
#version 120
#extension GL_ARB_draw_buffers : require
uniform sampler2D uPositionTex;
uniform sampler2D uLightSpacePositionTex;
uniform vec3 uLightPos;
varying vec2 vUv;

vec3 SafeNormalize(vec3 v) {
  float len = length(v);
  return len > 1e-8 ? v / len : vec3(0.0, 0.0, 1.0);
}

void main() {
  vec4 position_sample = texture2D(uPositionTex, vUv);
  float mask = position_sample.a;
  if (mask < 0.5) {
    gl_FragData[0] = vec4(0.0);
    gl_FragData[1] = vec4(0.0);
    return;
  }

  vec3 world_pos = position_sample.xyz;
  vec3 light_to_surface = world_pos - uLightPos;
  float surface_dist = length(light_to_surface);
  vec3 dir = SafeNormalize(light_to_surface);
  vec3 abs_dir = abs(dir);

  float face = 0.0;
  vec2 uv = vec2(0.5);
  if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
    if (dir.x > 0.0) {
      face = 0.0;
      uv = vec2(dir.z, dir.y) / abs_dir.x;
    } else {
      face = 1.0;
      uv = vec2(-dir.z, dir.y) / abs_dir.x;
    }
  } else if (abs_dir.y >= abs_dir.z) {
    if (dir.y > 0.0) {
      face = 2.0;
      uv = vec2(-dir.x, -dir.z) / abs_dir.y;
    } else {
      face = 3.0;
      uv = vec2(-dir.x, dir.z) / abs_dir.y;
    }
  } else {
    if (dir.z > 0.0) {
      face = 4.0;
      uv = vec2(-dir.x, dir.y) / abs_dir.z;
    } else {
      face = 5.0;
      uv = vec2(dir.x, dir.y) / abs_dir.z;
    }
  }

  uv = uv * 0.5 + 0.5;
  vec2 atlas_uv = vec2(uv.x, (uv.y + face) / 6.0);
  vec4 atlas_sample = texture2D(uLightSpacePositionTex, atlas_uv);
  float sampled_w = atlas_sample.w;
  float occluder_dist = length(atlas_sample.xyz - uLightPos);
  if (sampled_w < 1e-6) {
    occluder_dist = surface_dist + 1.0;
  }

  gl_FragData[0] = vec4(occluder_dist, surface_dist, sampled_w, mask);
  gl_FragData[1] = vec4(SafeNormalize(uLightPos - world_pos), mask);
}
)GLSL";

  const char* debug_fs = R"GLSL(
#version 120
uniform sampler2D uTex;
uniform int uDebugMode;
varying vec2 vUv;
void main() {
  vec4 raw = texture2D(uTex, vUv);
  vec3 color = raw.rgb;
  if (uDebugMode == 7) {
    float visibility = raw.r + 1e-3 >= raw.g ? 1.0 : 0.0;
    color = vec3(visibility);
  } else if (uDebugMode == 8) {
    color = raw.rgb * 0.5 + vec3(0.5);
  } else if (uDebugMode == 9) {
    color = raw.rgb * 0.25 + vec3(0.5);
  } else if (uDebugMode == 10) {
    color = raw.rgb * 0.5 + vec3(0.5);
  } else if (uDebugMode == 11) {
    color = raw.rgb / (vec3(1.0) + raw.rgb);
  }
  gl_FragColor = vec4(color, 1.0);
}
)GLSL";

  face_size_ = face_size > 0 ? face_size : 512;
  light_atlas_program_ = BuildProgram(atlas_vs, atlas_fs, "NeLiF light atlas");
  screen_program_ = BuildProgram(screen_vs, screen_fs, "NeLiF screen-space light");
  debug_program_ = BuildProgram(screen_vs, debug_fs, "NeLiF screen-space light debug");
  Resize(width, height);
  return true;
}

void ScreenSpaceLightPass::Resize(int width, int height) {
  width = width > 0 ? width : 1;
  height = height > 0 ? height : 1;
  if (width == width_ && height == height_ && output_fbo_ && light_space_fbo_) {
    return;
  }

  if (output_depth_) {
    glDeleteRenderbuffers(1, &output_depth_);
    output_depth_ = 0;
  }
  if (output_textures_[0]) {
    glDeleteTextures(kScreenSpaceLightAttachmentCount, output_textures_);
    for (int i = 0; i < kScreenSpaceLightAttachmentCount; ++i) {
      output_textures_[i] = 0;
    }
  }
  if (output_fbo_) {
    glDeleteFramebuffers(1, &output_fbo_);
    output_fbo_ = 0;
  }

  if (light_space_depth_) {
    glDeleteRenderbuffers(1, &light_space_depth_);
    light_space_depth_ = 0;
  }
  if (light_space_textures_[0]) {
    glDeleteTextures(kLightSpaceAttachmentCount, light_space_textures_);
    for (int i = 0; i < kLightSpaceAttachmentCount; ++i) {
      light_space_textures_[i] = 0;
    }
  }
  if (light_space_fbo_) {
    glDeleteFramebuffers(1, &light_space_fbo_);
    light_space_fbo_ = 0;
  }

  width_ = width;
  height_ = height;

  glGenFramebuffers(1, &output_fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, output_fbo_);

  glGenTextures(kScreenSpaceLightAttachmentCount, output_textures_);
  for (int i = 0; i < kScreenSpaceLightAttachmentCount; ++i) {
    glBindTexture(GL_TEXTURE_2D, output_textures_[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                           GL_TEXTURE_2D, output_textures_[i], 0);
  }

  glGenRenderbuffers(1, &output_depth_);
  glBindRenderbuffer(GL_RENDERBUFFER, output_depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width_, height_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, output_depth_);

  GLenum output_buffers[kScreenSpaceLightAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
  };
  glDrawBuffers(kScreenSpaceLightAttachmentCount, output_buffers);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    mju_error("NeLiF screen-space light framebuffer is incomplete: 0x%x", status);
  }

  glGenFramebuffers(1, &light_space_fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, light_space_fbo_);

  glGenTextures(kLightSpaceAttachmentCount, light_space_textures_);
  for (int i = 0; i < kLightSpaceAttachmentCount; ++i) {
    glBindTexture(GL_TEXTURE_2D, light_space_textures_[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, face_size_, face_size_ * 6, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                           GL_TEXTURE_2D, light_space_textures_[i], 0);
  }

  glGenRenderbuffers(1, &light_space_depth_);
  glBindRenderbuffer(GL_RENDERBUFFER, light_space_depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                        face_size_, face_size_ * 6);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, light_space_depth_);

  GLenum atlas_buffers[kLightSpaceAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
      GL_COLOR_ATTACHMENT2,
  };
  glDrawBuffers(kLightSpaceAttachmentCount, atlas_buffers);

  status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    mju_error("NeLiF light atlas framebuffer is incomplete: 0x%x", status);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint ScreenSpaceLightPass::Texture(ScreenSpaceLightAttachment attachment) const {
  return output_textures_[AttachmentIndex(attachment)];
}

GLuint ScreenSpaceLightPass::Texture(LightSpaceAttachment attachment) const {
  return light_space_textures_[AttachmentIndex(attachment)];
}

bool ScreenSpaceLightPass::Readback(ScreenSpaceLightAttachment attachment,
                                    std::vector<float>* rgba) const {
  return ReadTextureRGBA32F(Texture(attachment), width_, height_, rgba);
}

bool ScreenSpaceLightPass::Readback(LightSpaceAttachment attachment,
                                    std::vector<float>* rgba) const {
  return ReadTextureRGBA32F(Texture(attachment), face_size_, face_size_ * 6, rgba);
}

bool ScreenSpaceLightPass::UpdateActiveLight(const mjvScene* scene) {
  has_active_light_ = false;
  light_ = LightDescriptor{};

  for (int i = 0; i < scene->nlight; ++i) {
    const mjvLight& candidate = scene->lights[i];
    if (candidate.headlight || candidate.type == mjLIGHT_DIRECTIONAL) {
      continue;
    }

    light_.type = candidate.type == mjLIGHT_SPOT
                      ? LightType::kSpot
                      : (candidate.bulbradius > kFiniteRadiusEps
                             ? LightType::kSphereArea
                             : LightType::kPoint);
    light_.position[0] = candidate.pos[0];
    light_.position[1] = candidate.pos[1];
    light_.position[2] = candidate.pos[2];
    light_.direction[0] = candidate.dir[0];
    light_.direction[1] = candidate.dir[1];
    light_.direction[2] = candidate.dir[2];
    light_.color[0] = candidate.diffuse[0];
    light_.color[1] = candidate.diffuse[1];
    light_.color[2] = candidate.diffuse[2];
    light_.intensity = candidate.intensity > 0.0f ? candidate.intensity : 1.0f;
    light_.radius = candidate.bulbradius;
    light_.range = candidate.range;
    light_.cone_angle_degrees = candidate.cutoff;
    has_active_light_ = true;
    break;
  }

  return has_active_light_;
}

void ScreenSpaceLightPass::Render(const mjModel* model, const mjvScene* scene,
                                  const GBufferPass& gbuffer) {
  if (!UpdateActiveLight(scene)) {
    glBindFramebuffer(GL_FRAMEBUFFER, output_fbo_);
    glViewport(0, 0, width_, height_);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, light_space_fbo_);
    glViewport(0, 0, face_size_, face_size_ * 6);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return;
  }

  RenderLightAtlas(model, scene);

  glBindFramebuffer(GL_FRAMEBUFFER, output_fbo_);
  glViewport(0, 0, width_, height_);
  GLenum draw_buffers[kScreenSpaceLightAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
  };
  glDrawBuffers(kScreenSpaceLightAttachmentCount, draw_buffers);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glUseProgram(screen_program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, gbuffer.Texture(GBufferAttachment::kPosition));
  glUniform1i(glGetUniformLocation(screen_program_, "uPositionTex"), 0);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, Texture(LightSpaceAttachment::kSMPosition));
  glUniform1i(glGetUniformLocation(screen_program_, "uLightSpacePositionTex"), 1);
  glUniform3f(glGetUniformLocation(screen_program_, "uLightPos"),
              light_.position[0], light_.position[1], light_.position[2]);

  glBegin(GL_TRIANGLE_STRIP);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glEnd();

  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ScreenSpaceLightPass::RenderLightAtlas(const mjModel* model, const mjvScene* scene) {
  const Vec3 light_pos = {light_.position[0], light_.position[1], light_.position[2]};
  const float far_plane = light_.range > (kNearPlane + 0.1f) ? light_.range : 20.0f;
  const Mat4 proj = Perspective90(kNearPlane, far_plane);

  glBindFramebuffer(GL_FRAMEBUFFER, light_space_fbo_);
  glViewport(0, 0, face_size_, face_size_ * 6);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glEnable(GL_SCISSOR_TEST);

  glUseProgram(light_atlas_program_);
  glUniformMatrix4fv(glGetUniformLocation(light_atlas_program_, "uProj"),
                     1, GL_FALSE, proj.v);
  glUniform3f(glGetUniformLocation(light_atlas_program_, "uLightPos"),
              light_pos.x, light_pos.y, light_pos.z);
  glUniform3f(glGetUniformLocation(light_atlas_program_, "uLightColor"),
              light_.color[0], light_.color[1], light_.color[2]);
  glUniform1f(glGetUniformLocation(light_atlas_program_, "uLightIntensity"),
              light_.intensity > 0.0f ? light_.intensity : 1.0f);

  for (int face = 0; face < 6; ++face) {
    Mat4 view = LookAt(light_pos, CubeFaceForward(face), CubeFaceUp(face));
    glViewport(0, face * face_size_, face_size_, face_size_);
    glScissor(0, face * face_size_, face_size_, face_size_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUniformMatrix4fv(glGetUniformLocation(light_atlas_program_, "uView"),
                       1, GL_FALSE, view.v);

    for (int i = 0; i < scene->ngeom; ++i) {
      const mjvGeom& geom = scene->geoms[i];
      if (geom.category == mjCAT_DECOR || geom.rgba[3] <= 0.0f) {
        continue;
      }
      if (geom.type != mjGEOM_PLANE && geom.type != mjGEOM_BOX &&
          geom.type != mjGEOM_SPHERE && geom.type != mjGEOM_MESH) {
        continue;
      }
      DrawGeom(model, geom);
    }
  }

  glDisable(GL_SCISSOR_TEST);
  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ScreenSpaceLightPass::DrawDebug(int mode, int viewport_width, int viewport_height) const {
  ScreenSpaceLightAttachment attachment = ScreenSpaceLightAttachment::kSSLightDepth;
  bool use_light_space = false;
  LightSpaceAttachment light_attachment = LightSpaceAttachment::kSMPosition;
  if (mode == kDebugSSLightVec) {
    attachment = ScreenSpaceLightAttachment::kSSLightVec;
  } else if (mode == kDebugSMPosition) {
    use_light_space = true;
    light_attachment = LightSpaceAttachment::kSMPosition;
  } else if (mode == kDebugSMNormal) {
    use_light_space = true;
    light_attachment = LightSpaceAttachment::kSMNormal;
  } else if (mode == kDebugSMFlux) {
    use_light_space = true;
    light_attachment = LightSpaceAttachment::kSMFlux;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, viewport_width, viewport_height);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(debug_program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D,
                use_light_space ? Texture(light_attachment) : Texture(attachment));
  glUniform1i(glGetUniformLocation(debug_program_, "uTex"), 0);
  glUniform1i(glGetUniformLocation(debug_program_, "uDebugMode"), mode);

  glBegin(GL_TRIANGLE_STRIP);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glEnd();

  glUseProgram(0);
}

void ScreenSpaceLightPass::Shutdown() {
  if (output_depth_) {
    glDeleteRenderbuffers(1, &output_depth_);
    output_depth_ = 0;
  }
  if (output_textures_[0]) {
    glDeleteTextures(kScreenSpaceLightAttachmentCount, output_textures_);
    for (int i = 0; i < kScreenSpaceLightAttachmentCount; ++i) {
      output_textures_[i] = 0;
    }
  }
  if (output_fbo_) {
    glDeleteFramebuffers(1, &output_fbo_);
    output_fbo_ = 0;
  }

  if (light_space_depth_) {
    glDeleteRenderbuffers(1, &light_space_depth_);
    light_space_depth_ = 0;
  }
  if (light_space_textures_[0]) {
    glDeleteTextures(kLightSpaceAttachmentCount, light_space_textures_);
    for (int i = 0; i < kLightSpaceAttachmentCount; ++i) {
      light_space_textures_[i] = 0;
    }
  }
  if (light_space_fbo_) {
    glDeleteFramebuffers(1, &light_space_fbo_);
    light_space_fbo_ = 0;
  }

  if (light_atlas_program_) {
    glDeleteProgram(light_atlas_program_);
    light_atlas_program_ = 0;
  }
  if (screen_program_) {
    glDeleteProgram(screen_program_);
    screen_program_ = 0;
  }
  if (debug_program_) {
    glDeleteProgram(debug_program_);
    debug_program_ = 0;
  }
}

void ScreenSpaceLightPass::DrawGeom(const mjModel* mj_model, const mjvGeom& geom) const {
  Mat4 model_matrix = ModelFromGeom(geom);
  float normal_mat[9];
  NormalMatFromGeom(geom, normal_mat);
  glUniformMatrix4fv(glGetUniformLocation(light_atlas_program_, "uModel"),
                     1, GL_FALSE, model_matrix.v);
  glUniformMatrix3fv(glGetUniformLocation(light_atlas_program_, "uNormalMat"),
                     1, GL_FALSE, normal_mat);
  glUniform3f(glGetUniformLocation(light_atlas_program_, "uDiffuse"),
              geom.rgba[0], geom.rgba[1], geom.rgba[2]);

  if (geom.type == mjGEOM_PLANE) {
    DrawPlane();
  } else if (geom.type == mjGEOM_BOX) {
    DrawBox();
  } else if (geom.type == mjGEOM_SPHERE) {
    DrawSphere();
  } else if (geom.type == mjGEOM_MESH) {
    DrawMesh(mj_model, geom);
  }
}

void ScreenSpaceLightPass::DrawPlane() {
  glBegin(GL_TRIANGLES);
  glNormal3f(0.0f, 0.0f, 1.0f);
  glVertex3f(-1.0f, -1.0f, 0.0f);
  glVertex3f(1.0f, -1.0f, 0.0f);
  glVertex3f(1.0f, 1.0f, 0.0f);
  glVertex3f(-1.0f, -1.0f, 0.0f);
  glVertex3f(1.0f, 1.0f, 0.0f);
  glVertex3f(-1.0f, 1.0f, 0.0f);
  glEnd();
}

void ScreenSpaceLightPass::DrawBox() {
  struct Face {
    float n[3];
    float v[4][3];
  };
  const Face faces[6] = {
      {{1, 0, 0}, {{1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}}},
      {{-1, 0, 0}, {{-1, 1, -1}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}}},
      {{0, 1, 0}, {{-1, 1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}}},
      {{0, -1, 0}, {{1, -1, -1}, {-1, -1, -1}, {-1, -1, 1}, {1, -1, 1}}},
      {{0, 0, 1}, {{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}}},
      {{0, 0, -1}, {{-1, 1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}}},
  };
  glBegin(GL_TRIANGLES);
  for (const Face& f : faces) {
    glNormal3fv(f.n);
    glVertex3fv(f.v[0]);
    glVertex3fv(f.v[1]);
    glVertex3fv(f.v[2]);
    glVertex3fv(f.v[0]);
    glVertex3fv(f.v[2]);
    glVertex3fv(f.v[3]);
  }
  glEnd();
}

void ScreenSpaceLightPass::DrawSphere() {
  constexpr int stacks = 24;
  constexpr int slices = 48;
  for (int i = 0; i < stacks; ++i) {
    float v0 = static_cast<float>(i) / stacks;
    float v1 = static_cast<float>(i + 1) / stacks;
    float phi0 = (v0 - 0.5f) * kPi;
    float phi1 = (v1 - 0.5f) * kPi;
    glBegin(GL_TRIANGLE_STRIP);
    for (int j = 0; j <= slices; ++j) {
      float u = static_cast<float>(j) / slices;
      float theta = u * 2.0f * kPi;
      float cp0 = std::cos(phi0);
      float cp1 = std::cos(phi1);
      float p0[3] = {cp0 * std::cos(theta), cp0 * std::sin(theta), std::sin(phi0)};
      float p1[3] = {cp1 * std::cos(theta), cp1 * std::sin(theta), std::sin(phi1)};
      glNormal3fv(p0);
      glVertex3fv(p0);
      glNormal3fv(p1);
      glVertex3fv(p1);
    }
    glEnd();
  }
}

void ScreenSpaceLightPass::DrawMesh(const mjModel* model, const mjvGeom& geom) {
  if (!model || geom.dataid < 0) {
    return;
  }

  const int mesh_id = geom.dataid / 2;
  if (mesh_id < 0 || mesh_id >= model->nmesh) {
    return;
  }

  const int vert_adr = model->mesh_vertadr[mesh_id];
  const int face_adr = model->mesh_faceadr[mesh_id];
  const int face_num = model->mesh_facenum[mesh_id];
  const int normal_adr = model->mesh_normaladr[mesh_id];
  const bool has_normals =
      normal_adr >= 0 && model->mesh_normalnum[mesh_id] > 0 &&
      model->mesh_facenormal != nullptr;

  glBegin(GL_TRIANGLES);
  for (int i = 0; i < face_num; ++i) {
    const int face = 3 * (face_adr + i);
    const int i0 = model->mesh_face[face + 0];
    const int i1 = model->mesh_face[face + 1];
    const int i2 = model->mesh_face[face + 2];
    const float* v0 = model->mesh_vert + 3 * (vert_adr + i0);
    const float* v1 = model->mesh_vert + 3 * (vert_adr + i1);
    const float* v2 = model->mesh_vert + 3 * (vert_adr + i2);

    Vec3 edge0 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    Vec3 edge1 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    Vec3 face_normal = Normalize(Cross(edge0, edge1));

    for (int corner = 0; corner < 3; ++corner) {
      Vec3 normal_vec = face_normal;
      if (has_normals) {
        const int normal_index = model->mesh_facenormal[face + corner];
        const float* normal = model->mesh_normal + 3 * (normal_adr + normal_index);
        Vec3 mesh_normal = Normalize({normal[0], normal[1], normal[2]});
        if (Dot(face_normal, mesh_normal) >= 0.8f) {
          normal_vec = mesh_normal;
        }
      }
      glNormal3f(normal_vec.x, normal_vec.y, normal_vec.z);
      const int vertex_index = model->mesh_face[face + corner];
      glVertex3fv(model->mesh_vert + 3 * (vert_adr + vertex_index));
    }
  }
  glEnd();
}

}  // namespace mujoco::nelif
