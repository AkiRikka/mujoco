#include "experimental/nelif/gbuffer_pass.h"

#include <cmath>
#include <vector>

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

namespace mujoco::nelif {
namespace {

constexpr float kPi = 3.14159265358979323846f;

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
    return {0, 0, 0};
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

Mat4 ProjectionFromCamera(const mjvGLCamera& c, int width, int height) {
  float halfwidth = c.frustum_width ? c.frustum_width
                    : 0.5f * static_cast<float>(width) / static_cast<float>(height) *
                          (c.frustum_top - c.frustum_bottom);
  return Frustum(c.frustum_center - halfwidth,
                 c.frustum_center + halfwidth,
                 c.frustum_bottom,
                 c.frustum_top,
                 c.frustum_near,
                 c.frustum_far);
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

}  // namespace

bool GBufferPass::Init(int width, int height) {
  const char* gbuffer_vs = R"GLSL(
#version 120
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uCameraPos;
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vViewDir;
varying float vCameraDistance;
void main() {
  vec4 world = uModel * gl_Vertex;
  vWorldPos = world.xyz;
  vWorldNormal = normalize(uNormalMat * gl_Normal);
  vec3 toCamera = uCameraPos - vWorldPos;
  vCameraDistance = length(toCamera);
  vViewDir = normalize(toCamera);
  gl_Position = uProj * uView * world;
}
)GLSL";

  const char* gbuffer_fs = R"GLSL(
#version 120
#extension GL_ARB_draw_buffers : require
  uniform vec4 uDiffuse;
  uniform float uSpecular;
  uniform float uRoughness;
varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  varying vec3 vViewDir;
  varying float vCameraDistance;
  void main() {
  float gloss = 1.0 - uRoughness;
  gl_FragData[0] = vec4(vWorldPos, 1.0);
  gl_FragData[1] = vec4(normalize(vWorldNormal), 1.0);
  gl_FragData[2] = vec4(uDiffuse.rgb, 1.0);
  gl_FragData[3] = vec4(vec3(uSpecular), 1.0);
  gl_FragData[4] = vec4(gloss, 0.0, 0.0, 1.0);
  gl_FragData[5] = vec4(normalize(vViewDir), 1.0);
  }
)GLSL";

  const char* debug_vs = R"GLSL(
#version 120
varying vec2 vUv;
void main() {
  vUv = gl_MultiTexCoord0.xy;
  gl_Position = gl_Vertex;
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
  if (uDebugMode == 1) {
    color = color * 0.25 + vec3(0.5);
  } else if (uDebugMode == 2 || uDebugMode == 6) {
    color = color * 0.5 + vec3(0.5);
  } else if (uDebugMode == 5) {
    color = vec3(raw.r);
  }
  gl_FragColor = vec4(color, 1.0);
  }
)GLSL";

  gbuffer_program_ = BuildProgram(gbuffer_vs, gbuffer_fs, "G-buffer");
  debug_program_ = BuildProgram(debug_vs, debug_fs, "G-buffer debug");

  Resize(width, height);
  return true;
}

void GBufferPass::Resize(int width, int height) {
  width = width > 0 ? width : 1;
  height = height > 0 ? height : 1;

  if (width == width_ && height == height_ && fbo_) {
    return;
  }

  if (depth_) {
    glDeleteRenderbuffers(1, &depth_);
    depth_ = 0;
  }
  if (textures_[0]) {
    glDeleteTextures(kGBufferAttachmentCount, textures_);
    for (int i = 0; i < kGBufferAttachmentCount; ++i) {
      textures_[i] = 0;
    }
  }
  if (fbo_) {
    glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
  }

  width_ = width;
  height_ = height;

  glGenFramebuffers(1, &fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

  glGenTextures(kGBufferAttachmentCount, textures_);
  for (int i = 0; i < kGBufferAttachmentCount; ++i) {
    glBindTexture(GL_TEXTURE_2D, textures_[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D,
                           textures_[i], 0);
  }

  glGenRenderbuffers(1, &depth_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width_, height_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);

  GLenum draw_buffers[kGBufferAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
      GL_COLOR_ATTACHMENT2,
      GL_COLOR_ATTACHMENT3,
      GL_COLOR_ATTACHMENT4,
      GL_COLOR_ATTACHMENT5,
  };
  glDrawBuffers(kGBufferAttachmentCount, draw_buffers);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    mju_error("NeLiF G-buffer framebuffer is incomplete: 0x%x", status);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint GBufferPass::Texture(GBufferAttachment attachment) const {
  return textures_[AttachmentIndex(attachment)];
}

bool GBufferPass::Readback(GBufferAttachment attachment, std::vector<float>* rgba) const {
  return ReadTextureRGBA32F(Texture(attachment), width_, height_, rgba);
}

void GBufferPass::Render(const mjModel* model, const mjvScene* scene) {
  mjvGLCamera c = mjv_averageCamera(scene->camera, scene->camera + 1);
  Mat4 view = LookAt({c.pos[0], c.pos[1], c.pos[2]},
                     {c.forward[0], c.forward[1], c.forward[2]},
                     {c.up[0], c.up[1], c.up[2]});
  Mat4 proj = ProjectionFromCamera(c, width_, height_);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glViewport(0, 0, width_, height_);
  GLenum draw_buffers[kGBufferAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
      GL_COLOR_ATTACHMENT2,
      GL_COLOR_ATTACHMENT3,
      GL_COLOR_ATTACHMENT4,
      GL_COLOR_ATTACHMENT5,
  };
  glDrawBuffers(kGBufferAttachmentCount, draw_buffers);
  glClearColor(0, 0, 0, 0);
  glClearDepth(1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);

  glUseProgram(gbuffer_program_);
  glUniformMatrix4fv(glGetUniformLocation(gbuffer_program_, "uView"), 1, GL_FALSE, view.v);
  glUniformMatrix4fv(glGetUniformLocation(gbuffer_program_, "uProj"), 1, GL_FALSE, proj.v);
  glUniform3f(glGetUniformLocation(gbuffer_program_, "uCameraPos"), c.pos[0], c.pos[1], c.pos[2]);

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

  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GBufferPass::DrawDebug(int mode, int viewport_width, int viewport_height) const {
  GBufferAttachment attachment = GBufferAttachment::kPosition;
  switch (mode) {
    case kDebugPosition:
      attachment = GBufferAttachment::kPosition;
      break;
    case kDebugNormal:
      attachment = GBufferAttachment::kNormal;
      break;
    case kDebugDiffuse:
      attachment = GBufferAttachment::kDiffuse;
      break;
    case kDebugReflect:
      attachment = GBufferAttachment::kReflect;
      break;
    case kDebugGloss:
      attachment = GBufferAttachment::kGloss;
      break;
    case kDebugOutDir:
      attachment = GBufferAttachment::kOutDir;
      break;
    default:
      attachment = GBufferAttachment::kPosition;
      break;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, viewport_width, viewport_height);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(debug_program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, Texture(attachment));
  glUniform1i(glGetUniformLocation(debug_program_, "uTex"), 0);
  glUniform1i(glGetUniformLocation(debug_program_, "uDebugMode"), mode);

  glBegin(GL_TRIANGLE_STRIP);
  glTexCoord2f(0, 0);
  glVertex2f(-1, -1);
  glTexCoord2f(1, 0);
  glVertex2f(1, -1);
  glTexCoord2f(0, 1);
  glVertex2f(-1, 1);
  glTexCoord2f(1, 1);
  glVertex2f(1, 1);
  glEnd();

  glUseProgram(0);
}

void GBufferPass::Shutdown() {
  if (depth_) {
    glDeleteRenderbuffers(1, &depth_);
    depth_ = 0;
  }
  if (textures_[0]) {
    glDeleteTextures(kGBufferAttachmentCount, textures_);
    for (int i = 0; i < kGBufferAttachmentCount; ++i) {
      textures_[i] = 0;
    }
  }
  if (fbo_) {
    glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
  }
  if (gbuffer_program_) {
    glDeleteProgram(gbuffer_program_);
    gbuffer_program_ = 0;
  }
  if (debug_program_) {
    glDeleteProgram(debug_program_);
    debug_program_ = 0;
  }
}

void GBufferPass::DrawGeom(const mjModel* mj_model, const mjvGeom& geom) const {
  Mat4 model_matrix = ModelFromGeom(geom);
  float normal_mat[9];
  NormalMatFromGeom(geom, normal_mat);

  float roughness = std::sqrt(1.0f / std::fmax(geom.shininess * 128.0f, 1.0f));
  roughness = std::fmin(std::fmax(roughness, 0.02f), 1.0f);

  glUniformMatrix4fv(glGetUniformLocation(gbuffer_program_, "uModel"),
                     1, GL_FALSE, model_matrix.v);
  glUniformMatrix3fv(glGetUniformLocation(gbuffer_program_, "uNormalMat"), 1, GL_FALSE, normal_mat);
  glUniform4fv(glGetUniformLocation(gbuffer_program_, "uDiffuse"), 1, geom.rgba);
  glUniform1f(glGetUniformLocation(gbuffer_program_, "uSpecular"), geom.specular);
  glUniform1f(glGetUniformLocation(gbuffer_program_, "uRoughness"), roughness);

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

void GBufferPass::DrawPlane() {
  glBegin(GL_TRIANGLES);
  glNormal3f(0, 0, 1);
  glVertex3f(-1, -1, 0);
  glVertex3f(1, -1, 0);
  glVertex3f(1, 1, 0);
  glVertex3f(-1, -1, 0);
  glVertex3f(1, 1, 0);
  glVertex3f(-1, 1, 0);
  glEnd();
}

void GBufferPass::DrawBox() {
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

void GBufferPass::DrawSphere() {
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

void GBufferPass::DrawMesh(const mjModel* model, const mjvGeom& geom) {
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
