#include "experimental/nelif/runtime_pass.h"

#include <algorithm>
#include <vector>

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

namespace mujoco::nelif {
namespace {

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

bool ReadTextureRGBA32F(GLuint texture, int width, int height, std::vector<float>* rgba) {
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

void BindTexture2D(GLuint program, const char* uniform_name, GLenum unit, GLuint texture) {
  glActiveTexture(unit);
  glBindTexture(GL_TEXTURE_2D, texture);
  glUniform1i(glGetUniformLocation(program, uniform_name), unit - GL_TEXTURE0);
}

void DrawFullScreenQuad() {
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
}

}  // namespace

bool RuntimePass::Init(int width, int height) {
  const char* full_screen_vs = R"GLSL(
#version 120
varying vec2 vUv;
void main() {
  vUv = gl_MultiTexCoord0.xy;
  gl_Position = gl_Vertex;
}
)GLSL";

  const char* runtime_fs = R"GLSL(
#version 120
#extension GL_ARB_draw_buffers : require
uniform sampler2D uPosition;
uniform sampler2D uNormal;
uniform sampler2D uDiffuse;
uniform sampler2D uReflect;
uniform sampler2D uGloss;
uniform sampler2D uOutDir;
uniform sampler2D uSSLightDepth;
uniform sampler2D uSSLightVec;
uniform sampler2D uIndirect;
uniform sampler2D uShadow;
uniform int uUseIndirect;
uniform int uUseShadow;
uniform vec3 uLightColor;
uniform float uLightIntensity;
uniform float uShadowBias;
uniform float uSoftShadowRadius;
uniform float uDiffuseScale;
uniform float uSpecularScale;
varying vec2 vUv;

vec3 safe_normalize(vec3 value) {
  float len = length(value);
  if (len <= 1e-6) {
    return vec3(0.0);
  }
  return value / len;
}

void main() {
  vec4 position = texture2D(uPosition, vUv);
  float valid = step(0.5, position.a);
  vec3 normal = safe_normalize(texture2D(uNormal, vUv).rgb);
  vec3 albedo = max(texture2D(uDiffuse, vUv).rgb, vec3(0.0));
  vec3 specular = max(texture2D(uReflect, vUv).rgb, vec3(0.0));
  float gloss = clamp(texture2D(uGloss, vUv).r, 0.0, 1.0);
  vec3 light_vec = safe_normalize(texture2D(uSSLightVec, vUv).rgb);
  vec3 view_dir = safe_normalize(texture2D(uOutDir, vUv).rgb);
  vec2 depth = texture2D(uSSLightDepth, vUv).rg;

  float ndotl = max(dot(normal, light_vec), 0.0);
  float distance = max(depth.g, 1e-4);
  float attenuation = 1.0 / max(distance * distance, 1e-4);
  vec3 light_rgb = max(uLightColor * uLightIntensity, vec3(0.0));

  vec3 diffuse_direct = albedo * light_rgb * attenuation * ndotl * valid * uDiffuseScale;

  vec3 half_vec = safe_normalize(light_vec + view_dir);
  float ndoth = max(dot(normal, half_vec), 0.0);
  float spec_power = 8.0 + gloss * gloss * 120.0;
  float spec_lobe = pow(ndoth, spec_power) * ndotl;
  vec3 specular_direct = specular * light_rgb * attenuation * spec_lobe * valid * uSpecularScale;

  vec3 direct_unshadowed = diffuse_direct + specular_direct;

  float delta = depth.g - depth.r - uShadowBias;
  float visibility = 0.0;
  if (uSoftShadowRadius > 1e-8) {
    visibility = 1.0 - clamp(delta / uSoftShadowRadius, 0.0, 1.0);
  } else {
    visibility = delta <= 0.0 ? 1.0 : 0.0;
  }
  visibility *= valid;

  vec3 shadow = vec3(visibility);
  if (uUseShadow != 0) {
    shadow = clamp(texture2D(uShadow, vUv).rgb, 0.0, 1.0) * valid;
  }
  vec3 direct_shadowed = direct_unshadowed * shadow;

  vec3 indirect = vec3(0.0);
  if (uUseIndirect != 0) {
    indirect = max(texture2D(uIndirect, vUv).rgb, vec3(0.0)) * valid;
  }
  vec3 composed = direct_shadowed + indirect;

  gl_FragData[0] = vec4(direct_unshadowed, valid);
  gl_FragData[1] = vec4(direct_shadowed, valid);
  gl_FragData[2] = vec4(indirect, valid);
  gl_FragData[3] = vec4(shadow, valid);
  gl_FragData[4] = vec4(composed, valid);
}
)GLSL";

  const char* debug_fs = R"GLSL(
#version 120
uniform sampler2D uTex;
uniform float uExposure;
uniform int uIsShadow;
varying vec2 vUv;
void main() {
  vec4 raw = texture2D(uTex, vUv);
  vec3 color = raw.rgb;
  if (uIsShadow == 0) {
    color = color * max(uExposure, 0.0);
  }
  color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
  gl_FragColor = vec4(color, 1.0);
}
)GLSL";

  runtime_program_ = BuildProgram(full_screen_vs, runtime_fs, "NeLiF runtime");
  debug_program_ = BuildProgram(full_screen_vs, debug_fs, "NeLiF runtime debug");
  Resize(width, height);
  return true;
}

void RuntimePass::Resize(int width, int height) {
  width = width > 0 ? width : 1;
  height = height > 0 ? height : 1;
  if (width == width_ && height == height_ && fbo_) {
    return;
  }

  if (textures_[0]) {
    glDeleteTextures(kRuntimeAttachmentCount, textures_);
    for (int i = 0; i < kRuntimeAttachmentCount; ++i) {
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

  glGenTextures(kRuntimeAttachmentCount, textures_);
  for (int i = 0; i < kRuntimeAttachmentCount; ++i) {
    glBindTexture(GL_TEXTURE_2D, textures_[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D,
                           textures_[i], 0);
  }

  GLenum draw_buffers[kRuntimeAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
      GL_COLOR_ATTACHMENT2,
      GL_COLOR_ATTACHMENT3,
      GL_COLOR_ATTACHMENT4,
  };
  glDrawBuffers(kRuntimeAttachmentCount, draw_buffers);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    mju_error("NeLiF runtime framebuffer is incomplete: 0x%x", status);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RuntimePass::Render(const GBufferPass& gbuffer,
                         const ScreenSpaceLightPass& screen_space_light,
                         const RuntimeConfig& config,
                         GLuint indirect_texture,
                         GLuint shadow_texture) {
  Resize(gbuffer.width(), gbuffer.height());

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glViewport(0, 0, width_, height_);
  GLenum draw_buffers[kRuntimeAttachmentCount] = {
      GL_COLOR_ATTACHMENT0,
      GL_COLOR_ATTACHMENT1,
      GL_COLOR_ATTACHMENT2,
      GL_COLOR_ATTACHMENT3,
      GL_COLOR_ATTACHMENT4,
  };
  glDrawBuffers(kRuntimeAttachmentCount, draw_buffers);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glUseProgram(runtime_program_);
  BindTexture2D(runtime_program_, "uPosition", GL_TEXTURE0,
                gbuffer.Texture(GBufferAttachment::kPosition));
  BindTexture2D(runtime_program_, "uNormal", GL_TEXTURE1,
                gbuffer.Texture(GBufferAttachment::kNormal));
  BindTexture2D(runtime_program_, "uDiffuse", GL_TEXTURE2,
                gbuffer.Texture(GBufferAttachment::kDiffuse));
  BindTexture2D(runtime_program_, "uReflect", GL_TEXTURE3,
                gbuffer.Texture(GBufferAttachment::kReflect));
  BindTexture2D(runtime_program_, "uGloss", GL_TEXTURE4,
                gbuffer.Texture(GBufferAttachment::kGloss));
  BindTexture2D(runtime_program_, "uOutDir", GL_TEXTURE5,
                gbuffer.Texture(GBufferAttachment::kOutDir));
  BindTexture2D(runtime_program_, "uSSLightDepth", GL_TEXTURE6,
                screen_space_light.Texture(ScreenSpaceLightAttachment::kSSLightDepth));
  BindTexture2D(runtime_program_, "uSSLightVec", GL_TEXTURE7,
                screen_space_light.Texture(ScreenSpaceLightAttachment::kSSLightVec));
  if (indirect_texture) {
    BindTexture2D(runtime_program_, "uIndirect", GL_TEXTURE8, indirect_texture);
  } else {
    glUniform1i(glGetUniformLocation(runtime_program_, "uIndirect"), 0);
  }
  if (shadow_texture) {
    BindTexture2D(runtime_program_, "uShadow", GL_TEXTURE9, shadow_texture);
  } else {
    glUniform1i(glGetUniformLocation(runtime_program_, "uShadow"), 0);
  }

  float light_color[3] = {1.0f, 1.0f, 1.0f};
  float light_intensity = 0.0f;
  if (screen_space_light.HasActiveLight()) {
    const LightDescriptor& light = screen_space_light.light();
    light_color[0] = light.color[0];
    light_color[1] = light.color[1];
    light_color[2] = light.color[2];
    light_intensity = light.intensity;
  }
  glUniform3fv(glGetUniformLocation(runtime_program_, "uLightColor"), 1, light_color);
  glUniform1f(glGetUniformLocation(runtime_program_, "uLightIntensity"), light_intensity);
  glUniform1f(glGetUniformLocation(runtime_program_, "uShadowBias"), config.shadow_bias);
  glUniform1f(glGetUniformLocation(runtime_program_, "uSoftShadowRadius"),
              config.soft_shadow_radius);
  glUniform1f(glGetUniformLocation(runtime_program_, "uDiffuseScale"), config.diffuse_scale);
  glUniform1f(glGetUniformLocation(runtime_program_, "uSpecularScale"), config.specular_scale);
  glUniform1i(glGetUniformLocation(runtime_program_, "uUseIndirect"),
              indirect_texture ? 1 : 0);
  glUniform1i(glGetUniformLocation(runtime_program_, "uUseShadow"),
              shadow_texture ? 1 : 0);

  DrawFullScreenQuad();

  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RuntimePass::DrawDebug(RuntimeAttachment attachment, int viewport_width, int viewport_height,
                            float exposure) const {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, std::max(viewport_width, 1), std::max(viewport_height, 1));
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(debug_program_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, Texture(attachment));
  glUniform1i(glGetUniformLocation(debug_program_, "uTex"), 0);
  glUniform1f(glGetUniformLocation(debug_program_, "uExposure"), exposure);
  glUniform1i(glGetUniformLocation(debug_program_, "uIsShadow"),
              attachment == RuntimeAttachment::kShadow ? 1 : 0);
  DrawFullScreenQuad();
  glUseProgram(0);
}

void RuntimePass::Shutdown() {
  if (textures_[0]) {
    glDeleteTextures(kRuntimeAttachmentCount, textures_);
    for (int i = 0; i < kRuntimeAttachmentCount; ++i) {
      textures_[i] = 0;
    }
  }
  if (fbo_) {
    glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
  }
  if (runtime_program_) {
    glDeleteProgram(runtime_program_);
    runtime_program_ = 0;
  }
  if (debug_program_) {
    glDeleteProgram(debug_program_);
    debug_program_ = 0;
  }
}

GLuint RuntimePass::Texture(RuntimeAttachment attachment) const {
  return textures_[AttachmentIndex(attachment)];
}

bool RuntimePass::Readback(RuntimeAttachment attachment, std::vector<float>* rgba) const {
  return ReadTextureRGBA32F(Texture(attachment), width_, height_, rgba);
}

}  // namespace mujoco::nelif
