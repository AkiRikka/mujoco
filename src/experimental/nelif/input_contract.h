#ifndef MUJOCO_EXPERIMENTAL_NELIF_INPUT_CONTRACT_H_
#define MUJOCO_EXPERIMENTAL_NELIF_INPUT_CONTRACT_H_

namespace mujoco::nelif {

// This file is the shared contract between:
//   1. MuJoCo runtime passes that produce NeLiF-style inputs,
//   2. offline dataset export,
//   3. training tensor packing,
//   4. runtime neural inference.
//
// The raw EXR naming follows /References/data.
//
// All vectors are stored in MuJoCo world space unless stated otherwise.
// Radiance/color values are expected to be linear. Debug views may remap values
// for display, but tensors used for training/inference should not.
//
// Important: this contract describes the raw screen-space / light-space EXR
// buffers. Decoder-side packed tensors such as half vectors, shadow deltas,
// z_f / z ratios, and dot products are derived later during tensor packing.

struct AttachmentContract {
  const char* name;
  const char* format;
  const char* x;
  const char* y;
  const char* z;
  const char* w;
};

enum class GBufferAttachment : int {
  kPosition = 0,
  kNormal = 1,
  kDiffuse = 2,
  kReflect = 3,
  kGloss = 4,
  kOutDir = 5,
  kCount,
};

constexpr int kGBufferAttachmentCount =
    static_cast<int>(GBufferAttachment::kCount);

static constexpr AttachmentContract kGBufferContract[kGBufferAttachmentCount] = {
    {
        "Position",
        "RGBA32F",
        "world position x",
        "world position y",
        "world position z",
        "valid surface mask",
    },
    {
        "Normal",
        "RGBA32F",
        "world unit normal x",
        "world unit normal y",
        "world unit normal z",
        "valid surface mask",
    },
    {
        "Diffuse",
        "RGBA32F",
        "linear diffuse/albedo r",
        "linear diffuse/albedo g",
        "linear diffuse/albedo b",
        "valid surface mask",
    },
    {
        "Reflect",
        "RGBA32F",
        "linear specular/reflect color r",
        "linear specular/reflect color g",
        "linear specular/reflect color b",
        "valid surface mask",
    },
    {
        "Gloss",
        "RGBA32F",
        "gloss = 1 - roughness",
        "reserved",
        "reserved",
        "valid surface mask",
    },
    {
        "OutDir",
        "RGBA32F",
        "surface-to-camera unit direction x",
        "surface-to-camera unit direction y",
        "surface-to-camera unit direction z",
        "valid surface mask",
    },
};

enum class ScreenSpaceLightAttachment : int {
  kSSLightDepth = 0,
  kSSLightVec = 1,
  kCount,
};

constexpr int kScreenSpaceLightAttachmentCount =
    static_cast<int>(ScreenSpaceLightAttachment::kCount);

static constexpr AttachmentContract
    kScreenSpaceLightContract[kScreenSpaceLightAttachmentCount] = {
    {
        "ssLightDepth",
        "RGBA32F",
        "occluder-to-light distance",
        "surface-to-light distance",
        "sampled light-space depth/reserved",
        "valid surface mask",
    },
    {
        "ssLightVec",
        "RGBA32F",
        "surface-to-light unit direction x",
        "surface-to-light unit direction y",
        "surface-to-light unit direction z",
        "valid surface mask",
    },
};

enum class LightSpaceAttachment : int {
  kSMPosition = 0,
  kSMNormal = 1,
  kSMFlux = 2,
  kCount,
};

constexpr int kLightSpaceAttachmentCount =
    static_cast<int>(LightSpaceAttachment::kCount);

static constexpr AttachmentContract kLightSpaceContract[kLightSpaceAttachmentCount] = {
    {
        "smPosition",
        "RGBA32F",
        "light-view visible world position x",
        "light-view visible world position y",
        "light-view visible world position z",
        "light-to-surface distance",
    },
    {
        "smNormal",
        "RGBA32F",
        "light-view visible world normal x",
        "light-view visible world normal y",
        "light-view visible world normal z",
        "valid surface mask",
    },
    {
        "smFlux",
        "RGBA32F",
        "linear reflected flux r",
        "linear reflected flux g",
        "linear reflected flux b",
        "valid surface mask",
    },
};

enum class Stage2Target : int {
  kDirectShading = 0,
  kDirectShadowShading = 1,
  kIndirectShading = 2,
  kDiffuseDirectShading = 3,
  kSpecularDirectShading = 4,
  kDiffuseIndirectShading = 5,
  kSpecularIndirectShading = 6,
  kShading = 7,
  kCount,
};

constexpr int kStage2TargetCount = static_cast<int>(Stage2Target::kCount);

static constexpr AttachmentContract kStage2TargetContract[kStage2TargetCount] = {
    {
        "DirectShading",
        "RGBA32F",
        "unshadowed direct radiance r",
        "unshadowed direct radiance g",
        "unshadowed direct radiance b",
        "valid surface mask",
    },
    {
        "DirectShadowShading",
        "RGBA32F",
        "shadowed direct radiance r",
        "shadowed direct radiance g",
        "shadowed direct radiance b",
        "valid surface mask",
    },
    {
        "IndirectShading",
        "RGBA32F",
        "indirect radiance r",
        "indirect radiance g",
        "indirect radiance b",
        "valid surface mask",
    },
    {
        "DiffuseDirectShading",
        "RGBA32F",
        "shadowed direct diffuse radiance r",
        "shadowed direct diffuse radiance g",
        "shadowed direct diffuse radiance b",
        "valid surface mask",
    },
    {
        "SpecularDirectShading",
        "RGBA32F",
        "shadowed direct specular radiance r",
        "shadowed direct specular radiance g",
        "shadowed direct specular radiance b",
        "valid surface mask",
    },
    {
        "DiffuseIndirectShading",
        "RGBA32F",
        "indirect diffuse radiance r",
        "indirect diffuse radiance g",
        "indirect diffuse radiance b",
        "valid surface mask",
    },
    {
        "SpecularIndirectShading",
        "RGBA32F",
        "indirect specular radiance r",
        "indirect specular radiance g",
        "indirect specular radiance b",
        "valid surface mask",
    },
    {
        "Shading",
        "RGBA32F",
        "final path-traced radiance r",
        "final path-traced radiance g",
        "final path-traced radiance b",
        "valid surface mask",
    },
};

// Soft-shadow supervision is derived later during training preparation:
//   soft_shadow = DirectShadowShading / max(DirectShading, eps)
// It is intentionally not a first-class EXR target in the raw data contract.

enum class LightType : int {
  kPoint = 0,
  kSphereArea = 1,
  kRectArea = 2,
  kSpot = 3,
};

struct LightDescriptor {
  LightType type = LightType::kPoint;
  float position[3] = {0.0f, 0.0f, 0.0f};
  float direction[3] = {0.0f, 0.0f, -1.0f};
  float color[3] = {1.0f, 1.0f, 1.0f};
  float intensity = 1.0f;
  float radius = 0.0f;
  float range = 0.0f;
  float cone_angle_degrees = 180.0f;
};

constexpr int AttachmentIndex(GBufferAttachment attachment) {
  return static_cast<int>(attachment);
}

constexpr int AttachmentIndex(ScreenSpaceLightAttachment attachment) {
  return static_cast<int>(attachment);
}

constexpr int AttachmentIndex(LightSpaceAttachment attachment) {
  return static_cast<int>(attachment);
}

constexpr int TargetIndex(Stage2Target target) {
  return static_cast<int>(target);
}

}  // namespace mujoco::nelif

#endif  // MUJOCO_EXPERIMENTAL_NELIF_INPUT_CONTRACT_H_
