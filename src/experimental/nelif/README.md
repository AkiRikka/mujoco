# MuJoCo NeLiF Raw Inputs

这份 README 只描述 **MuJoCo / G-buffer / light-space raw input** 这一侧。
它对应的是 NeLiF stage 2 的输入基础设施，不包括 Blender/Cycles 那边的 GT 生成。

当前实现代码主要在：

- [input_contract.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/input_contract.h)
- [gbuffer_pass.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/gbuffer_pass.h)
- [gbuffer_pass.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/gbuffer_pass.cc)
- [screen_space_light_pass.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/screen_space_light_pass.h)
- [screen_space_light_pass.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/screen_space_light_pass.cc)
- [onnx_backend.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/onnx_backend.h)

## 目标

这条链路的目标是把 MuJoCo 当前帧转换成一组和 `References/data` 风格接近的 raw EXR：

- 屏幕空间 G-buffer
- 屏幕空间 light clues
- 光源空间 RSM-like atlas

这些 EXR 是：

1. 训练前的数据输入
2. runtime tensor packing 的原始来源
3. 之后神经渲染推理时需要对齐的输入协议

## 输入契约

统一契约定义在 [input_contract.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/input_contract.h)。

全局约定：

- 坐标系：默认都是 **MuJoCo world space**
- 颜色和辐射量：默认都是 **linear**
- 文件格式：`RGBA32F`
- `w` 通道通常存 `mask / distance / reserved`

### 1. G-buffer

由 [GBufferPass](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/gbuffer_pass.h) 生成。

#### `Position.exr`

- `rgb`: world position
- `a`: valid surface mask

作用：

- 表示屏幕像素命中的世界空间位置
- 后续 `ssLightDepth / ssLightVec` 都依赖它

#### `Normal.exr`

- `rgb`: world unit normal
- `a`: valid surface mask

作用：

- 直接光、阴影、镜面、间接光都会用到

实现上：

- primitive 绘制时显式送法线
- world normal 在 shader 里归一化后输出

#### `Diffuse.exr`

- `rgb`: linear diffuse / albedo
- `a`: valid surface mask

作用：

- 材质本色输入
- 不是光照结果

#### `Reflect.exr`

- `rgb`: specular / reflect color
- `a`: valid surface mask

当前实现：

- 直接用 MuJoCo 的 scalar `geom.specular` 复制到 RGB
- 这是简化版，不是完整 PBR specular color

#### `Gloss.exr`

- `r`: `1 - roughness`
- `g/b`: reserved
- `a`: valid surface mask

当前实现：

- roughness 由 MuJoCo 的 `shininess` 做简化映射后再转成 gloss
- 这部分只是目前的工程近似，不是最终材质标准

#### `OutDir.exr`

- `rgb`: surface-to-camera unit direction
- `a`: valid surface mask

作用：

- specular / half vector / view-dependent shading 的输入

### 2. Screen-space Light Clues

由 [ScreenSpaceLightPass](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/screen_space_light_pass.h) 生成。

当前只支持：

- 取 `mjvScene.lights` 中第一个非 `headlight`、非 `directional` 的 local light
- `point`、`sphere_area` 和 `spot` 目前都走同一套 local-light atlas 逻辑
- `sphere_area` 是 `bulbradius > 0` 的 finite-radius point light approximation；raw clue 仍以光源中心为准，训练时需要把 `radius` 作为额外 light feature

#### `ssLightDepth.exr`

- `r`: occluder-to-light distance
- `g`: surface-to-light distance
- `b`: sampled light-space depth / reserved
- `a`: valid surface mask

作用：

- shadow decoder 的核心 raw clue
- `r` 和 `g` 后面可用于构造：
  - depth delta
  - hard shadow visibility
  - depth ratio

实现方式：

1. 先从光源位置渲一个 6-face light atlas
2. 用当前像素 `Position` 推出 `light -> surface` 方向
3. 按 cubemap face 规则采样 atlas
4. atlas 中 `smPosition.rgb` 给出光源方向上最近遮挡点的世界坐标
5. 当前像素自身到光源的距离直接由 `Position` 和 light position 计算

注意：

- 这张图 **不适合直接当普通图片看**
- 直接转 PNG 通常会因为距离范围偏大而发白
- 真正检查它时，应该分别看 `R/G` 通道或看它们的差值
- `b` 通道保留从 atlas 采到的 `smPosition.a`，也就是 Koo 代码里的 `sampledW`

#### `ssLightVec.exr`

- `rgb`: surface-to-light unit direction
- `a`: valid surface mask

作用：

- direct / shadow decoder 的入射光方向输入

### 3. Light-space Atlas

同样由 [ScreenSpaceLightPass](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/screen_space_light_pass.h) 生成。

这部分不是普通 2D camera image，而是 **6 个 light-view face 纵向拼起来的 atlas**：

- 每个 face 是一个 90 度视角
- face 顺序按 cubemap 方向
- 当前存储形式是 `face_size x (face_size * 6)`

#### `smPosition.exr`

- `rgb`: light-view visible world position
- `a`: light-to-surface distance

作用：

- light-space 几何位置
- `ssLightDepth.r` 会从这里采到的 `rgb` 和 light position 重新计算
- `ssLightDepth.b` 会保留这里采到的 `a` 通道，作为 `sampledW`

#### `smNormal.exr`

- `rgb`: light-view visible world normal
- `a`: valid surface mask

作用：

- RSM / indirect 输入
- 后续如果做 VPL sampling，会需要这里的法线

实现方式：

- 使用 light atlas pass 重新绘制 primitive
- primitive 绘制时必须显式提供法线

备注：

- 这里之前有过一个实现错误：绘制 box/sphere/plane 时没有送 `glNormal`
- 结果会导致 `smNormal` 几乎全部退化成 `(0,0,1)`
- 现在已经修正

#### `smFlux.exr`

- `rgb`: reflected flux
- `a`: valid surface mask

当前实现是近似量：

```text
flux = diffuse * light_color * intensity * max(dot(n, l), 0) / dist^2
```

这不是 path-traced GT，只是先把 RSM-like raw input 基础设施立起来。

作用：

- 为后续 indirect/VPL 提供能量近似

## 当前实现方式

### GBufferPass

[GBufferPass](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/gbuffer_pass.cc) 当前做的是：

1. 建一个 FBO
2. 挂 6 张 `RGBA32F` texture
3. 逐个绘制 MuJoCo scene 里的 supported geom
4. shader 里输出 `Position / Normal / Diffuse / Reflect / Gloss / OutDir`

当前支持的 geom：

- `mjGEOM_PLANE`
- `mjGEOM_BOX`
- `mjGEOM_SPHERE`
- `mjGEOM_MESH`

还不支持：

- `capsule`
- `cylinder`
- `ellipsoid`

mesh 当前走的是 baked geometry 路线：raw pass 从 `mjModel` 读取编译后的 `mesh_vert / mesh_face / mesh_normal` 绘制，snapshot 也把这份编译后几何写进 JSON，Blender/Cycles 直接从 snapshot 重建 mesh。这样做优先保证 MuJoCo raw 和 Cycles GT 几何对齐。

外部 OBJ/STL mesh 需要 MuJoCo 的 decoder plugin。NeLiF sample 工具会在 `mj_loadXML` 前自动尝试加载：

- `MUJOCO_PLUGIN_DIR`
- sample 可执行文件旁的 `../lib`
- sample 可执行文件旁的 `../mujoco_plugin` / `mujoco_plugin`
- 当前目录下的 `mujoco/build/lib` / `build/lib`

当前只主动加载 `libobj_decoder` 和 `libstl_decoder`，避免把整个 `build/lib` 里的动态库全部扫进去。也就是说，直接写 `<mesh file="xxx.obj">` 或 `<mesh file="xxx.stl">` 已经可以进入 NeLiF raw/snapshot 链路；texture/UV 仍然没有进入训练 contract。

### ScreenSpaceLightPass

[ScreenSpaceLightPass](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/screen_space_light_pass.cc) 分两步：

1. `RenderLightAtlas`
   - 从光源位置向 `+X/-X/+Y/-Y/+Z/-Z` 六个方向渲染
   - 输出 `smPosition / smNormal / smFlux`

2. 屏幕空间 full-screen pass
   - 读取 `Position`
   - 根据 `light -> surface` 方向选择 atlas face 和 UV
   - 生成 `ssLightDepth / ssLightVec`

### 导出

#### 交互调试

可用 sample：

- [nelif_gbuffer.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/sample/nelif_gbuffer.cc)

编译和运行：

```bash
cd /Users/aki/Projects/mujoco-pipeline/mujoco
cmake --build build --target nelif_gbuffer
./build/bin/nelif_gbuffer model/nelif_test/gbuffer_scene.xml
```

窗口里：

- `1..6` 看 G-buffer
- `7..11` 看 light-space 相关 debug
- `D` 导出当前帧 raw EXR

#### 一次性导出 raw EXR

可用工具：

- [nelif_export_raw.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/sample/nelif_export_raw.cc)

命令：

```bash
cd /Users/aki/Projects/mujoco-pipeline/mujoco
cmake --build build --target nelif_export_raw
./build/bin/nelif_export_raw model/nelif_test/gbuffer_scene.xml \
  --output-dir /Users/aki/Projects/mujoco-pipeline/tmp-res/export_raw_test \
  --width 512 \
  --height 512 \
  --face-size 1024
```

导出结果包括：

- `Position.exr`
- `Normal.exr`
- `Diffuse.exr`
- `Reflect.exr`
- `Gloss.exr`
- `OutDir.exr`
- `ssLightDepth.exr`
- `ssLightVec.exr`
- `smPosition.exr`
- `smNormal.exr`
- `smFlux.exr`
- `manifest.json`

#### 实时 runtime scaffold

可用 sample：

- [nelif_runtime.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/sample/nelif_runtime.cc)
- [runtime_pass.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/runtime_pass.h)
- [runtime_pass.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/runtime_pass.cc)
- [onnx_backend.h](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/onnx_backend.h)
- [onnx_backend.cc](/Users/aki/Projects/mujoco-pipeline/mujoco/src/experimental/nelif/onnx_backend.cc)

编译和运行：

```bash
cd /Users/aki/Projects/mujoco-pipeline/mujoco
cmake --build build --target nelif_runtime
./build/bin/nelif_runtime model/nelif_test/gbuffer_scene.xml
```

窗口里：

- `0`：MuJoCo 原始渲染
- `1`：`RuntimeShading`
- `2`：`RuntimeDirectUnshadowed`
- `3`：`RuntimeDirectShadowed`
- `4`：`PredShadow`
- `5`：`PredIndirect`
- `6`：`Position`
- `7`：`Normal`
- `8`：`ssLightDepth`
- `9`：`ssLightVec`
- `[` / `]`：调整显示 exposure
- `D`：导出当前 runtime 输出 EXR

第一版 runtime pass 的目标是先把实时渲染链路搭起来：

```text
GBufferPass + ScreenSpaceLightPass
-> RuntimePass
-> RuntimeDirectUnshadowed * PredShadow + PredIndirect
-> RuntimeShading
```

当前 `RuntimeDirectUnshadowed` 是 GLSL 里的解析 direct baseline，对齐 Python `tools/nelif/inference/direct_from_raw.py` 的简化 Lambert + Blinn-Phong + distance attenuation 逻辑。不启用 ONNX backend 时，`PredShadow` 由 `ssLightDepth` 派生 hard/soft visibility，`PredIndirect` 是 0 stub；启用后，`PredShadow` 和 `PredIndirect` 分别来自 exported shadow / indirect ONNX model。

ONNX backend 是可选编译项。Python `onnxruntime` wheel 通常只带 runtime 动态库，不带 C/C++ header；C++ sample 需要 ONNX Runtime C/C++ SDK：

```bash
cd /Users/aki/Projects/mujoco-pipeline/mujoco
cmake -S . -B build \
  -DMUJOCO_NELIF_ENABLE_ONNXRUNTIME=ON \
  -DMUJOCO_NELIF_ONNXRUNTIME_ROOT=/path/to/onnxruntime-osx-arm64
cmake --build build --target nelif_runtime
```

运行时传入 indirect 和 shadow ONNX：

```bash
./build/bin/nelif_runtime model/nelif_test/gbuffer_scene.xml \
  --indirect-onnx ../tmp-res/export_runtime/indirect_v1_128.onnx \
  --shadow-onnx ../tmp-res/export_runtime/shadow_v0_128.onnx \
  --onnx-provider coreml \
  --onnx-screen-size 128 \
  --onnx-rsm-face-size 64
```

`--onnx-provider` 支持 `coreml`、`cuda`、`tensorrt` 和 `cpu`。在 macOS + ONNX Runtime arm64 SDK 上使用 `coreml`；在 NVIDIA Linux 机器上优先测试 `cuda`，之后再测试 `tensorrt`。`tensorrt` 会先注册 TensorRT EP，再注册 CUDA EP 作为 fallback。MPS 是 PyTorch backend，这条 C++ ONNX Runtime 路径不直接使用 MPS。`cpu` 只建议作为诊断/数值对照路径。

当前 ONNX backend 是第一版对齐路径：CPU `glReadPixels/glGetTexImage` readback、固定 ONNX shape、ONNX inference、再上传 indirect / shadow texture 给 `RuntimePass` 采样。它用于确认 MuJoCo runtime tensor contract 和 Python exported model 能对齐，不代表最终实时性能方案。indirect 和 shadow 现在共用同一份 runtime input packing，避免每帧重复 readback/resize 同一批 G-buffer / RSM 输入。

可以打开 runtime profiler 看每帧分阶段耗时：

```bash
./build/bin/nelif_runtime model/nelif_test/gbuffer_scene.xml \
  --indirect-onnx ../tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx ../tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --onnx-provider coreml \
  --onnx-screen-size 128 \
  --onnx-rsm-face-size 64 \
  --profile-runtime \
  --profile-warmup-frames 15 \
  --exit-after-frames 90 \
  --profile-output ../tmp-res/runtime_profile.json
```

输出示例字段：

```text
total / sim / scene / gbuffer / light / input_pack / indirect_onnx / shadow_onnx / upload / compose / draw / swap_poll
```

`input_pack` 是共享 runtime tensor 构造时间；`profile.json` 还会写出 `input_pack_breakdown_ms`，拆成 `gbuffer_read`、`gbuffer_resize`、`screen_read`、`screen_resize`、`light_read`、`light_downsample`、`light_mask` 和 `finalize`。`indirect_onnx` 和 `shadow_onnx` 是 ONNX Runtime 推理时间。当前应用端主要看 macOS `coreml` 和 Linux NVIDIA `cuda/tensorrt`；`cpu` 只用于确认数值或 provider fallback 问题。`--profile-output` 会写出 `nelif.runtime_profile.v1` JSON，适合脚本化比较；`--exit-after-frames` 用于自动结束 benchmark。

批量 benchmark 可以用 Python 包装脚本：

```bash
python3 tools/nelif/inference/benchmark_runtime.py \
  --indirect-onnx tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --provider coreml \
  --screen-size 128 \
  --output-dir tmp-res/runtime_benchmark_coreml
```

脚本默认跑 `gbuffer_scene.xml`、`table_ball_drop.xml`、`table_arm_spin.xml`，如果本机存在 `tmp-res/pilot32/sample_000000/scene/scene.xml` 也会加入训练分布样本。macOS 默认 provider 是 `coreml`；Linux 默认 provider 是 `cuda`。输出包括每个 run 的 `profile.json`、`stdout.txt`、`stderr.txt`、首帧 dump，以及汇总的 `metrics.json` / `summary.md`。当前 ONNX export 是固定 shape，所以 `--screen-size` 必须和导出的 ONNX 输入尺寸一致；要比较 64/96/128，需要分别导出对应尺寸的 ONNX。

NVIDIA Linux 侧需要使用带 CUDA/TensorRT EP 的 ONNX Runtime C/C++ SDK，并确保 `libonnxruntime_providers_cuda.so` / `libonnxruntime_providers_tensorrt.so` 及 CUDA/cuDNN/TensorRT 动态库能被运行时加载。CUDA smoke：

```bash
python3 tools/nelif/inference/benchmark_runtime.py \
  --indirect-onnx tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --provider cuda \
  --screen-size 128 \
  --output-dir tmp-res/runtime_benchmark_cuda
```

TensorRT smoke：

```bash
python3 tools/nelif/inference/benchmark_runtime.py \
  --indirect-onnx tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --provider tensorrt \
  --screen-size 128 \
  --output-dir tmp-res/runtime_benchmark_tensorrt
```

更贴近训练分布的 runtime 场景可以直接加载已有 pilot sample：

```bash
./build/bin/nelif_runtime ../tmp-res/pilot32/sample_000000/scene/scene.xml \
  --key nelif_pose \
  --indirect-onnx ../tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx ../tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --onnx-screen-size 128 \
  --onnx-rsm-face-size 64 \
  --face-size 512
```

动态 smoke 场景：

```bash
./build/bin/nelif_runtime model/nelif_test/table_ball_drop.xml \
  --indirect-onnx ../tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx ../tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --onnx-screen-size 128 \
  --onnx-rsm-face-size 64 \
  --face-size 512

./build/bin/nelif_runtime model/nelif_test/table_arm_spin.xml \
  --key spin \
  --indirect-onnx ../tmp-res/export_runtime/indirect_v1_128_dynamo.onnx \
  --shadow-onnx ../tmp-res/export_runtime/shadow_v0_128_dynamo.onnx \
  --onnx-screen-size 128 \
  --onnx-rsm-face-size 64 \
  --face-size 512
```

可以用首帧导出做 smoke：

```bash
./build/bin/nelif_runtime model/nelif_test/gbuffer_scene.xml \
  --dump-first-frame \
  --exit-after-dump \
  --output-dir /Users/aki/Projects/mujoco-pipeline/tmp-res/nelif_runtime_smoke \
  --width 512 \
  --height 512 \
  --face-size 128
```

输出：

```text
RuntimeDirectUnshadowed.exr
RuntimeDirectShadowed.exr
PredIndirectStub.exr 或 PredIndirectShading.exr
PredShadow.exr
RuntimeShading.exr
```

`manifest.json` 会记录这次导出的关键 metadata：

- 实际使用的 camera
  - `requested_name`: 命令行 `--camera` 指定的名字，没有指定时为 `null`
  - `resolved_name`: 实际解析到的 MuJoCo camera，例如默认的 `nelif_fixed_cam`
  - `position / forward / up`
  - `vertical_fov_degrees`
  - `frustum`
- 实际使用的 active light
  - `type / type_name`
  - `position / direction`
  - `color / intensity / brightness`
  - `emission_rgb = color * brightness`
  - `radius / range / cone_angle_degrees`
- 每张 EXR 的尺寸、格式和通道语义

这些信息主要用于：

- 检查 MuJoCo raw input 和 Blender/Cycles GT 是否对齐
- 后续训练前做坐标系转换、深度重建或 tensor packing
- 复现实验导出参数

不要依赖模型自己修复 camera/light 不一致的问题；如果 paired data 的 pose 不一致，应该先在数据管线里修正。

如果要额外输出方便检查的派生调试图，可以加：

```bash
--write-debug
```

这会在输出目录下生成 `_debug/`：

- `ssLightDepth_delta.exr`
  - 灰度图
  - `max(surface_to_light - occluder_to_light - debug_bias, 0)` 归一化到 `[0, 1]`
  - 理论上会在被遮挡区域形成和阴影接近的团块
- `ssLightDepth_visibility.exr`
  - 黑白图
  - `1 = lit`
  - `0 = shadowed`
  - 使用同一个 debug bias 过滤 shadow map 自遮挡噪声；当前为 `0.02` scene units

这些文件只是检查工具，不属于训练 raw input contract。

#### Mesh 测试场景

当前仓库里有一个最小 mesh 测试场景：

```bash
cd /Users/aki/Projects/mujoco-pipeline/mujoco
./build/bin/nelif_export_raw model/nelif_test/mesh_scene.xml \
  --output-dir /Users/aki/Projects/mujoco-pipeline/tmp-res/mesh_raw \
  --width 512 \
  --height 512 \
  --face-size 1024 \
  --write-debug
./build/bin/nelif_export_snapshot model/nelif_test/mesh_scene.xml \
  --output /Users/aki/Projects/mujoco-pipeline/tmp-res/mesh_snapshot/scene_snapshot.json
```

这个场景里包含：

- 一个 inline MJCF mesh：`green_pyramid`
- 一个 box：`red_box`
- floor / back wall
- 一个 local point/spot light，可用 `bulbradius` 验证 finite-radius soft-shadow GT

它用于验证：

- G-buffer 能画 `mjGEOM_MESH`
- light-space atlas 能画 `mjGEOM_MESH`
- snapshot 能写出 baked mesh geometry
- Blender/Cycles 能从 snapshot 重建 mesh

外部 mesh smoke test：

```bash
cd /Users/aki/Projects/mujoco-pipeline
./mujoco/build/bin/nelif_export_snapshot mujoco/model/mug/mug.xml \
  --output tmp-res/external_mesh_probe/mug_snapshot.json
./mujoco/build/bin/nelif_export_snapshot References/DISCOVERSE/models/mjcf/manipulator/robot_panda.xml \
  --output tmp-res/external_mesh_probe/robot_panda_snapshot.json
```

已验证 `mug.xml` 的 OBJ mesh 和 DISCOVERSE Panda 的 OBJ/STL mesh 都能通过 decoder plugin 进入 snapshot。

## 已知限制

当前这条 raw/runtime 链路仍然是最小可用版本，不是完整 NeLiF 复现。

已知限制包括：

- 只支持 `plane / box / sphere / mesh`
- 只支持第一个 local light
- `spot` 目前没有完整 cone/cookie/attenuation 细节
- `smFlux` 是解析近似，不是 path-traced flux
- `Reflect / Gloss` 仍是 MuJoCo 材质到简化 PBR 参数的工程映射
- `ssLightDepth` 目前只把 raw depth clue 导出来，后续 tensor packing 才会再构造 delta / ratio / hard shadow
- C++ runtime 里的 ONNX indirect/shadow backend 已有第一版 CPU/readback scaffold；未启用 ONNX Runtime C/C++ SDK 时仍退回 `PredIndirectStub = 0` 和解析 shadow
- C++ runtime 里的 direct 是解析 baseline，还不是 direct neural decoder
- mesh 支持 compiled geometry，但不支持 texture / UV / per-face material 的完整导出

## 现在这条链路的定位

这部分不是最终 renderer，也不是 GT。

它的定位是：

1. 固定 MuJoCo 侧 stage 2 输入契约
2. 提供可导出的 raw/runtime EXR 数据
3. 给 Blender/Cycles GT 对齐、训练和 C++ neural backend 接入提供稳定输入和显示路径

## 后续实现思路：Paired Data Pipeline

下一步不应该直接开始训练模型，而是先把 MuJoCo raw input 和 Blender/Cycles GT 接成稳定的 **成对数据管线**。

原因是：

- 模型训练依赖严格对齐的 input/GT pair
- 如果 camera、几何、材质或光源在 MuJoCo 和 Blender 两侧有任何错位，模型会学习到错误映射
- 成对数据管线一旦稳定，后续 batch dataset、训练、评估和论文展示都可以复用同一个入口

### 目标目录结构

建议每个样本固定成下面的结构：

```text
sample_000001/
  input/
    Position.exr
    Normal.exr
    Diffuse.exr
    Reflect.exr
    Gloss.exr
    OutDir.exr
    ssLightDepth.exr
    ssLightVec.exr
    smPosition.exr
    smNormal.exr
    smFlux.exr
  gt/
    Shading.exr
    DirectShading.exr
    DirectShadowShading.exr
    IndirectShading.exr
    DiffuseDirectShading.exr
    SpecularDirectShading.exr
    DiffuseDirectUnshadowed.exr
    SpecularDirectUnshadowed.exr
    DiffuseIndirectShading.exr
    SpecularIndirectShading.exr
  snapshot/
    scene_snapshot.json
  debug/
    ssLightDepth_delta.exr
    ssLightDepth_visibility.exr
  manifest.json
```

其中：

- `input/` 来自 MuJoCo raw exporter
- `gt/` 来自 Blender/Cycles renderer
- `snapshot/` 是 MuJoCo 到 Blender 的场景交换文件
- `debug/` 只放人工检查用的派生图，不作为训练输入
- `manifest.json` 记录分辨率、face size、camera、frame、light-energy-scale、执行命令和版本信息

### 第一版 Dataset Driver

第一版 Python driver 已经放在：

- [export_dataset.py](/Users/aki/Projects/mujoco-pipeline/tools/nelif/export_dataset.py)
- [tools/nelif/README.md](/Users/aki/Projects/mujoco-pipeline/tools/nelif/README.md)

它的职责是编排已有工具，而不是重新实现渲染逻辑。当前会为每个 sample 生成一个随机 MJCF 场景，场景里包含：

- 一个有限半径 local point light，也就是 snapshot 里的 `sphere_area`
- 一个固定相机
- floor / back wall
- `primitive_arm` 模板里的简单机械臂和操作球/方块
- `discoverse_tasks_v1` 模板里的 DISCOVERSE Panda 外部 OBJ/STL mesh、桌面和任务物体
- 随机化的 light、camera、物体 pose/size 和材质参数

现在支持两个 scene template：

- `primitive_arm`
  - 最早的管线验证场景
  - 几何简单，适合 smoke test
- `discoverse_tasks_v1`
  - 参考 `References/DISCOVERSE/models/mjcf/task_environments`
  - 当前包含 `peg_in_hole / stack_block / close_laptop`
  - 使用 raw/snapshot/Cycles 都支持的 `plane / box / sphere / mesh` 几何重建任务布局
  - include DISCOVERSE Panda 资产，并为每个 sample 写 `scene/panda_pose.xml` 随机化 joint `ref`
  - 保留有限半径光源、随机相机、随机任务物体

只生成 XML、manifest 和命令，不实际渲染：

```bash
python3 tools/nelif/export_dataset.py \
  --output-root tmp-res/dataset_dry_run \
  --count 2 \
  --seed 42 \
  --dry-run \
  --skip-gt \
  --overwrite
```

只验证 MuJoCo 能加载随机场景并导出 snapshot：

```bash
python3 tools/nelif/export_dataset.py \
  --output-root tmp-res/dataset_snapshot_check \
  --count 1 \
  --seed 42 \
  --skip-raw \
  --skip-gt \
  --overwrite
```

生成一个小规模 smoke dataset：

```bash
python3 tools/nelif/export_dataset.py \
  --output-root tmp-res/nelif_dataset_smoke \
  --count 20 \
  --seed 1000 \
  --width 512 \
  --height 512 \
  --face-size 1024 \
  --cycles-samples 256 \
  --light-energy-scale 64 \
  --debug-raw \
  --overwrite
```

生成 DISCOVERSE-inspired V1 输入集：

```bash
python3 tools/nelif/export_dataset.py \
  --output-root tmp-res/discoverse_v1_inputs \
  --count 30 \
  --seed 2000 \
  --scene-template discoverse_tasks_v1 \
  --task-variant auto \
  --skip-gt \
  --debug-raw \
  --overwrite
```

内部步骤：

1. 写 `sample/scene/scene.xml`
2. 调 `nelif_export_raw`
   - 输出 MuJoCo raw inputs 到 `sample/input/`
   - 如果开启 debug，把 `_debug` 内容移动或复制到 `sample/debug/`
3. 调 `nelif_export_snapshot`
   - 输出 `sample/snapshot/scene_snapshot.json`
4. 调 `tools/blender/render_snapshot_cycles.py`
   - 读取 snapshot
   - 输出 Cycles GT 到 `sample/gt/`
5. 写 `sample/manifest.json`
   - 记录所有参数和输入模型路径
   - 记录 MuJoCo raw exporter、snapshot exporter、Blender renderer 的命令
   - 记录光照能量标定参数

第一版只需要支持：

- 每个 sample 一个随机生成 XML
- 单 camera
- 单 local light
- `plane / box / sphere / mesh`

后续再扩展到多帧、多 camera、多场景和 batch config。

### 必须先做的对齐检查

生成第一批 paired sample 后，先不要训练，先检查几何和光照是否一致：

- `input/Position.exr` 和 `gt/Position.exr` 的前景 mask 是否重合
- `input/Normal.exr` 和 `gt/Normal.exr` 的方向是否一致
- MuJoCo camera 和 Blender camera 的 FOV、pose、分辨率是否一致
- 物体位置、尺寸、旋转是否一致
- light position 是否一致
- light brightness / emission_rgb 是否一致
- `light-energy-scale` 是否让 GT 的亮度落在合理范围
- `DirectShading.exr` 是否确实是无遮挡 direct result
- `DiffuseDirectUnshadowed.exr` / `SpecularDirectUnshadowed.exr` 是否相加等于 `DirectShading.exr`
- `DirectShadowShading.exr` 是否保留了 Cycles 阴影
- `ssLightDepth_delta.exr` 的阴影团是否和 GT shadow 大致对应

这一步最好也输出几张 normalized preview PNG，但 preview 只用于检查，不要替代 EXR raw data。

### 训练前的数据边界

在 paired data pipeline 稳定前，暂时不做下面这些事：

- 不急着做大规模数据集
- 不急着训练最终 NeLiF 模型
- 不急着接 runtime inference
- 不把 raw EXR 为了显示方便强行归一化

可以先做一个很小的 sanity dataset，例如 10 到 50 个样本，确认：

- input 和 GT 能稳定生成
- EXR channel 命名和 shape 一致
- mask 对齐
- direct / shadow / specular pass 没有明显错位

确认这些都成立后，再进入最小模型实验。
