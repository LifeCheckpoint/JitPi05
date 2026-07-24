# JitPi05

一个尽量简洁的分层 VLA 研究原型：

- **Qwen3.5-4B**：读取 LIBERO 双相机首帧，生成当前高层 subtask。
- **显式 logits 调制**：逐 token 保存原始/调制后 logits，并在 argmax 前暴露可编辑回调。
- **LeRobot π₀.₅ LIBERO**：把总任务与 subtask 组成低层语言条件，显式执行 prefix 编码、KV cache、10 步 flow-matching 去噪和反归一化。
- **四组对照**：原始任务、人工 subtask、Qwen subtask、调制后的 Qwen subtask。

代码只按研究逻辑拆成几个平铺模块，没有配置框架、服务层或实验管理系统：

- [`settings.py`](settings.py)：模型、数据集和实验常量。
- [`data.py`](data.py)：单个 LIBERO episode 的读取与图像转换。
- [`planner.py`](planner.py)：Qwen3.5 高层规划、逐 token logits 记录与调制。
- [`pi05_pipeline.py`](pi05_pipeline.py)：显式 π₀.₅ 流水线、四组对照和动作指标。
- [`main.py`](main.py)：单帧离线实验入口。
- [`sim_eval.py`](sim_eval.py)：固定初始状态、闭环 action chunk、成功判定、视频和评测产物。
- [`eval_sim.py`](eval_sim.py)：独立仿真评测入口。

## 离线分析

要求 NVIDIA CUDA GPU，建议至少 16 GB 显存。两个模型顺序加载，不同时驻留显存。

```bash
uv sync
uv run main.py
```

第一次运行会下载：

- `Qwen/Qwen3.5-4B`
- `lerobot/pi05_libero_base`
- `HuggingFaceVLA/libero` 的 episode 0
- 一个与官方 PaliGemma 相同的非 gated tokenizer 镜像

官方 `google/paligemma-3b-pt-224` tokenizer 仓库需要先接受 Gemma 许可并登录 Hugging Face。为了让本实验的 tokenizer 阶段不依赖登录，默认使用 `nnh-pbbb/paligemma-3b-pt-224`；其 Gemma tokenizer 词表大小、BOS/PAD ID 与官方版本一致。若你已经获得官方访问权，可直接修改 `PI05_TOKENIZER_ID`。

## 最重要的研究入口

### 高层 prompt

修改 [`planning_prompt()`](planner.py:13)。当前要求 Qwen 只输出一个立即执行的英文动词短语，并关闭 thinking。

### logits 调制

修改 [`make_phrase_bias()`](planner.py:78)，或者直接向 [`generate_subtask()`](planner.py:105) 传入自己的回调：

```python
def modifier(step, input_ids, logits):
    # logits: [batch, vocab_size]
    logits[:, target_token_id] += bias
    return logits
```

当前示例依次提升 `BIAS_PHRASE` 中各 token 的 logit。它只是验证“调制会传导到规划文本和 π₀.₅ 动作”的最小实验，不是完整控制算法。

逐 token 信息保存在：

- `artifacts/planning_logits.pt`：完整原始/调制后 logits，使用 bfloat16 保存。
- `artifacts/summary.json`：top-k token、概率、最终 token 与规划文本。

### π₀.₅ 显式流水线

[`explicit_pi05_flow()`](pi05_pipeline.py:67) 没有调用 `select_action()` 或黑盒 `predict_action_chunk()`，而是显式执行：

1. LeRobot preprocessor：批维、数据集统计归一化、状态离散化、PaliGemma tokenization、搬到 CUDA。
2. π₀.₅ 图像预处理与 VLM prefix embedding。
3. prefix attention mask、position IDs 与 KV cache。
4. 固定相同初始噪声的 10 步 flow-matching 去噪。
5. 裁剪到 LIBERO 7 维动作。
6. LeRobot postprocessor 反归一化并搬回 CPU。

函数同时保存每一步 action trajectory 和 vector field，方便后续研究低层去噪过程。

## 四组条件

同一首帧、同一 π₀.₅ 权重、同一初始噪声下运行：

1. `original_task`：原始完整任务。
2. `oracle_subtask`：人工提供 `pick up the white mug`。
3. `qwen_subtask`：未调制的 Qwen 规划结果。
4. `modulated_qwen_subtask`：经过 logits 调制的 Qwen 规划结果。

如果切换 episode，应同步修改 `EPISODE_INDEX` 和 `ORACLE_SUBTASK`。

## 输出

- `artifacts/summary.json`
  - 两次 Qwen 规划文本
  - 逐 token top-k
  - 四组实际语言条件
  - 动作数值统计、示范 MAE/RMSE、相对原始任务输出的 RMSE
- `artifacts/planning_logits.pt`
  - 完整 Qwen logits
- `artifacts/pi05_actions.pt`
  - 四组反归一化动作 chunk
  - episode 0 示范动作 chunk
  - 固定初始噪声
  - π₀.₅ 去噪 trajectory、vector fields 与 prefix tokens

## LIBERO 闭环评测

闭环入口只支持 Linux。无桌面服务器建议使用 EGL：

```bash
MUJOCO_GL=egl uv run eval_sim.py
```

第一次导入 LIBERO 时，上游包会询问数据目录；直接选择默认路径即可。评测模型为当前 LeRobot 0.6 / Transformers 5 兼容的 `lerobot/pi05-libero`，与离线分析使用的 base checkpoint 分开配置。

默认运行两个任务，每个任务 5 个固定初始状态，并在同一初始状态上配对运行四组语言条件：

1. `libero_object` task 0：`pick up the alphabet soup and place it in the basket`。这是 π₀.₅ LIBERO 微调分布内的闭环基准，用于确认环境、processor、动作空间和 checkpoint 均正常。
2. `libero_90` task 79：`pick up the book and place it in the left compartment of the caddy`。LIBERO-90 未进入该 π₀.₅ LIBERO checkpoint 的微调任务集合，因此这里把它作为任务级零样本探针，而不是宣称跨 embodiment 的通用零样本控制。

高层 Qwen 只在每个 episode 的初始双相机观测上规划一次，然后释放显存。低层 π₀.₅ 显式预测 50 步 action chunk，只执行前 10 步，再用新观测重新预测。四组条件在同一 episode 的第 n 次重规划中复用相同 flow-matching 初始噪声。

仿真产物保存在 `artifacts/sim_eval/`：

- `summary.json`：逐条件 success rate、每个 episode 的 success/reward/steps、Qwen 文本和 token top-k。
- `planning_logits.pt`：所有 episode 的原始/调制后完整 Qwen logits。
- `rollouts.pt`：每次预测的完整 action chunk、实际执行动作和 reward。
- `videos/<task>/<condition>/episode_<id>.mp4`：逐条件 rollout 视频。

完整默认评测共 `2 tasks × 5 episodes × 4 conditions = 40` 个 rollout，并包含 10 次 Qwen 规划对，耗时会明显长于离线 sanity check。可直接修改 [`SIM_EPISODES`](settings.py:16)、[`SIM_ACTION_STEPS`](settings.py:17) 和 [`SIM_TASKS`](settings.py:19) 缩小实验。

## 解释限制

[`main.py`](main.py) 仍是**单帧离线 sanity check**，用于回答：

- Qwen 是否能输出可读的高层动作？
- logits 调制是否真的改变 token 分布和最终 subtask？
- 高层文本变化是否传导到 π₀.₅ 的低层动作？
- π₀.₅ 输出是否有限、尺度是否正常、是否与示范动作处在相近范围？

示范动作 MAE/RMSE 不是策略成功率，也不能单独证明动作语义正确；闭环成功率应以 [`eval_sim.py`](eval_sim.py) 的 LIBERO rollout 为准。LIBERO-90 task 79 的结果是单任务诊断探针，不应外推成整个 LIBERO-90 或真实机器人上的普遍零样本能力。
