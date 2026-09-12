# CycleVLA 官方实现对照与复现差距分析

> 结论先行：当前实现是「零训练推理包装」，而论文核心是「**训练出 9 维 subtask-aware policy**」。
> 在不训练 9 维策略的前提下，stop/progress 信号只能靠 fixed-horizon 代理，VLM 触发时机与物理进度脱节；
> 同时当前用 Qwen 自由文本作为低层条件（condition），与 7D policy 的训练提示分布不一致，这是「抓错物体」的深层根因。
> 参考仓库：`https://github.com/dannymcy/cyclevla_code`（已克隆至 `.reference_cyclevla/`）。

---

## 1. 官方参考实现要点（.reference_cyclevla）

### 1.1 核心推理管线（MBR 版主循环）

官方主脚本：[`run_libero_eval_decomposed_progress_mbr.py`](../.reference_cyclevla/experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py)

- **策略输出 9 维 action**：`[6D EEF delta, gripper, stop s_t, progress p_t]`，`env.step` 只执行 `action[:-2]`，后两维仅作信号。
- **子任务由确定性 FSM 生成**（非 VLM 分解）：
  - pick-place 模板 4 态：`Move the gripper above {obj}.` → `Close the gripper to grasp {obj}.` → `Move the gripper above {place} while holding {obj}.` → `Open the gripper to release {obj}.`
  - 复杂任务用 `fsm_complex_libero` 硬编码字典（`fsm_utils/build.py`）。
  - 子任务文本来自**固定模板**，与训练数据 `libero_decomposed_progress` 的 prompt 分布**一致**。
- **两阶段子任务状态机**（每个 subtask 内）：
  - `to_check`：监控 `progress_signal >= 0.90`，连续 2 次或 gap≥2 重确认后触发 VLM 检查；
  - `to_complete`：监控 `termination_signal == -1`（stop），连续 2 次或重确认后判定子任务完成并推进。
- **跳过 gripper 子任务的 VLM 检查**：`close the gripper to grasp ...` / `open the gripper to release ...` 直接进入 `to_complete`。
- **VLM 检查 @ ~90%**：`VLMDetector(model_name="gpt-5.2")`，双视角 FRONT/WRIST，输出 `next_subtask / type(transit|backtrack) / reason`。
  - backtrack 目标必须是 `subtasks` 列表中的**精确文本**；
  - 明确要求回溯到「恢复缺失前置条件的**最早** subtask」（reach/align/target-identification），**禁止**回退到 trivial 的 open/close gripper。
- **backtrack 物理回退**：`restore_robot_only`（保存完整 sim state，只恢复 robot qpos/qvel/mocap，保持物体位置）+ 逐快照 reverse-replay（`backtrace_robot_states`）。
- **MBR 重试**（回溯后）：采样 **8 个 seed**（`original_seed + i*10`）重新推断完整动作 chunk → 用动作增量**预测状态轨迹**（6D eef pos+euler）→ `cdist` 距离矩阵 → **r-NN 密度估计（pocket/medoid）+ 失败轨迹 repulsion** 排序 → 用代表 seed **重新 rollout 整个子任务**（不是只执行一个 chunk）。
- **retry 上限 per-subtask**：`max_retries=3`，超限后**强制 `to_complete`**（强制推进，不停止）。
- **LIBERO-90 max_steps = 400**（`TASK_MAX_STEPS`）。

### 1.2 训练侧（论文效果的根基）

- 数据集：`libero_decomposed_progress`（Stage-3 RLDS，9 维动作 `[6D, gripper, s_t, p_t]`，`is_last → stop`，`is_terminal → progress`）。
- 微调：从 openpi 官方 **`pi05_libero` checkpoint** 继续微调（不是 `pi05_base`），学习两个新监督维与 subtask 级 prompt 分布；`prompt_from_task=True`，prompt = `Task: {task}. The current subtask: {state}`。
- OpenVLA-OFT 版：`libero_sub_decomposed_progress_A100` 检查点，同样基于模板子任务训练。

### 1.3 openpi 9D 接口

- [`libero_policy.py`](../.reference_cyclevla/openpi/src/openpi/policies/libero_policy.py)：`LiberoOutputs.action_dim=9`，保证 stop/progress 两维存活到推理。
- [`openpi_utils.py`](../.reference_cyclevla/experiments/robot/openpi_utils.py)：openpi gripper 保持 raw RLDS（不 invert）；stop/progress 为 raw float（`stop>0.5`、`progress>=0.90` 直接阈值）。

---

## 2. 当前实现（JitPi05）与官方逐项差距

| # | 维度 | 官方 CycleVLA | 当前 JitPi05 | 差距等级 |
|---|------|--------------|--------------|---------|
| 1 | 策略 | **9D subtask-aware policy**（训练出 stop/progress） | **7D stock** `lerobot/pi05_libero_base`，无 learned signal | **🔴 根本** |
| 2 | stop/progress 信号 | 模型预测，物理进度对齐 | fixed-horizon 代理：`CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS=3`、`CYCLE_PROXY_PROGRESS_THRESHOLD=0.75` | 🔴 根本 |
| 3 | 子任务程序来源 | 确定性 FSM 模板 + 规则解析（文本稳定，与训练分布一致） | `build_cycle_subtask_program` 正则 fallback + **Qwen 自由规划** + `match_cycle_subtask` 文本/action-type 近似匹配 | 🔴 根本（condition 与训练分布不匹配） |
| 4 | 喂给低层策略的 condition | 固定模板文本 `Task: ... The current subtask: ...` | `conditioned_task(overall_task, qwen_free_text)`，自由文本 | 🔴 根本 |
| 5 | 训练流程 | 有（数据生成 + 微调 + norm stats） | **无**（零训练推理包装，不改变 JitRL 边界） | 🔴 根本 |
| 6 | VLM 触发时机 | ~90% progress（物理信号确认） | fixed-horizon 3 chunks 代理触发 | 🟠 大 |
| 7 | VLM 模型 | GPT-5.2（OpenAI） | Gemini（pydantic_ai_google，`JITRL_EVALUATOR_MODEL`） | 🟠 大 |
| 8 | gripper 子任务特判 | 跳过 VLM 检查，直接 to_complete | 无此特判（每个 anchor 都做 CHECK） | 🟠 大 |
| 9 | MBR 语义 | 8 seed 重 rollout 整个子任务，r-NN 密度 + failed repulsion | 8 个 flow-noise chunk，`select_mbr_chunk` **只选一个 chunk 执行**（不重 rollout 整个 subtask，无 seed 级重采样、无 failed repulsion） | 🟠 大 |
| 10 | retry 超限行为 | 超 3 次**强制 to_complete**（推进） | per-target retry 上限存在，超限行为未见强制推进 | 🟡 中 |
| 11 | backtrack 目标语义 | 精确文本；最早恢复前置条件；禁止 trivial gripper | program ID 匹配；`select_recovery_anchor` 要求严格更早；已禁 self-target（较官方更严） | 🟡 中 |
| 12 | 物理回退 | restore_robot_only + reverse-replay | `rewind_robot_configuration_history` 平滑历史恢复（方向一致） | 🟢 对齐 |
| 13 | episode budget | LIBERO-90 = 400 | `unified_800`（`CYCLE_UNIFIED_MAX_STEPS=800`） | 🟡 中（预算不同影响判定） |
| 14 | 评估协议 | 两阶段：transit 全跑 → 对失败 episode 跑 full method（或 `--rerun_all True`） | cycle 与 baseline 独立跑后配对 | 🟡 中 |
| 15 | VLM 决策门控 | `type==backtrack` 即回溯（无 likelihood 门槛） | `CYCLE_VLM_BACKTRACK_LIKELIHOODS=("low",)`（仅 low 才回溯，更保守） | 🟡 中 |

---

## 3. 为什么「依然复现不出来」——三层根因

### 3.1 第一层：没有训练 9D subtask-aware policy（不可推理侧修复）

论文的收益来自「策略自己学会在 ~90% 时预测 progress、在完成时预测 stop」。
当前用 7D stock 策略 + fixed-horizon 代理：
- VLM 触发时机是**步数代理**（3 chunks），与真实物理进度脱节；
- 没有 stop 信号，子任务「完成」只能靠 fixed-horizon 或 VLM 判断，无法精确推进。
- 这一层**无法在推理侧修复**，只能退化为「降级模式」。

### 3.2 第二层：condition 与训练分布不一致（推理侧可对齐，当前最关键）

官方给低层策略的 prompt 是**确定性模板文本**（与训练分布一致）。当前是 **Qwen 自由文本 + `conditioned_task` 拼接**：
- 7D policy 是在标准 LIBERO prompt（如 `pick up the red block and place it on the wooden cabinet`）上训练的；
- 当前用「Qwen 生成的自由语义动作」作为条件，prompt 分布漂移 → 策略行为退化 → 抓错物体、错误对齐。
- **这一层可以在推理侧对齐**：把子任务程序改为官方式确定性 FSM 模板文本，condition 也改为模板形式。

### 3.3 第三层：MBR 语义不对（推理侧可对齐）

官方 MBR = 回溯后**重采样 8 个 seed 重新 rollout 整个子任务**，选代表轨迹。当前 MBR = 选**一个 chunk** 执行一次。两者在「重试能否真正把子任务做对」上有本质区别——单 chunk 修正无法保证整个 grasp/place 子任务成功。

---

## 4. 对齐优先级建议（P0 已全部修复，2026-08-09）

### P0（推理侧，已修复，不动训练）

1. **子任务程序模板化** ✅：`build_cycle_subtask_program` 已改为官方式确定性模板（pick-place 四态 + `fsm_complex_libero` 复杂任务字典，见 `cycle.py` 的 `_extract_pick_place_objects` / `_COMPLEX_TASK_STATES`）；新增 `format_cycle_condition`，condition 统一为 `Task: {task}. The current subtask: {state}` 模板，rollout 的 anchor 创建与回溯后 replan 均改用模板 condition（不再注入 Qwen 自由文本）。
2. **MBR 改为「重 rollout 整个子任务」** ✅：回溯后 `_run_mbr_retry` 记录 MBR 选中的 seed 坐标基址（`mbr_rollout_coordinate`），主循环恢复期间沿用该基址滚动生成后续 chunk（`mbr_rollout_noise`），实现「多 chunk 滚动执行」而非单 chunk 修正。
3. **gripper 子任务特判** ✅：CHECK 阶段对 `grasp/release` 类 anchor 跳过 VLM 预测（`gripper_subtask` 标记），保留物理证据检查。
4. **retry 超限强制推进** ✅：CHECK 阶段对已达 `CYCLE_MAX_RETRIES` 的 anchor 跳过 VLM（`retry_limit_reached` 标记），`resolve_cycle_decision` 返回 transit 强制推进。
5. **子任务推进信号** ✅：COMPLETE 阶段新增物理证据代理 stop（`explicit_success` 满足即提前 `_finish_active_subtask`），并新增指标 `cycle_proxy_stop_finish_count`。

### P1（推理侧增强，已实现，保留）

- 回溯后重新高层规划（修复1）、self-target 禁止（修复2）、program/anchor/物体 identity 绑定（修复3）、simulator object-level evidence（修复4）——官方没有但方向正确，可保留为消融。

### P2（训练侧，才能真正复现论文）

- 生成 `libero_decomposed_progress` 数据（含 stop/progress 监督 + 模板子任务 prompt）；
- 从 `pi05_libero` 微调 9D policy（或使用作者已发布的 9D checkpoint）；
- 用官方两阶段评估协议（transit baseline → full method）在同一 budget/seed 下对比。

---

## 5. 需要向作者确认 / 官方仓库未公开点

- 官方发布的具体 checkpoint（HF/GS 链接未在 README 中给出，仅给出训练命令与路径约定）。
- MBR `num_seeds=8` 是硬编码默认，论文是否用该值作为主结果。
- VLM 决策是否有 likelihood 门槛：官方代码**无门槛**（`type==backtrack` 即回溯），当前 `("low",)` 更保守，可能是效果差异来源之一（建议对齐为官方语义后再对比）。

---

## 6. P0 修复后首轮实验（v6, task79, seed 17, 10 eps）结果与诊断

### 6.1 配对结果

| 指标 | jitrl-free (baseline) | jitrl-free-cycle | 配对差 |
| --- | --- | --- | --- |
| success_rate | **0.5** (5/10) | **0.1** (1/10) | **-0.4** |
| 成功 episode | ep2,3,4,6,9 | ep9 | harm=4, rescue=0 |
| 双双成功 | — | — | 1 (ep9) |
| 双双失败 | — | — | 5 |

- **harm=4, rescue=0**：Cycle 仍系统性损坏 baseline 能成功的 4 个 episode（ep2,3,4,6），且未救回任何 baseline 失败的 episode。与 v5（baseline 7/15, cycle 0/15, harm=7）相比，harm 比例 46.7%→40% 仅微降。

### 6.2 P0 修复生效情况

- ✅ **self-target**：实际 0 次（请求 13 次全部被拒），rate 10.7%（v5 39.7%）——修复2 有效。
- ✅ **gripper 子任务特判**（P0-3）：14 次跳过 VLM。
- ✅ **retry 超限强制推进**（P0-4）：4 次触发。
- ✅ **replan**（P0-1/修复1）：26/28 次回溯触发 replan，0 次 planner 错误。
- ❌ **replan 一致性差**：`consistent_with_target` 仅 **8/26 ≈ 31%**——18 次回溯后 replan 选出与回溯目标不同的程序节点（例如回溯到 approach 却 replan 出 release），condition 与场景错配仍是失败主因之一。
- ❌ **P0-5 proxy stop 未生效**：`cycle_proxy_stop_finish_count = 0`。根因：`_physical_evidence` 的 `postcondition_satisfied` 只绑定**全局** robosuite success（`success or _robosuite_success`），子任务进行中恒为 False，`explicit_success` 永远不成立，物理证据代理 stop 实际从未触发。

### 6.3 剩余根本差距（与最初问题一致）

- **抓错物体仍是最主要失败信号**：`gripper holds 'red coffee mug' but requested target is 'black book'` 出现 **8 次**（task79 目标黑书，策略抓了红咖啡杯）。物理证据已正确检测到（`target_identity_ok=False`），但回溯+MBR 未有效纠正 → 7D stock policy 在 grasp 阶段本身就抓错，Cycle 只能事后发现、无法事前预防。
- **触发时机仍是 fixed-horizon**：121 次检查全部 `fixed_horizon_degraded`，无 learned progress → VLM 检查与物理进度脱节。
- **MBR 仍非完整复刻**：medoid 单块滚动，未做官方 r-NN 密度 + failed repulsion 轨迹级重跑。

### 6.4 已实施修复（2026-08-09，v7 前）

1. ✅ **P0-5 语义修复**：新增 [`proxy_subtask_stop()`](src/jitpi05/jitrl/cycle.py) 纯函数，把「子任务完成」判定从「全局任务 success」改为「按动作类型的子任务级物理达成」——grasp/lift 看 `object_following_gripper`/`target_identity_ok`、transport 看 `destination_reached`、place/release 看 `released`/`destination_reached`、approach/align 看 `target_identity_ok`，事实未知返回 None。COMPLETE 阶段（[`rollout.py`](src/jitpi05/jitrl/rollout.py)）改用该函数替代 `explicit_success`，使 7D 代理能按物理动作真正达成而提前推进子任务（v6 中 `cycle_proxy_stop_finish_count=0` 的根因已消除）。
2. ✅ **replan 一致性修复**：回溯后 replan 若与回溯目标程序节点不一致（`replan_program_id != target_anchor.program_id`），强制丢弃 replan 的自由 condition，回退到回溯目标节点的官方模板 condition（`format_cycle_condition(overall_task, cycle_program[target_anchor.program_position].text)`），即「重做目标子任务」；recovery_event 新增 `forced_alignment` 标记。官方本身就是直接沿用回溯目标模板重 rollout，不自由 replan。
3. 抓错物体问题属**策略本身**（7D 无 subtask 感知），仅靠推理侧 Cycle 无法根治——仍指向训练侧 9D subtask-aware policy（P2）。

### 6.5 v7 结果与新根因（2026-08-09）

**v7（task79, seed 17, 10 eps, 仅 cycle）：success_rate = 0/10**（v6 为 1/10）。所有 episode 均 800 步耗尽。

- ✅ 两个修复均生效：`cycle_proxy_stop_finish_count` 0→**13**（P0-5 子任务级物理判定开始触发）；replan `forced_alignment` 出现 **27 次**（不一致时强制对齐目标节点）。
- ❌ 但成功率未提升反而微降。**新根因**：`SemanticAnchor.action_type` 此前用 **Qwen 自由文本分类**（`semantic_action_type(selected_candidate)` → "place"），而程序节点是模板类型（transport/release）——v7 中 **29 次 transport 被误标成 place、17 次 release 被误标成 place、4 次 grasp 被误标成 place**。这使 `proxy_subtask_stop`/`physical_failure_signal`/gripper 特判全部按错误动作语义判断（place 分支要求 released/destination，transport 阶段物体仍在跟随 → 判定逻辑错乱）。
- ✅ **已修复**（2026-08-09）：anchor 的 `action_type` 改为使用 `program_node.action_type`（模板类型 approach/grasp/transport/release），与官方模板文本语义一致。
- 抓错物体（`red coffee mug` vs `black book`）仍出现 17 次——这是 7D 策略本身行为，物理证据已检测但回溯+MBR 未根治。

---

## 7. 最终全面对照表（官方 CycleVLA vs 本实现，2026-08-09）

图例：✅ 一致 / ⚠️ 部分一致 / ❌ 不一致 / ➕ 本实现独有（官方没有）

| # | 维度 | 官方 CycleVLA | 本实现 | 状态 |
|---|------|--------------|--------|------|
| 1 | 策略动作空间 | 9 维 `[6D EEF, gripper, stop s_t, progress p_t]` | 7D stock `pi05_libero_base`（`adapt_cycle_policy_output` 支持 9D 但实际无） | ❌ |
| 2 | stop/progress 信号 | **learned**，物理进度对齐 | fixed-horizon 代理：`CYCLE_PROXY_PROGRESS_THRESHOLD=0.75`、3 chunks 触发 | ❌ |
| 3 | 子任务程序 | 确定性 FSM 模板（pick-place 4 态 + `fsm_complex_libero` 字典） | [`build_cycle_subtask_program()`](src/jitpi05/jitrl/cycle.py) 官方式模板（`_extract_pick_place_objects` + `_COMPLEX_TASK_STATES`） | ✅ |
| 4 | 低层 condition | `Task: {task}. The current subtask: {state}`（prompt_from_task） | [`format_cycle_condition()`](src/jitpi05/jitrl/cycle.py) 同模板；anchor 创建与 replan 均用模板文本 | ✅ |
| 5 | VLM 模型 | GPT-5.2（OpenAI，`type==backtrack` 即回溯、无 likelihood 门槛） | Gemini（pydantic_ai_google）+ `CYCLE_VLM_BACKTRACK_LIKELIHOODS=("low",)` 更保守 | ⚠️ |
| 6 | VLM 触发时机 | ~90% progress（learned 信号确认，连续 2 次或 gap≥2） | fixed-horizon 3 chunks 代理触发（`fixed_horizon_degraded`） | ❌ |
| 7 | 双视角 VLM 提示 | FRONT/WRIST 双视角、transit 默认、backtrack 精确目标 | [`cycle_failure_predictor_prompt()`](src/jitpi05/jitrl/vlm.py) FRONT/WRIST、transit 默认、程序 ID 校验 | ✅ |
| 8 | gripper 子任务特判 | 跳过 VLM 检查，直接 to_complete | `is_gripper_subtask` 跳过 VLM（保留物理检查） | ✅ |
| 9 | retry 上限 | per-subtask，超限强制 to_complete | per-target `CYCLE_MAX_RETRIES`，`retry_limit_reached` 跳过 VLM 并强制 transit | ✅ |
| 10 | 物理回退 | `restore_robot_only`（只回退机器人）+ reverse-replay | `rewind_robot_configuration_history` 平滑历史恢复（`reverse_robot_state_replay`） | ✅ |
| 11 | 回溯目标选择 | 精确文本；最早恢复前置条件；禁止 trivial gripper | `select_recovery_anchor`：program ID 匹配 + 严格更早（`<`）+ 禁 self-target + 同物体优先 | ✅（更严格） |
| 12 | 回溯后行为 | **直接重做回溯目标子任务**（沿用模板 condition，不自由 replan） | replan 后**强制对齐目标节点**（`forced_alignment`，不一致时回退目标模板 condition） | ✅ |
| 13 | 子任务推进 | learned stop 信号推进 | [`proxy_subtask_stop()`](src/jitpi05/jitrl/cycle.py) 子任务级物理完成判定（抓稳/到位/释放） | ⚠️（物理代理替代 learned） |
| 14 | MBR | 8 seed 重 rollout 整个子任务 + r-NN 密度 + failed repulsion | 8 候选 medoid 选 chunk + `mbr_rollout_coordinate` 滚动执行后续 chunk | ⚠️ |
| 15 | episode budget | LIBERO-90 = 400 | `unified_800`（`CYCLE_UNIFIED_MAX_STEPS=800`） | ❌ |
| 16 | 训练侧 | 有（`libero_decomposed_progress` 数据 + 从 `pi05_libero` 微调 9D） | **无**（零训练推理包装，不改 JitRL 边界） | ❌ |
| 17 | 评估协议 | 两阶段：transit baseline → full method 重跑失败 episode（或 `--rerun_all True`） | cycle 与 baseline 独立跑后配对（harm/rescue） | ⚠️ |
| 18 | 回溯后重规划审计 | 无 | `recovery_event.replanned`（triggered/consistent/forced_alignment/condition） | ➕ |
| 19 | 物体级物理证据 | 无（官方仅 restore robot） | `SceneObject`/`target_identity_ok`/`object_following_gripper`/`destination_reached`/`failure_reasons` | ➕ |
| 20 | program/anchor/物体绑定 | 无 | `SemanticAnchor.target_object` + 同物体恢复偏好 + `match_cycle_subtask` | ➕ |
| 21 | 代理 stop 审计指标 | 无 | `cycle_proxy_stop_finish_count`、`cycle_self_target_request_count/rate`、`cycle_replan_count/rate` | ➕ |

**结论**：推理侧状态机已与官方对齐（子任务模板、双视角 VLM、gripper 特判、retry 超限、物理回退、回溯后重做目标），并新增官方没有的物体级证据与审计；**核心不一致仍是训练侧**（9D subtask-aware policy + learned stop/progress，第 1/2/16 行），以及因此衍生的触发时机（第 6 行）、MBR 精度（第 14 行）与 budget（第 15 行）。其中第 5 行 VLM 回溯门槛比官方保守，可能是效果差异来源之一。

---

## 8. 同状态反事实恢复实验：中间结果（formal_v2，8/12 run）

实验设计：从同一个完整 MuJoCo 快照出发，对比 `no_op`（原 condition 继续）、`target_only`（切目标模板 condition）、
`full_cycle`（回溯 + 目标 condition + MBR）三条分支，分支 horizon = 剩余 episode 预算（对齐官方 `1.5x` 语义）。
运行配置已启用全部加速旋钮：`JITPI05_COUNTERFACTUAL_BRANCHES=no_op,target_only,full_cycle`、`JITPI05_RECORD_VIDEO=false`。

### 8.1 配对结果（64 个事件，8 个完成 run）

| branch | paired | rescue | harm | both_success | both_failure | net_rescue |
|--------|--------|--------|------|--------------|--------------|-----------|
| `full_cycle`（static-free-cycle） | 42 | 2 | 2 | 1 | 37 | **+0** |
| `target_only`（static-free-cycle） | 42 | 0 | 3 | 0 | 39 | **-3** |
| `target_only` / `full_cycle`（direct-cycle） | 22 | 0 | 0 | 0 | 22 | **+0** |

### 8.2 三条硬结论

1. **`full_cycle` 的净恢复收益为 0。** 逐事件联合分布：`both_failed=59`、`both_succeeded=1`、
   `full_cycle_only=2`、`no_op_only=2`。回溯+MBR 救回 2 个事件、同时毁掉 2 个事件，**完全相互抵消**。
   这是本实验最关键的结果：它把"MBR 是否有用"从"平均成功率差异"（会被噪声淹没）转化为**paired 判别量**，
   答案是"有正效应，但被等量的负效应抵消"。

2. **`target_only` 的 -3 是纯 prompt 切换损伤，不是恢复能力不足。** 在 `direct-cycle` 下 anchor condition
   就是 raw task prompt（[`_direct_condition()`](src/jitpi05/jitrl/rollout.py:162)），因此 `no_op` 与 `target_only`
   配置**完全等价**；实测 22/22 事件的 `success`/`executed_steps`/`termination_reason` 三项**逐字节相同**。
   这既是分支隔离正确性的强验证，也说明 `target_only` 的失败不能归因于"没回溯"，
   而应归因于 condition 文本本身在该任务上更难执行。

3. **触发精度只有 4.7%。** `no_op` 在 64 个被判定为失败的事件中仍有 3 个成功，
   即 VLM/物理检查的**假阳性率 ≈ 4.7%**。但代价不对称：一次假阳性触发一次完整回溯，
   平均回放 53 个 waypoint（见 8.3），而这些步数对预算是"免费"的、对物理是真实的。

### 8.3 回溯规模（修正版：早期脚本读错了字段名）

⚠️ **本节数据经过一次修正。** 早先的分析脚本把 `recovery_events` 里的
`target_anchor_id` / `waypoint_indices` 当作每个事件的字段，但该列表**交错两种事件类型**：

- `backtrack` → `from_anchor_id` / `to_anchor_id` / `replayed_steps` / `waypoint_indices`
- `mbr_retry` → `anchor_id` / `candidate_cursor` / `selected_index` / `failed_trajectory_count`

因此 `len(recovery_events)` 是**真实回溯数的 2 倍**，且 `target_anchor_id` 全部读到 `None`
（该字段只在 `counterfactual` 事件里存在）。修正后（[`_backtrack_truth.py`](artifacts/counterfactual_recovery_formal_v2/_backtrack_truth.py)）：

| run | checks | backtracks | mbr_retries | replay_sum | replay_mean | backward | self_retry | invalidations |
|-----|--------|-----------|-------------|-----------|-------------|----------|-----------|---------------|
| task62/direct-cycle/seed_17 | 105 | 57 | 57 | 2030 | 35.6 | 4 | 53 | 44 |
| task62/direct-cycle/seed_23 | 105 | 53 | 53 | 2070 | 39.1 | 6 | 47 | 28 |
| task62/static-free-cycle/seed_17 | 295 | 97 | 95 | 5150 | 53.1 | 26 | 71 | 150 |
| task62/static-free-cycle/seed_23 | 285 | 102 | 98 | 5565 | 54.6 | 28 | 74 | 169 |
| task69/static-free-cycle/seed_17 | 347 | 30 | 30 | 2180 | 72.7 | **29** | **1** | 60 |
| task69/static-free-cycle/seed_23 | 349 | 29 | 29 | 2110 | 72.8 | **28** | **1** | 57 |
| task79/static-free-cycle/seed_17 | 53 | 11 | 10 | 1160 | 105.5 | 9 | 2 | 45 |

聚合：`checks=1562`、`backtracks=382`、`mbr_retries=375`、`replay steps=20355`（均值 53.3）、
`backward=130`、`self_retry=252`、`invalidations=553`、**check→backtrack rate = 0.24**。

**修正后的结论与先前相反：回溯目标确实在推进，不存在全局振荡。**
`retry_counts` 显示每个 target 达 `CYCLE_MAX_RETRIES=3` 即退休并强制 transit
（task62 典型 episode：`{'anchor-0': 3, 'anchor-1': 1, 'anchor-3': 3}`），
且 `check→backtrack = 0.24` 说明 76% 的检查判定为"继续"，不是"每次失败都回溯"。

**但两个任务呈现完全相反的失败模式**，这是真正有价值的发现：

- **task62：`self_retry` 占绝对多数**（53/57、47/53）。回溯**退回同一个 anchor**
  （`to_anchor_id == from_anchor_id`，如 `anchor-1 → anchor-1`），即"原地重试"。
  `decision_reason` 为 `explicit physical postcondition failure; rewind to earliest valid anchor`。
  有效后退仅 4–6 次。这是**低效原地打转**。
- **task69：`backward` 占绝对多数**（29/30、28/29）。回溯**严格退到更早的 anchor**
  （如 `anchor-2 → anchor-0`），`invalidated_anchor_ids` 累计 60/57 个失效标记。
  这是**正确的恢复行为**——但成功率仍只有 1/20。

官方 `backtrace_robot_states` 是 **`restore_robot_only` + 每步 `dummy_action` 物理推进**
（[`run_libero_eval_openpi_cyclevla.py:640-646`](.reference_cyclevla/experiments/robot/libero/run_libero_eval_openpi_cyclevla.py:640)），
**只回退机器人关节、不回退场景物体**。我们的
[`rewind_robot_configuration_history()`](src/jitpi05/jitrl/libero_recovery.py:260) 与该语义一致。
因此 task69 式"正确后退却仍失败"**不是实现缺陷，而是这一机制的固有上限**：
物体已被撞飞/抓错位时，只回退机器人而不回退场景，重试会重复触发同样的物理失败。
而 task62 式"原地重试"指向 `select_recovery_anchor` 的目标选择——当
`physical_failure and target_anchor is None` 时会回落到 `current_anchor`
（[`cycle.py:850-862`](src/jitpi05/jitrl/cycle.py:850)），这是 self-retry 的来源。

### 8.4 4 个判别性事件（唯一携带信息的样本）

| run | ep | winner | `failure_anchor` | `target` | `full_cycle` rewind | MBR idx |
|-----|----|--------|------------------|----------|---------------------|---------|
| task69/static-free/seed_17 | 15 | `no_op` | anchor-1 | anchor-0 | 70 | 1 |
| task69/static-free/seed_23 | 15 | `full_cycle` | anchor-1 | anchor-0 | 70 | 6 |
| task69/static-free/seed_23 | 19 | `no_op` | anchor-1 | anchor-0 | 70 | 6 |
| task79/static-free/seed_17 | 10 | `full_cycle` | anchor-1 | anchor-1 | 30 | 2 |

注意 ep15 在 seed_17 与 seed_23 上**结论相反**（同一任务、同一 episode、同一 target），
且 seed_23 的 ep15 与 ep19 在**相同配置**（rewind=70、mbr=6）下也结论相反。
这直接证明：在当前样本量下，`full_cycle` 的 ±2 净效应**落在噪声区间内**，
任何基于单 seed 的"MBR 有效/无效"判断都不可靠。

### 8.5 对后续实验的设计含义

1. **判别量必须用 paired net_rescue，不能用平均成功率。** 前者已经显示为 +0，避免了"成功率没涨 → MBR 无效"的错误结论。
2. **样本量不足：需要 ≥6 个判别性事件才能拒绝噪声假设。** 当前 64 事件仅产生 4 个判别样本。
   提高判别效率的办法是 `CYCLE_COUNTERFACTUAL_BRANCHES=no_op,full_cycle`（去掉冗余的 `target_only`，
   direct-cycle 下它与 no_op 等价），并把预算从"每 episode 1 个事件"放宽到 ≥3 个。
3. **瓶颈是"机器人回退 + 场景不回退"的物理不可逆性，优先级高于 MBR 调参。**
   task69 给出了最干净的证据：`backward=29/30`（回溯策略完全正确）、`invalidations=60`
   （锚点确实被作废重规划），但成功率仍为 1/20。这说明**问题不在选择哪个 target，
   而在回退后场景物体已被推到不可恢复的位置**。官方只回退机器人（`restore_robot_only`），
   我们忠实复现了这一点，所以这是**机制固有上限**而非实现缺陷。
4. **task62 的 `self_retry` 是一个独立的、可修复的实现问题（已定位到具体分支）。**
   逐事件核对 `decision_reason` × 回溯方向的交叉分布
   （[`_self_retry_cause.py`](artifacts/counterfactual_recovery_formal_v2/_self_retry_cause.py)）：

   | task | self_retry（物理失败回落） | backward（物理失败回落） | backward（VLM 强证据） |
   |------|---------------------------|-------------------------|----------------------|
   | task62 | **245** | 29 | 35 |
   | task69 | 5 | **52** | 5 |
   | task79 | 2 | 5 | 4 |

   关键观察：**task62 与 task69 走的是完全相同的 reason 字符串**
   （`explicit physical postcondition failure; rewind to earliest valid anchor`），
   但结果一个是 self_retry、一个是 backward。所以决定因素**不是 reason，而是
   "是否存在严格更早的同阶段锚点"**：当没有时，
   [`resolve_cycle_decision()`](src/jitpi05/jitrl/cycle.py:850) 的
   `min(same_stage, ...)` 回落到 `current_anchor`，产生"原地重试"——
   回退到当前锚点等于重复同一动作序列，**物理上不可能改变结果**。

   task62 有 **245/309 = 79%** 的回溯落在这一退化路径上，这是它 0/40 episode 的直接原因。
   修复方向：移除 `current_anchor` 回落，改为**直接 transit**（或只接受严格更早的同阶段锚点）。
   task69 已证明正确的 backward 路径可正常工作，因此该修复不会破坏已有的正常恢复行为。
5. **下一步实验应验证场景回退。** `FullEnvironmentSnapshot` 已具备完整快照恢复能力
   （[`capture_full_environment_state()`](src/jitpi05/jitrl/counterfactual.py:59)），
   但 rewind 分支未使用它。新增一个 `scene_rewind` 分支做对照，即可直接检验
   "场景不可逆"这一假设是否成立——若该分支能显著打破 task69 的 1/20 天花板，
   则说明官方机制的瓶颈确在此处。
