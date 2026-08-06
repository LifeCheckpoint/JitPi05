# Harness VLA × JitRL 消融实验完整结果

> 分支：`eliminate` ｜ Benchmark：LIBERO-10 全 10 任务 ｜ Seed：17 ｜ 每任务 10 episodes
> 数据来源：`artifacts/jitrl_eval_libero10_long_seed17_qwen2b_workspace_v2_free/`（自由候选）与
> `artifacts/jitrl_eval_libero10_long_seed17_qwen2b_workspace_v2_no_stop_diagnostic/`（固定原语）。

## 1. 四格配置与定义

| # | 配置 | HarnessVLA（固定原语） | JitRL（在线记忆调制） | 对应 method |
|---|---|---|---|---|
| 1 | **Pi0.5** | —（无 Qwen 高层） | — | 纯 π₀.₅ 直接闭环 |
| 2 | **Pi0.5 + JitRL** | ✗（自由候选） | ✓ | `jitrl-free` |
| 3 | **Pi0.5 + HarnessVLA** | ✓（固定原语） | ✗（static） | `static`（固定工作集） |
| 4 | **Pi0.5 + HarnessVLA + JitRL** | ✓（固定原语） | ✓ | `jitrl`（固定工作集） |

术语：
- **HarnessVLA = 固定原语方案**（固定九类语义动作工作集，见 [`semantic_actions.py`](../src/jitpi05/jitrl/semantic_actions.py:9)）；非固定原语则为**自由候选方案**（Qwen 自由生成候选动作文本，见 [`free_proposal_prompt`](../src/jitpi05/jitrl/planner.py:506)）。
- **JitRL = 使用在线记忆 + advantage logit 调制**（`z'=z+β·Â`，见 [`apply_jitrl_update`](../src/jitpi05/jitrl/planner.py:447)）；否则为 **static**（advantage 恒 0、不读写记忆）。

所有配置共享：终止由环境判定；高层策略为 Qwen3.5-2B（NF4）；低层执行为冻结 π₀.₅（LeRobot `pi05-libero`）。

## 2. 完整结果

| 配置 | method | episodes | 成功率 | 平均成功步数 | 终止分布 |
|---|---|---|---|---|---|
| Pi0.5 + JitRL（自由候选） | `jitrl-free` | 100 | **95.0%** (95/100) | **249.4** | env_success 95 / max_steps 5 |
| Pi0.5 + HarnessVLA | `static`（固定） | 83* | **96.4%** (80/83) | **230.2** | env_success 80 / max_steps 3 |
| Pi0.5 + HarnessVLA + JitRL | `jitrl`（固定） | 76* | **97.4%** (74/76) | **228.6** | env_success 74 / max_steps 2 |
| （参考）自由候选 + static | `static-free` | 100 | 96.0% (96/100) | 256.5 | env_success 96 / max_steps 4 |

\* 固定工作集版本为历史产物，episodes 未跑满 100，成功率按已完成 episode 计。

### 2.1 配对差值（自由候选版本，`jitrl-free` vs `static-free`）

| 指标 | 差值 | 95% bootstrap CI |
|---|---|---|
| success_rate | −1.0pt | [−6.0, +3.0] |
| 成功步数缩减（static−jitrl，per-task paired） | +3.9 步 | [−2.7, +12.0]（方向有利 JitRL，不显著） |

补充（池化口径）：成功 episode 平均步数 `jitrl-free` 249.4 vs `static-free` 256.5，池化差约 7.1 步；per-task paired 均值 3.9 步受个别任务（如 task7 单差 27.6 步）拉低，CI 跨 0，均不显著。

### 2.2 机制指标（自由候选版本）

| method | free_action_match_rate | free_unseen_rate | choice_change_rate | 平均 logit shift | memory 规模 |
|---|---|---|---|---|---|
| `jitrl-free` | 0.58 | 0.42 | 13.2% | 0.18 | 70.7 条 |
| `static-free` | 0.0 | 1.0 | 0.0% | 0.0 | 0 |

## 3. 逐格解读

1. **Pi0.5（纯 π₀.₅）**：LIBERO-10 面板**未评测**。参考 `sim_eval` 诊断任务的 `original_task` 条件（π₀.₅ 直接执行任务描述）：分布内 `libero_object_task0` 100%（145.6 步）、零样本 `libero_90_task79` 20%（387.4 步）。说明 π₀.₅ 在分布内强、零样本弱；该格需在 LIBERO-10 面板补充 direct baseline。
2. **Pi0.5 + JitRL（自由候选）**：95.0% / 249.4 步。JitRL 记忆调制确实改变 13.2% 的决策、累积约 71 条经验，但成功率与 static 对照（96.0%）无显著差异。
3. **Pi0.5 + HarnessVLA（固定原语 + static）**：96.4% / 230.2 步。
4. **Pi0.5 + HarnessVLA + JitRL（固定原语 + jitrl）**：97.4% / 228.6 步。

## 4. 消融分析

**HarnessVLA（固定原语）的贡献**：以 static 对照看，固定原语 96.4% vs 自由候选 96.0% —— **成功率无显著差异**；固定原语的成功步数更低（230 vs 257，约 −26 步），但两版本 `HIGH_LEVEL_STEPS` 不一致（30 vs 40）、样本不同，不能严格归因。综合看，固定原语约束在本任务面板上未带来成功率增益，自由候选方案（Qwen 自由生成候选 + 环境终止）同样可靠（95–97%）。

**JitRL 的贡献**：两种架构下 JitRL 相对 static 的成功率变化为 ±1pt（不显著）；成功步数上 JitRL 均略省（固定 −1.6 步、自由候选 paired −3.9 步），但 CI 跨 0。**JitRL 的在线记忆 advantage 调制在本任务面板上未带来可测增益**，主要原因是 LIBERO-10 静态基线已接近成功率天花板（96%），缺少可提升的边际空间。

**综合结论**：四个配置的成功率均落在 95–97% 的饱和区间；HarnessVLA（固定原语）与 JitRL（记忆调制）两个机制在本面板上均为近似零的独立效应。即：在环境终止 + 冻结 π₀.₅ 的高层语义动作体系下，是否使用固定原语词表、是否启用 JitRL 记忆调制，对 LIBERO-10 最终成功率均无可测影响。
