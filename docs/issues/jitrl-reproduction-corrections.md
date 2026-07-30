# JitRL 复现偏差与修正方案

## 核心问题

当前实现没有在 Qwen3.5-4B 的基础策略上复现 JitRL，而是在每轮决策前先用 Gemini 扩展动作候选：

```text
Gemini 生成 5 个候选动作
→ Qwen 只对编号 1..5 评分
→ 记忆更新候选分数
→ 选择动作
```

因此，Qwen 的 logits 表示：

$$
\pi_{\mathrm{Qwen}}\!\left(i \mid s, C_{\mathrm{Gemini}}\right)
$$

而不是 Qwen 自身的高层动作策略：

$$
\pi_{\mathrm{Qwen}}\!\left(a \mid s\right)
$$

### 1. Gemini 候选属于外部动作增广

Gemini 每轮生成五个候选，相当于在 JitRL 更新前引入一个更强的 proposal model，扩展并重构了 Qwen 的可选动作范围。哪些动作能够被选择，首先由 Gemini 决定，而不是由冻结的 4B 基础策略决定。

当前实验实际测试的是：

> Gemini 扩展候选后，Qwen 能否结合 memory 进行候选重排。

这与目标问题不同：

> 记忆估计的 advantage 能否直接改善冻结 Qwen3.5-4B 的基础策略。

### 2. JitRL 应重分配基础策略中的概率

JitRL 的更新为：

$$
z'(s,a)=z(s,a)+\beta A(s,a)
$$

其作用是在同一个基础策略和动作空间内重新分配概率，使概率质量向高 advantage 动作集中、从低 advantage 动作移开。直观上，这是对基础策略有效决策范围的收缩和强化，而不是依赖外部模型不断扩展候选空间。

Gemini 已经在更新前改变了动作集合，因此后续的 logit modulation 不再是对原始 4B policy 的独立更新。

### 3. 当前 logits 只是候选编号分数

Qwen 没有生成这些动作，只看到了 Gemini 已经写好的五个候选。因此，编号 `1..5` 的下一 token logits 只能表示 Qwen 对这组外部候选的相对偏好，不能表示 4B 自己会提出哪些动作及其原始概率。

这使 Gemini 的 ICL/proposal 能力与 JitRL memory 的作用混在一起，无法判断实验效果来自哪一部分。

### 4. Memory-only 动作进一步引入人工先验

记忆检索还会加入五个 Gemini 候选之外的历史动作。这些动作没有对应的 Qwen 基础 logit，当前实现将其人工设置为 `base_logit=0`。

增广后的向量同时包含 Qwen logits 和人工 logits。在该向量上执行 advantage 更新，无法被解释为对同一个基础策略的更新。

### 5. 当前结果不能评价 JitRL

现有结果混合了：

- Gemini 每轮候选增广；
- Qwen 编号重排；
- 记忆历史动作注入；
- 人工初始化的 memory-only logits；
- advantage modulation。

因此，当前效果不好不能说明 JitRL 对机器人高层控制无效，只能说明这套外部候选扩展与 memory 重排系统没有产生收益。

## 修正方案

### 1. 移除 Gemini 高层候选生成

Gemini 不再生成五个动作候选。高层候选的状态绑定、参数实例化和对应选择 logits 必须由同一个 Qwen3.5-4B 完成，确保被调制的基础策略就是研究对象。

### 2. 为 Qwen 提供固定的语义动作工作集

参考 Harness VLA，在评测开始前定义固定的原语词表，并在部署期间禁止动态发明新的动作类型。参考 SARL，将参数化的高层语言指令定义为控制冻结 VLA 的 semantic actions。本文将二者组合为一个固定的“语义动作工作集”。

语义动作工作集应采用参数化模板，例如：

```text
approach(target)
align(gripper, target)
grasp(target)
lift(target)
transport(target, destination)
place(target, destination)
release(target, destination)
retract(direction)
stop()
```

具体模板应根据 π₀.₅ 在 LIBERO 中能够可靠执行的能力确定，而不是直接照搬 Harness VLA 的解析控制原语。

完整动作类型在评测前固定，并通过 system prompt、动作 schema 和少量 ICL 示例提供给 4B。ICL 示例负责教会 4B：

- 根据当前状态选择动作类型；
- 将任务中的物体、目标区域和空间关系绑定到参数；
- 将长任务分解为可执行的语义动作；
- 在成功、失败和完成状态下选择继续、恢复或停止。

这是一种对 4B 的固定能力支架，不是每轮动态扩展 action space。每轮变化的只有：

- Qwen 根据观测填写的物体、目标区域和方向参数；
- 当前状态下有效的模板实例；
- Qwen 对这些实例计算的基础 logits。

Gemini 不参与动作类型或候选实例的生成。

第一轮 debug 应只保留：

```text
固定语义动作工作集 + ICL
→ Qwen 绑定参数并形成当前候选
→ Qwen 计算基础 logits
→ 记忆为当前候选估计 advantage
→ 更新 logits
→ 选择动作
```

暂时禁止 Gemini 候选增广和 memory-only 动作注入，单独验证 advantage modulation 是否能够改善 Qwen 基础策略。

### 3. 分开验证各机制

先完成固定语义动作工作集上的 4B-only JitRL 基线，再将任何动态候选扩展作为独立消融实验。

## 最终建议

当前实现应回退到单一 4B policy：

```text
固定语义动作工作集 + ICL
→ Qwen 状态绑定与候选 logits
→ 记忆估计 advantage
→ Logit 重加权
→ 动作选择
```

只有这一版本能够直接回答：JitRL 是否通过测试时 logit 更新改善了冻结的 Qwen3.5-4B 策略。

## 设计参考

- [Harness VLA](../papers/Harness%20VLA.pdf)：使用部署前固定的 primitive vocabulary，规划器只能选择原语并绑定参数。
- [SARL](../papers/SARL.pdf)：将语言 prompt 定义为 semantic action，通过语义动作空间调用冻结的 VLA skill prior。
