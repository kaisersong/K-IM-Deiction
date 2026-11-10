# K-IM-Deiction

A judgment / triage model for Yunzhijia (云之家) group messages: it classifies a message window, scores importance and urgency, and outputs calibrated probabilities for "needs my attention" and "contains a to-do". It only makes judgments — what to do with them (surface, create a task, escalate, ignore) is decided by surrounding rules and people.

云之家群消息判断 / 分诊模型：对消息窗口做分类、重要 / 紧急打分，并给出「需关注」与「含待办」的校准概率。它只负责判断，冒泡、建任务、升级还是忽略由外围规则和人决定。

> **Weights live on Hugging Face — this repository contains no model files.**
> **模型权重在 Hugging Face，本仓库不含任何权重文件。**
> Download / 下载：https://huggingface.co/Songkai/K-IM-Deiction

## What it is / 这是什么

- Base model / 基座：Qwen/Qwen3.5-0.8B (text core only)
- Method / 方法：LoRA + 7 linear judgment heads on the pooled last-token hidden state, with temperature + isotonic calibration
- Current release / 当前版本：**V4** (HF revision `566cd57cf2`, promoted 2026-11-10; relevance signals as a model input — windows must carry a signal header and a 16-d signal feature vector). The previous release **V2** is preserved in the HF repo's `v2/` folder.
- Two forms in one HF repo / HF 仓内双形态：merged full package at the root, LoRA adapter under `adapter/`
- An MLX build of V4 for Apple Silicon is provided in the `mlx/` folder of the HF repo (blind-set F1 0.664, needs-attention AUC 0.879, 99.20% top-1 agreement with the torch run); the V2 MLX edition is preserved under `v2/mlx/`. HF 仓 `mlx/` 目录另有 Apple Silicon 用的 V4 MLX 版（V2 的 MLX 版保全在 `v2/mlx/`）。

## Key results / 关键指标

Frozen blind set, 498 windows from nine groups held out whole, narrow-caliber gold / 冻结盲测集 498 条（9 个整群隔离，窄口径金标）：

| Metric / 指标 | V4 |
|---|---|
| Macro-F1 (primary 5 classes) / 分类 F1（主五类） | 0.670 |
| Macro-F1 (all 7 classes) / 分类 F1（全 7 类） | 0.620 |
| Needs-attention AUC / 需关注 AUC | 0.880 |
| Needs-attention ECE / 需关注 ECE | 0.017 |
| Has-todo AUC / 含待办 AUC | 0.945 |
| Has-todo ECE / 含待办 ECE | 0.064 |
| Shadow replay (4 days, L0) / 影子回放（四日，L0） | P@20 0.0889 · Recall@20 1.00 · key misses 0 |

V2's numbers on its own 1,000-window held-out set are kept in the model card's V2 evaluation section. V2 在其 1,000 条保密集上的指标保留在模型卡的 V2 评测节。

## Resource footprint / 资源占用（2026-11-07 实测）

| Edition / 版本 | Process RAM / 进程内存 | GPU / unified memory / 显存·统一内存 | On disk / 磁盘 |
|---|---|---|---|
| PyTorch package (Windows, WSL2, RTX 4090 D) / PyTorch 整包 | ~1.8 GB steady / 稳态约 1.8 GB | ~2.2 GB VRAM / 显存约 2.2 GB | 1.83 GB |
| MLX edition (Apple Silicon, slim) / MLX 瘦身版 | ~1.7 GB | MLX active ~1.5 GB, peak ~1.6 GB / 活跃约 1.5 GB、峰值约 1.6 GB | 1.53 GB |

## Data & privacy / 数据与隐私

Training and held-out/evaluation data are **not released**. In both, sender fields were replaced by pseudonymous IDs (`S####`) before labeling; labeling itself was done through third-party APIs (BigModel's GLM-5.3 as primary teacher, DeepSeek for second-pass labels), with sender pseudonyms in place but message body text verbatim at the time. On 2026-11-07 the retained local copies were also body-scrubbed (person names, phone numbers, company names, ID numbers and emails in message bodies replaced by placeholders, ~127k replacements, post-scrub pattern scans return zero). Window headers still carry real group names, so the local datasets are pseudonymized and body-scrubbed rather than provably fully anonymized, and exist only on the author's machines. Scrubbing the retained copies does not retroactively change the trained weights, and no zero-leakage guarantee is claimed for them. The example window on the model card is synthetic.
训练与评测数据**不公开**。两者的发送者字段在打标前均已替换为假名编号（`S####`）；标注本身经第三方接口完成（BigModel 的 GLM-5.3 主标、DeepSeek 二标），当时发送者为假名、正文为原文。2026-11-07 又对留存的本地副本做了正文级脱敏（正文中的人名、电话、公司名、身份证号与邮箱替换为占位符，约 12.7 万处，脱敏后复扫零命中）。窗口头仍为真实群名，因此本地数据集属于已假名化并经正文脱敏、但不宣称完全匿名化，仅存放在作者的机器上。留存副本的清洗不追溯改变已训练的权重，对权重不宣称零泄漏风险。模型卡中的示例窗口为虚构。

## Download & use / 下载与使用

```python
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="Songkai/K-IM-Deiction")
# Root = merged full package; adapter/ = LoRA adapter for the base model.
# The 7 judgment heads (heads.pt) and calibration files must be loaded
# together with the backbone — see the Hugging Face model card.
```

It is **not** a generative chat model: loading the adapter alone into a chat pipeline produces no judgments. Full usage code (both forms), the local HTTP service wrapper, and plain-language explanations of AUC/ECE are on the Hugging Face model card — this GitHub repo is a pointer, the card is the source of truth.
它**不是**生成式聊天模型：只挂适配器当聊天模型用得不到任何判断。完整使用代码（双形态）、本地服务封装与 AUC/ECE 的大白话解释见 Hugging Face 模型卡；本 GitHub 仓库只是索引，细节以模型卡为准。

## Known limitations / 已知短板

- Decision/conclusion is the weakest of the main classes (F1 0.444 on the V4 blind set). 决策 / 结论是主要类别里最弱的一类（V4 盲测集 F1 0.444）。
- Sentiment accuracy is below a large model judging zero-shot (91.2% here). 情绪判断弱于大模型零样本直判（本模型 91.2%）。
- V4 needs its relevance signals: relevant messages carrying no @ / reply / participation signal still score near zero (six-day merged shadow Recall@20 0.50). V4 依赖相关性信号：不带任何指向信号的相关消息分数仍在近零区（六日合并影子召回 0.50）。
- Trained only on Chinese enterprise group chats; it cannot see attachments, images or link contents. 训练域仅中文企业群聊；看不见附件、图片与链接内容。

## Roadmap (summary) / 下一步计划（摘要）

Full roadmap on the Hugging Face model card. 完整版见 Hugging Face 模型卡。

- **Phase 1 / 第一期**：decision-boundary arbitration (freeze labeling rules for implicit sign-offs, pending confirmations, vetoes), user-day shadow-trial design (20/day budget unchanged), a small double-blind human anchor sample, and strict dev/calibration/blind-test separation. decision 边界仲裁、用户日级影子试验设计（每天 20 条预算不变）、人工锚点小样、评测隔离制度化。
- **Phase 2 / 第二期**：calibration on an independent set with PR-AUC/Brier/reliability reporting, a low-confidence escalation path (human or authorized local path by default), and personalization-conditioning groundwork. 独立集校准与低置信升级通道、个性化条件预研。
- **Phase 3 / 第三期**：targeted top-up training and one-at-a-time structural ablations. 定向补训与结构消融逐项做。

Acceptance criteria are set before each experiment, from actual loss and budget — no pre-committed hard gates. 验收标准在实验前按实际损失与预算定，不预设硬门槛。
