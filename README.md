# 🎯 推荐系统实战体系 · 从入门到精通

> 📘 28章 实战体系 · 交互式 HTML 课程

> **English TL;DR:** An all-Chinese 28-chapter **recommendation-system** course from the industrial full-link architecture outward: overview & history, offline/online metrics & A/B testing, collaborative filtering (UserCF/ItemCF) & matrix factorization (ALS, Bias MF), feature engineering (statistical/real-time/embedding, feature store), the LR→FM→FFM evolution and deep CTR models (Wide&Deep, DeepFM, DCN, xDeepFM, DIN/DIEN sequence interest), recall (two-tower / YouTube DNN / MIND multi-interest / graph PinSAGE, multi-channel recall), ranking (coarse/fine, distillation, GBDT+LR), multi-task (ESMM/MMoE/PLE), rerank (MMR/DPP), real-time systems, cold start & Bandit, bias debiasing, RL in recommendation, multimodal content understanding, toolchain (feature platform / training infra), LLM-driven recommendation (LLM-as-Ranker, RAG recsys, conversational; OneRec/LUM), and frontier trends (generative rec, Semantic ID, integrated search+rec) — ending with a MovieLens end-to-end capstone (two-tower recall + Faiss + DeepFM).

## 📖 课程简介

本课程从**工业级推荐全链路**出发，系统构建推荐系统实战体系：先建立评估体系与协同过滤、矩阵分解的根基，再走通「特征工程 → LR/FM → Wide&Deep/DeepFM/DCN/DIN/DIEN」的 CTR 建模演进；召回侧覆盖双塔向量检索、多兴趣与图召回（MIND/PinSAGE），排序侧深入粗排精排、多任务（ESMM/MMOE）与重排（MMR/DP）。随后覆盖实时系统、冷启动、消偏、强化学习与多模态内容理解、工具链，以及大模型驱动的 LLM 推荐（LLM as Ranker、RAG、OneRec/LUM），最后以 MovieLens 全链路（双塔召回 + Faiss + DeepFM）的综合实战收束。

## 🚀 快速开始

```bash
open index.html   # macOS，纯静态即开即看
```

## 📂 项目结构

```text
recommendation-system-tutorial/
├── index.html / 01.html ~ 28.html / courses.json / theme.css
```

## 📖 章节分段

| 阶段 | 章节 | 核心 |
|------|------|------|
| **基础与评估** | 01–03 | 概览、简史、评估体系/A/B |
| **协同与 MF** | 04–05 | CF、矩阵分解/ALS |
| **特征与 CTR 模型** | 06–12 | 特征工程、LR→FM→FFM、Wide&Deep/DeepFM/DCN、DIN/DIEN |
| **召回与排序** | 13–17 | 双塔/多兴趣/图召回、粗精排/蒸馏/GBDT+LR、多任务、重排 |
| **系统与问题** | 18–22 | 实时、冷启动、消偏、强化学习、多模态 |
| **大模型与实战** | 23–28 | 工具链、LLM 推荐、工业实践（OneRec/LUM）、前沿、架构、MovieLens 实战 |

## ✨ 亮点

- 覆盖 CF 到 CTR 到召回/排序到 LLM 推荐的完整演进
- 含 OneRec / LUM 等 LLM 推荐工业实践与前沿
- MovieLens 端到端综合实战（双塔+Faiss+DeepFM）

## 🎯 前置知识

- 适合：推荐算法工程师 / 数据科学从业者
- 建议具备：Python + 机器学习基础

## ✨ 特色

- 以工业全链路为骨架，兼顾算法演进 + 工程系统
- 即开即用纯静态 HTML

---
*本课程由 `recommendation-system-tutorial/` 项目维护。*