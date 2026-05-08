# Source Adapter Quality Gate 定義書

## 1. 概要

**Source Adapter Quality Gate** は、Neural-Scalpel において、LoRA / Adapter を別モデルへ Structural Projection する前に、その source adapter が本当に移植する価値のある task delta を持っているかを検査するための標準ゲートである。

Neural-Scalpel の Structural Projection は、source adapter の差分を target model へ転写する仕組みである。したがって、source adapter が有効な改善信号を持っていれば、その一部が target model でも有効に働く可能性がある。一方で、source adapter が自身の base model を劣化させる場合、その干渉成分や悪いバイアスも target 側へ転写される可能性がある。

そのため、Projection 前に以下を必ず検証する。

```text
source base model
vs
source base model + source adapter
```

この比較によって、source adapter が **Positive Teacher** なのか、**Neutral Teacher** なのか、**Negative Teacher** なのかを判定する。

---

## 2. 目的

Source Adapter Quality Gate の目的は、以下である。

```text
1. Projection 前に source adapter の品質を定量評価する
2. 有効な task delta と干渉 delta を区別する
3. 無意味または有害な adapter の投影を防ぐ
4. Structural Projection の成功確率を高める
5. 失敗時の原因切り分けを容易にする
6. adapter migration を「運任せ」から「診断付きプロセス」へ変える
```

特に重要なのは、Structural Projection を以下のように誤解しないことである。

```text
誤り:
Structural Projection すれば、どんな LoRA でも target model を改善できる

正しい理解:
Structural Projection は source adapter の delta を target model へ転写する。
その delta が有益であれば改善し得るが、有害であれば干渉も転写され得る。
```

---

## 3. 背景

Neural-Scalpel の Qwen2.5 SQL projection 実験では、2種類の source adapter によって明確に異なる結果が得られた。

### 3.1 Negative Teacher の例

ある Qwen2.5-7B SQL LoRA は、source base model 自身の SQL-50 accuracy を以下のように劣化させた。

```text
Qwen2.5-7B base:        62.0%
Qwen2.5-7B + SQL LoRA: 56.0%
Delta:                 -6.0%
```

この adapter を Structural Projection した場合、target 側では改善と干渉が混在した。

```text
Qwen2.5-3B:   +4.0%
Qwen2.5-1.5B: -2.0%
Qwen2.5-0.5B: +4.0%
```

この結果から、source adapter が有害な delta を含む場合、その干渉成分も target に転写され得ることが示唆された。

### 3.2 Positive Teacher の例

一方、Qwen2.5-Coder 系の高品質な SQL adapter は、source teacher を大きく改善した。

```text
Qwen2.5-Coder-7B base:        62.0%
Qwen2.5-Coder-7B + SQL LoRA: 78.0%
Delta:                       +16.0%
```

この Positive Teacher を Structural Projection した場合、すべての target size で改善が観測された。

```text
Qwen2.5-Coder-3B:   +6.0%
Qwen2.5-Coder-1.5B: +6.0%
Qwen2.5-Coder-0.5B: +4.0%
```

この比較から、以下の設計原則が導かれる。

```text
Projection 前に source adapter quality を評価する必要がある。
```

---

## 4. 用語定義

### 4.1 Source Base Model

adapter が本来適用される base model。

### 4.2 Source Adapter

source base model 用に学習された LoRA / PEFT adapter。

### 4.3 Source Adapter Delta

source adapter が source base model に与える挙動差分または重み差分。

### 4.4 Positive Teacher

source adapter が source base model の下流評価指標を有意に改善する状態。

### 4.5 Neutral Teacher

source adapter が source base model にほとんど影響しない状態。

### 4.6 Negative Teacher

source adapter が source base model を劣化させる状態。

### 4.7 Quality Gate

Projection を実行してよいかを判断する検査段階。

---

## 5. 基本思想

Source Adapter Quality Gate は、以下の仮説に基づく。

```text
Structural Projection は source adapter delta の転写である。
したがって、source adapter delta が source model 上で有益であるほど、
target model 上でも有益に働く可能性が高い。
```

---

## 6. 判定指標 (SQL)

- **execution_accuracy**: プライマリ指標。
- **execution_success_rate**: 実行成功率。
- **syntax_validity**: 構文の正当性。
- **failure_classification**: 改善（fixed）と回帰（regressed）の件数。

---

## 7. 判定基準 (Verdict)

| Verdict | Gate Status | Recommendation | 意味 |
|---|---|---|---|
| **POSITIVE_TEACHER** | **PASS** | PROCEED_TO_PROJECTION | 改善が顕著。投影推奨。 |
| **WEAK_POSITIVE** | **WARNING** | PROCEED_WITH_CAUTION | 小幅な改善。注意して投影。 |
| **NEUTRAL_TEACHER** | **WARNING** | NOT_PRIORITIZED | 変化なし。優先度低。 |
| **NEGATIVE_TEACHER** | **FAIL** | DO_NOT_PROJECT | 劣化。投影非推奨。 |
| **UNSTABLE_TEACHER** | **FAIL** | DO_NOT_PROJECT | 崩壊・反復など。投影禁止。 |

---

## 8. 推奨ワークフロー

1. **Evaluate**: source adapter を自身の base model 上で評価する。
2. **Gate**: Quality Gate を通過した adapter のみ採用する。
3. **Project**: target model へ Structural Projection する。
4. **Validate**: downstream benchmark で検証してから公開する。
