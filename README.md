# Experiment Configuration Manifest

## 1. Models

| Display Name | model_id | Provider | Endpoint | Context | Temperature | max_tokens | timeout | API Key |
|---|---|---|---|---|---|---|---|---|
| claude-opus-4-6 | claude-opus-4-6 | Anthropic (via gateway) | platform-api.xaminim.com | 1M | 0.7 | 4096 | 120s | single |
| gpt-5.4 | gpt-5.4 | OpenAI (via gateway) | platform-api.xaminim.com | 1M | 0.7 | 4096 | 120s | single |
| gemini-3.1-pro-preview | gemini-3.1-pro-preview | Google (via gateway) | platform-api.xaminim.com | 2M | 0.7 | 4096 | 120s | single |
| MiniMax-M2.5 | MiniMax-M2.5 | MiniMax (via gateway) | platform-api.xaminim.com | — | 0.7 | 4096 | 120s | single |
| DeepSeek-V4-Pro | deepseek-chat | DeepSeek (official) | api.deepseek.com | 1M | 0.7 | 4096 | 120s | single |
| GLM-5.1 | glm | Zhipu (via SJTU) | models.sjtu.edu.cn | 128k | 0.7 | 4096 | 300s | 3-key rotation |
| Qwen3.5 | qwen | Alibaba (via SJTU) | models.sjtu.edu.cn | 256k | 0.7 | 4096 | 300s | 3-key rotation |
| DeepSeek-V3.2-reasoner | deepseek-reasoner | DeepSeek (via SJTU) | models.sjtu.edu.cn | 32k | 0.7 | 4096 | 300s | 3-key rotation |

### LLM Call Parameters (shared)
- Temperature: **0.7** (all models, all conditions)
- max_tokens per call: **4096** (Condition A tool-calling), **32768** (Condition B), **8192** (Condition C)
- 429 retry: **3 retries**, exponential backoff (5s → 10s → 20s)

---

## 2. Experimental Conditions

| Condition | Tools | max_rounds | max_per_tool | max_tokens | System Prompt | Final Validation |
|---|---|---|---|---|---|---|
| **A** Full Tool | 10 FANGCUN endpoints | 24 | 20 | 4096/round | 1321 chars (workflow + readable) | validate_meter |
| **B** Description Only | None (text only) | 0 | — | **32768** | 1274 chars (tool knowledge + 入声字表) | — |
| **C** Baseline | None | 0 | — | **8192** | 117 chars (minimal) | — |
| **D1** −Template | 8 (no rules_list, examples) | 24 | 20 | 4096/round | 871 chars | validate_meter |
| **D2** −Validation | 9 (no validate_meter) | 24 | 20 | 4096/round | 939 chars | — |

### System Prompt Design
- **A**: Structured workflow (query → draft → validate → revise), references `readable` prosodic template
- **B**: Describes all tool capabilities in text, includes 入声字表, emphasizes Cilinzhengyun/Pingshuiyun distinction. Token budget disclosed in prompt.
- **C**: Minimal role assignment. Token budget disclosed but no prosodic guidance.
- All conditions: `\boxed{}` output format, anti-plagiarism clause, punctuation required

---

## 3. FANGCUN Tool Suite

### Endpoints (3 hosts)

| Host | Endpoint | Tool | Method | Purpose |
|---|---|---|---|---|
| <FANGCUN_WRITE> | /api/validate_meter | validate_meter | POST | Core prosody validation |
| <FANGCUN_WRITE> | /api/char/lookup | char_lookup | GET | Character tone/rhyme lookup |
| <FANGCUN_WRITE> | /api/rhyme/lookup | rhyme_lookup | GET | Rhyme category character list |
| <FANGCUN_WRITE> | /api/rhyme/list | rhyme_list | GET | Rhyme category overview |
| <FANGCUN_WRITE> | /api/rules/list | rules_list | GET | Prosodic template database |
| <FANGCUN_WRITE> | /api/dictionary/search | dictionary_search | GET | Word association / antithetical pairing |
| <FANGCUN_WRITE> | /api/dictionary/allusion | dictionary_allusion | GET | Literary allusion retrieval |
| <FANGCUN_WRITE> | /api/free_rhyme | free_rhyme | POST | Free verse rhyme detection |
| <FANGCUN_CHECKER> | /api/examples | examples | GET | Historical exemplar poems |
| <FANGCUN_SHI> | /api/search/text | search_text | GET | Full-text corpus search |

### Orchestration Layer Features
- **Genre ↔ rhyme book enforcement**: Ci → Cilinzhengyun, Shi → Pingshuiyun (auto-fill or reject mismatch)
- **Warnings auto-inject**: Ci → `["2gram", "rhyme_duplicate"]`, Shi → `"default"`
- **Parameter filtering**: Each tool only receives parameters declared in its schema
- **Response slimming**: `raw_response`, `display_segments` stripped from LLM-facing messages
- **Rule entry slimming**: Only `name`, `char_count`, `readable`, `sentence_count` sent to LLM
- **Error hints with auto-bundled lookups** (≤10 errors):
  - Tone errors: bundled `dictionary_search(mode=head/tail, tone=P/Z)` suggestions
  - Rhyme errors (known category): bundled `rhyme_lookup` top 20 chars
  - Rhyme errors (unknown category): bundled `char_lookup` for each error char
  - 404 errors: bundled `rhyme_list` with correct category names
- **Round countdown**: Warning at ≤5 remaining rounds, force output at last round
- **Placeholder support**: □ allowed in drafts and validate_meter, forbidden in final `\boxed{}` output

### Rate Limiting
| Endpoint | Limit | Implemented |
|---|---|---|
| validate_meter | 60/min (API) → 55/min (client) | Sliding-window RateLimiter |
| free_rhyme | 60/min → 55/min | Same |
| dictionary_search | 120/min → 110/min | Same |
| dictionary_allusion | 120/min → 110/min | Same |
| Others | 60/min → 55/min | Same |

---

## 4. Test Suite

### Prompt Construction
- **Source**: 297 内容 keywords from "Shi-Ci's Last Exam" (基础题·内容)
- **Assignment**: Pseudo-random 1:1 (seed=42), each cipai gets exactly 1 keyword
- **Format**: `围绕【{keyword}】主题，以《{cipai_name}》为词牌名填词一首，符合中国古典诗词的格律规范。`
- **Regulated verse format**: `围绕【{keyword}】主题，作{rule_name}一首，符合中国古典诗词的格律规范。`

### Splits (144 cípái patterns)

| Split | Count | Source | Description |
|---|---|---|---|
| changdiao_popular | 50 | 全宋词 top 50 by frequency | Popular long forms (≥91 chars) |
| changdiao_rare | 50 | 全宋词 bottom 50 by frequency | Rare long forms (≤10 extant examples) |
| zhongdiao_popular | 5 | Top 5 medium forms | 蝶恋花, 忆江南, 踏莎行, 渔家傲, 洞仙歌 |
| zhongdiao_rare | 5 | Bottom 5 medium forms | 甘露歌, 荷华媚, 钿带长中调, 隔帘听, 韵令 |
| xiaoling_popular | 5 | Top 5 short forms | 浣溪沙, 浪淘沙, 生查子, 鹧鸪天, 菩萨蛮 |
| xiaoling_rare | 5 | Bottom 5 short forms | 破字令, 竹香子, 荔子丹, 荷叶铺水面, 醉高歌 |
| regulated | 16 | All 16 regulated verse types | 五绝/七绝/五律/七律 × 4 tonal variants |
| special | 8 | Hand-picked difficulty cases | 戚氏(212字), 六州歌头, 皂罗特髻, 调笑令, 定风波, 西江月, 减字木兰花, 翠楼吟 |

---

## 5. Evaluation Metrics

### Prosodic Compliance (from validate_meter)
| Metric | Definition | Notes |
|---|---|---|
| **ZER** | Fraction of poems with zero errors | Primary metric |
| **CTA** | Per-character tonal accuracy | 1 − (tone_errors / total_chars) |
| **SA** | Structural accuracy (char count match) | Binary |
| **RA** | Rhyme position accuracy | correct_rhyme / total_rhyme |
| **WER** | (errors + 0.5×warnings) / total_chars | Warnings at 0.5× weight |

### Error Classification
| Type | Source | Description |
|---|---|---|
| Tone | validate_meter errors | 平仄 mismatch at specific position |
| Rhyme | validate_meter errors | 韵脚 not in expected rhyme category |
| Punctuation | validate_meter errors | Sentence break mark incorrect |
| Structural | validate_meter errors | Character count mismatch |
| Warning | validate_meter warnings | Repeated characters (2gram, rhyme_duplicate) |

### Warning Exemptions
- 调笑令_钦谱_格一: warnings zeroed (规定叠字句)
- 皂罗特髻_钦谱_格一: warnings zeroed (特殊回环结构)

### Literary Quality (LLM-as-Judge, 1-5 scale)
| Dimension | What it measures |
|---|---|
| Fluency | Grammatical correctness, natural phrasing |
| Coherence | Thematic consistency across lines/stanzas |
| Poetic Quality | Artistic effect, emotional resonance (意境) |

### Originality (Posthoc similarity analysis)
- Endpoint: `<FANGCUN_SHI>/api/search/similar`
- Method: Bigram Jaccard similarity against ~800K classical poems
- Leniency: Sentences ≤4 chars excluded (idiomatic phrases)
- Metrics: max_similarity, mean_similarity, originality_score (1 − mean)
- Plagiarism flag: score=1.0 match to pre-modern author

### Posthoc Validation (Conditions B/C)
- Endpoint: `<FANGCUN_CHECKER>/api/validate_batch` (24/batch)
- `rule_name`: cipai name prefix (allows all variants)
- `rhyme_book_name`: **Cilinzhengyun** (Ci) / **Pingshuiyun** (Shi)
- `include_punctuation`: **false** (Conditions B/C don't have 句读 guidance)
- Condition A: uses online trace validation results as fallback (exact rule_name, include_punctuation=true)

---

## 6. Execution Parameters

| Parameter | Value |
|---|---|
| Concurrency per model | 2-4 (adaptive) |
| Resume support | Yes (skip records with text + non-error finish_reason) |
| Output format | JSONL (slim) + trace JSON (tool conditions) |
| Trace storage | `{outdir}/{model}/traces/{condition}/{prompt_id}.json` |
| Post-processing | `\boxed{}` extraction, `<think>` tag stripping |
| Output validation | Reject if contains □ placeholder |
