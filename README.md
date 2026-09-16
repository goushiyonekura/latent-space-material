# 《潜在空間》作曲素材生成ツール — headless / gain-only / spec v1.1 (H5)

固定音声ループ＋連続音量制御だけで、`OPEN → CONTRACT → GOAL_HOLD → REOPEN` の周期を持つ
オフライン音響を生成する CLI。GUI なし、依存は **Python 3.9 + numpy** のみ（GPU 不要）。

```
latent_space/            エンジン本体
  audio_io.py            WAV 読み書き（PCM 8/16/24/32, float32/64）
  curves.py              Q5 ランプ・ホールド、式(7)〜(11) の解析的ハード検査
  form.py                INTRO/OPEN/CONTRACT/GOAL_HOLD/REOPEN の整数フレーム計画、開放度 o(t)
  analysis.py            解析窓ごとの Gram 行列（実 PCM の和の厳密な二次形式）→ 配合状態 ξ、d_ξ、
                         レンダリング PCM からの配合再計算（composition_from_render）
  bank.py                合法な共同ゲイン軌道候補の生成・変異、ゴール露出方針、運動時間分布
  objective.py           E_form / E_hist / E_motion / J、選択則（argmin／許容差内 softmax）
  history.py             履歴 H（h_c, M_H, 時刻順イベント列, 変化方向, 親系譜, 方式別要約）
  modes/                 diffusion.py / vae.py / transformer.py / gan.py（四方式の制御器）
  engine.py              §12.4 の実行手順、最終目標での再採点、介入プローブ、ハード検査、trace
  render.py              式(1) だけを行う正式レンダラー
  trace.py               result.wav / gain_curves.csv / state_trace.json
  fixtures.py            人工 WAV 素材の生成（開発・検査用）
scripts/smoke.py         検査 A〜C（§16）＋監査対応の追加検査
scripts/prepare_inputs.sh  MP3 → WAV デコード（macOS afconvert）
scripts/run_all.sh       四方式一括生成
project.json             実素材用設定（inputs/*.wav）
```

## 実行

```bash
pip3 install --user numpy            # 未導入なら
sh scripts/prepare_inputs.sh         # mp3 → inputs/*.wav（済み）
python3 -m latent_space generate --config project.json --mode diffusion --output output/diffusion
```

四方式まとめて `sh scripts/run_all.sh`。`--mode` は設定の `mode` を上書き、`--seed` で seed 上書き可、
`--out` は `--output` の別名。実際に使った設定は trace の `resolved_config` に保存される。

## 出力（各 output/<mode>/）

| ファイル | 内容 |
|---|---|
| `result.wav` | 式(1) でレンダリングした float32 WAV |
| `gain_curves.csv` | 0.05 秒刻みの 0〜100 表示ゲイン（確認用の間引き列） |
| `state_trace.json` | 正本：解析的曲線 (`curve_segments_per_track`)、実行設定と適用した別名、素材ハッシュ、履歴要約、方式内部状態、ハード判定、ソフト残差、`run_status` |

trace の曲線から `render.render_equation_1` で再レンダリングすると `result.wav` とビット一致する
（`scripts/smoke.py` の検査 C）。

## 設定（監査対応で追加・整理した項目）

| キー | 既定値 | 意味 |
|---|---|---|
| `form.goal_exposure.policy` | `contract_only` | ゴールトラックの露出方針。INTRO/OPEN は上限以下（既定 0）、CONTRACT 内で連続的に立ち上げて到達時に 1、REOPEN は 1 を引き継ぎ合法速度で下降。`free` で旧挙動 |
| `form.goal_exposure.intro_max` / `open_max` | 0.0 | 各段階のゴールゲイン上限 |
| `search.selection` | `argmin` | 最終候補の選び方。`softmax_within_margin` なら `acceptance_margin` 内だけを温度抽選 |
| `search.acceptance_margin` | 0.0 | 許容差 `J <= J_min + margin`（最終目標で全候補を再採点した同一尺度） |
| `search.final_targets` | `best_round` | 採用する固定目標集合：最良ラウンドの目標 or `last_round` |
| `search.tolerance_applies_to` | `mode_error` | 早期終了の目安を方式誤差（純粋距離＋方式ペナルティ）に適用するか `target_fit`（純粋距離）か |
| `search.amplitude_mixture` / `duration_preference_beta` | 小/中/大 = 0.45/0.35/0.20、Beta(1,4) | 候補の運動振幅と所要時間の分布（未校正の工学値） |
| `spec_document` | null | 正本仕様のパス。指定すると SHA-256 を trace に記録 |

別名 `schedule.*`（→`form.*`）、`motion.relative_rate_max`、`motion.ramp_component_duration_max_seconds`、
`search.max_rounds`、`search.candidate_count` は正本キーへ変換され `config_aliases_applied` に記録される。
未知のキーはエラーになる（黙って無視しない）。

## 状態ラベル

- `VALID_APPROXIMATION`：ハード条件・構造契約を満たし、**採用した軌道**が tolerance 以下。
- `BEST_EFFORT`：ハード条件・構造契約を満たすが近似目安が未達（正常出力）。
- `HARD_CONSTRAINT_FAILURE`：合法な軌道が得られない／検査不合格（WAV は出力しない）。
- `INCOMPLETE_IMPLEMENTATION`：指定方式が未実装（別方式で代用しない）。
- `INPUT_ERROR`：入力不備（fs/ch 不一致、N<2、無音ゴール等）。

`calibration_status` / `perceptual_status` は常に `UNVERIFIED`。

## trace に分けて記録する三つの誤差

| 項目 | 内容 |
|---|---|
| `units[*].chosen.target_fit_error` | 採用目標と実現配合の純粋な平均距離 d_ξ² |
| `units[*].chosen.mode_penalty` | 方式固有の追加項（Diffusion の κ_E·𝓔_D 等）。`normalized_mode_error = fit + penalty` |
| `hard_checks.rendered_composition_check` | 同じ解析窓の連続ブロックで、代理評価（窓中心ゲイン保持の Gram 評価）／採用目標／実レンダリング PCM の配合を相互比較 |

履歴の介入検査 `units[*].history_probe` は、同じユニット開始前の方式状態・同じ素材位相・同じ乱数状態から、
履歴だけを実履歴／空履歴に変えて `begin_unit` → 提案 → 同一候補群上の argmin を両側で行った差である。

## 検査

```bash
python3 scripts/smoke.py             # 検査 A〜C ＋ ゴール露出・運動分布・選択・履歴順・GAN 更新の確認
```

## 設計上の要点

- 音響評価：各解析窓（2048 frames、hop 0.1 s、窓は書き出し末尾を越えない）で全トラックの実 PCM から
  時間領域と 8 帯域の Gram 行列を事前計算し、任意のゲインベクトルに対する合算 PCM のエネルギー・帯域パワーを
  厳密な二次形式として得る（窓内はゲインを窓中心値で一定とみなす）。
- ゲインは各トラックとも Q5 ランプと ZERO_HOLD / GOAL_HOLD の列。各 Q5 は一つの単調運動エピソードで、
  接続点は反転・保持・開始・到達のみ。移動回数は段階時間と合法所要時間から決まり、固定上限で制限しない。
- ゴール保持区間は `g = e_0` を定数として保持し、出力が `B·x_0` と一致することを検証。
- 基準レベルは共通静的ゲイン `B = min(1, 0.85 / Σ peak_i)`。出力後のノーマライズは行わない。
