# 《潜在空間》作曲素材生成ツール — headless / gain-only / spec v1.1 (H5)

固定音声ループ＋連続音量制御だけで、`OPEN → CONTRACT → GOAL_HOLD → REOPEN` の周期を持つ
オフライン音響を生成する CLI。GUI なし、依存は **Python 3.9 + numpy** のみ（GPU 不要）。

再監査（コード002）対応版：**四方式の理想配合を固定した上で、実際の音響誤差に沿って合法な音量軌道を
局所的に改善し、短い確定区間ごとに採用結果を履歴へ返す**。理想を候補へ引き寄せる更新は行わない。

```
latent_space/            エンジン本体
  audio_io.py            WAV 読み書き（PCM 8/16/24/32, float32/64）
  curves.py              Q5 ランプ・ホールド（解析的検査）、局所補正 Bump = a·64s³(1−s)³、
                         実微分 state_at / derivatives、合成曲線の実エピソード検査 check_composite
  form.py                INTRO/OPEN/CONTRACT/GOAL_HOLD/REOPEN の整数フレーム計画、開放度 o(t)
  analysis.py            解析窓ごとの Gram 行列（実 PCM の和の厳密な二次形式）→ 配合状態 ξ、d_ξ、
                         レンダリング PCM からの配合再計算（composition_from_render）
  bank.py                合法な共同ゲイン軌道の初期解（暖機解）、sync（全区間・不等号修正済）／counter（内点区間の局所関係）
  objective.py           E_form（OPEN退化のSOFT項 N_eff・実エネルギー含む）/ E_hist / E_motion / J_rel
  history.py             履歴 H：時定数 τ_H による確定区間ごとの更新 ρ(Δt)=1−exp(−Δt/τ_H)、時刻順イベント
  modes/                 diffusion.py / vae.py / transformer.py / gan.py（四方式の制御器）
  engine.py              暖機解 → 確定区間ループ（窓参照の凍結・局所補正の座標探索・確定・履歴・方式観測）→ 検査・trace
  render.py              式(1) だけを行う正式レンダラー
  trace.py               result.wav / gain_curves.csv / state_trace.json
  fixtures.py            人工 WAV 素材の生成（開発・検査用）
scripts/smoke.py         検査 A〜C ＋ 監査対応の確認（sync/counter、固定参照改善、確定区間、非ゼロ速度接続）
scripts/prepare_inputs.sh  MP3 → WAV デコード（macOS afconvert）
scripts/run_all.sh       四方式一括生成
project.json             実素材用設定（182 秒の比較用形式、旧結果再現用）
project.explore.json     実素材用の長尺形式（OPEN 320 秒、合計 782 秒、素材が一度周回する検討用）
docs/MODE_CONTRACT.md    方式実装の契約（再監査版）
```

## 実行

```bash
pip3 install --user numpy
sh scripts/prepare_inputs.sh         # mp3 → inputs/*.wav（済み）
python3 -m latent_space generate --config project.json --mode diffusion --output output/diffusion
python3 -m latent_space generate --config project.explore.json --mode diffusion --output output_explore/diffusion
```

`--mode` は設定の `mode` を上書き、`--seed` で seed 上書き可、`--out` は `--output` の別名。

## 実現層（監査 C4〜C7）

1. **暖機解**：Bank の合法な全ユニット候補に対し、方式の全ユニット提案から一つを到達性で選んで凍結（R0、hash 記録）し、R0 に対する最良候補と有限回の変異を暖機解とする。
2. **確定区間ループ**：`realization.commit_seconds`（既定 1.5 s）ごとに、先読み窓（`lookahead_seconds` 10 s）の参照を
   方式が `prepare_reference`（確定履歴＋実現済み現在配合から）で提案し、現在の尾部に対する J で一つを選んで**凍結**。
   凍結参照に対し、局所補正（各トラック 2 本、端で値・速度・加速度 0）の振幅を有限差分の座標探索で改善する
   （受理は改善時のみ、合成曲線の実エピソード検査を通過した変更のみ）。先頭区間を確定し、その実配合を
   `history.observe_committed`（ρ(Δt)）へ入れ、完了した運動イベントを時刻順に追加し、方式の `observe_committed` を呼ぶ。
3. **記録**：各ステップの `reference_hash`、初期／最終の固定目標誤差と共同目的、項別内訳、受理数、`reference_updated_during_realization=false`。
   全グリッドの固定参照残差、確定行で実際に使った参照列（`fixed_reference_rows`）、代理／固定目標／実レンダリングの三者距離
   （各段階 20/50/80% とゴール境界の窓、flux は隣接前窓から）。

`mode.update()`（実現候補への目標引き寄せ）はエンジンから呼ばれない。

## 設定（主なキー）

| キー | 既定値 | 意味 |
|---|---|---|
| `realization.commit_seconds` / `lookahead_seconds` | 1.5 / 10 | 確定長と先読み窓（試験開始案、知覚閾値ではない） |
| `realization.max_refinement_sweeps` / `initial_step` / `max_evaluations_per_step` | 3 / 0.06 / 80 | 局所補正の座標探索 |
| `realization.w_relation` / `w_smooth` | 0.25 / 0.01 | 関係項 J_rel（方式供給）と運動エネルギーの重み |
| `history.time_constant_seconds` | 60 | 履歴の時定数 τ_H |
| `analysis.model_step_seconds` | 0.1 | 方式の内部時間（解析 hop と分離） |
| `objective.w_neff` / `n_eff_target` / `w_energy` / `energy_min_ratio` | 0.25 / 2 / 0.25 / 0.05 | OPEN 退化の SOFT 項（強制ではない） |
| `form.goal_exposure.*` | contract_only, 0, 0 | ゴール露出方針（INTRO/OPEN は 0、CONTRACT で連続上昇、REOPEN は 1 を継承） |
| `motion.profile` | baseline | `responsive_x2_UNVERIFIED` で k=2 の時間短縮則（V·k, A·k², J·k³, L·k、聴覚未承認） |
| `mode_defaults.vae_latent_dim` | 2（実素材設定では 3） | 共有潜在因子数。probe 差分の分散捕捉率を trace に記録 |

別名 `schedule.*`→`form.*`、`motion.relative_rate_max`、`motion.ramp_component_duration_max_seconds`、
`search.max_rounds`、`search.candidate_count` は正本キーへ変換され、未知キーはエラー。

## 出力（各 output/<mode>/）

| ファイル | 内容 |
|---|---|
| `result.wav` | 式(1) でレンダリングした float32 WAV |
| `gain_curves.csv` | 0.05 秒刻みの 0〜100 表示ゲイン |
| `state_trace.json` | 正本：解析的曲線（Q5/HOLD 区間＋局所補正 `BUMPS`）、設定、素材ハッシュとループ回数、履歴要約、方式内部状態、ハード判定、ステップ記録、固定参照、ソフト残差、`run_status` |

trace の曲線から再レンダリングすると `result.wav` とビット一致する（検査 C）。

## 状態ラベル

- `VALID_APPROXIMATION`：ハード条件を満たし、採用軌道が tolerance 以下。
- `BEST_EFFORT`：ハード条件を満たすが近似目安が未達（正常出力）。
- `HARD_CONSTRAINT_FAILURE` / `INCOMPLETE_IMPLEMENTATION` / `INPUT_ERROR` / `INTERNAL_ERROR`。

`calibration_status` / `perceptual_status` は常に `UNVERIFIED`。

## 検査

```bash
python3 scripts/smoke.py
```
