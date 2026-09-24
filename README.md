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

## 高精細版（hires 拡張、2026-09-16 に利用者が許可）

仕様 v1.1 の枠を超える三つの操作を、**opt-in** で追加した（既定では無効。`hires.enabled=false`／`render.master.enabled=false` なら従来どおり）。

| 許可された操作 | 実装 |
|---|---|
| 急峻なスイッチ | 各確定区間（0.5 s）で素材の終了レベルを座標探索で決め、0.25 s の Q5 ランプで到達。運動上限（速度・加速度・ジャーク・低速規則）は適用しない（`hard_checks.slow_motion_rules = "waived"`）。値の連続性・0〜1・ゴール保持の厳密性は維持 |
| 再生位置の変更・切り貼り | 各素材は自分の音源内の別位置へジャンプできる（`CLIPS` 位置マップ、継ぎ目は 50 ms のクロスフェード）。候補位置は音源全体のソロ特徴バンクから、凍結参照の帯域プロファイルに近い位置＋探索用の乱数位置。最小クリップ長 2 s。ゴールは連続時計のまま |
| リミッター／コンプレッサー／ノーマライズ | 式(1)の生レンダリングでゴール一致を検証したあと、`latent_space/master.py`（RMS コンプレッサー → ピーク正規化 −1 dBFS → 先読みリミッター）を適用。適用量は trace の `hard_checks.master_chain` に記録 |

実現は `latent_space/hires.py`：窓（4 s）ごとに方式の参照を凍結し、位置ジャンプ候補（トラックごと）とレベルの組合せを、その位置での実 PCM から計算した窓 Gram（合算 PCM に対して厳密）で評価して最良を採用、先頭 0.5 s を確定して履歴へ返す。方式が見る素材特徴も実際に鳴っている位置のものに更新する。

周期：`scripts/sweep_period.py` が各方式について OPEN+CONTRACT ＝ 120／180／240 s を実行し、固定参照残差（全グリッド平均）が最小の周期を採用して `project.hires.<mode>.json` と `output_hires/<mode>/` を書く。

```bash
python3 -m latent_space generate --config project.hires.diffusion.json --mode diffusion --output output_hires/diffusion
```

**元に戻すには**：`project.json`（従来設定）で生成すればよい。高精細版は設定のみの切替で、従来パイプラインのコードは変更していない。

## 断片語彙版（fragment-vocabulary、2026-09-16 利用者承認）

高精細版の測定で、律速は実現層ではなく**理想側**（方式の参照が遅く、連続時計の probe に係留）と分かったため、
四方式の理想を「断片語彙」の上に置き直した版。opt-in（`hires.fragment_vocabulary: true`）で、従来版・高精細版 v1 は
そのまま残る（v1 の状態は git タグ `hires-v1`、出力は `output_hires_v1_backup/`）。

- **断片語彙**：各素材の 0.1 秒刻みの位置について「そこからジャンプしたら `clip_feature_seconds`（2 s）鳴る音」の
  平均特徴を持つ。再生の最小単位は `min_clip_seconds`（2 s）以上の連続区間で、0.1 秒の細切れにはならない。
- **厳密な断片配合**：`Analyzer.fragment_composition` が「各素材が指定位置から指定レベルで鳴る混合」の配合状態を、
  その位置の実 PCM の Gram から厳密に計算する。方式の基底・アンカー・トークン・参照分布はこれで作る。
- **探索**：ジャンプ候補は断片特徴（2 秒平均）と参照の帯域プロファイルの距離で選び、複数トラック同時ジャンプを
  ビーム探索（幅 3、行の部分集合で厳密混合を評価）で決め、生き残った組合せでレベル探索。
- **特徴空間**：24 帯域（対数間隔）、d_ξ = 41。方式の内部時間 `model_step_seconds` = 0.5 s（確定間隔と同じ）。
- **方式**：Diffusion は場の SDE を音楽時間で積分（アンカーは断片配合の標本平均）、VAE は断片配合のランダム標本の SVD 基底
  （k=4）、Transformer は断片トークンへの注意（値は断片を前景化した厳密配合）、GAN は断片配合の参照分布と系譜。

```bash
python3 scripts/sweep_period.py --base=project.frag.json --out=output_frag --tag=frag   # 周期 120/180/240 s を各方式で選ぶ
python3 scripts/ideal_vs_realized.py output_frag output_hires                              # 理想がどれだけ音に届いたか
```

## 保持参照・実現可能計画版（hold、2026-09-16 利用者指示）

利用者の指示「理想を 0.5 秒ごとに引き直さず 2 秒ほど固定して音に追わせる／理想の一手を素材の揺れより大きくする／
GAN の参照を確定履歴に近い断片へ絞る」の実装。opt-in（`hires.reference_hold_seconds: 2.0`）で、従来版・高精細 v1・
断片語彙版は設定を変えなければビット一致で動く（断片語彙版の状態はローカルタグ `frag-v1`）。方式契約は `docs/HOLD_CONTRACT.md`。

**測定で分かったこと**（`scripts/closeness.py`）：断片語彙版の理想は 0.5 秒ごとに「いま鳴っている配合」へ係留し直されるため、
理想が音を追いかけており、実現器が方式を無視しても残差の大半は自動的に小さくなっていた。理想を 2 秒固定するだけでは
Diffusion と GAN は追従しない（理想が 41 次元空間の「音量と位置では出せない方向」を向く）。一方、理想が**実現可能な計画の
厳密な配合**なら、実現器は要求された動きの 97〜99% を追える（検証モード）。律速は理想側の到達可能性だった。

- **エンジン（`latent_space/hires.py`）**：保持中は同じ凍結参照を追う（`prepare_reference` は保持ごとに 1 回、アンカーは直前確定
  ブロックの平均）。方式へ `unit.realizer_state`（現在のレベル・再生位置・ジャンプ可能時刻・厳密な計画評価器 `plan_rows`）を渡し、
  方式が `Target.meta["plan"]` に入れた計画（確定格子上のジャンプ位置と終了レベル）をジャンプ候補・レベル探索の出発点に加える。
  採否は従来どおり凍結参照に対する目的関数が決める。候補は「いまの選択＋この先は方式の計画どおり」として先読み窓で評価する
  （`hires.plan_aware_lookahead`。「いまの音量が 4 秒続く」と仮定すると、0.5 秒ごとに音量が変わる計画とは比べられず、
  なめらかな計画の方式だけが有利になる）。保持ごとのアンカーと要求移動量は trace の `hold_segments` / `hold_summary`。
- **Diffusion**：自由な SDE を、実現可能な計画の上の Langevin／Metropolis 連鎖に置換（場のエネルギー E_D と開放度温度は同じ）。
- **VAE**：潜在 OU 経路とデコーダはそのまま、D(z) を実現可能集合へ射影した計画を公開（射影残差を記録）。
- **Transformer**：注意が断片トークンを平均せず**選択**し、選ばれた断片への移動が計画の一手になる（記憶ヘッドは既出断片への回帰）。
- **GAN**：参照分布＝現在の状態から続く実現可能な計画で、確定履歴の近くに絞ったもの。生成器は親計画とレベル空間の変位を選ぶ。

```bash
sh scripts/run_hold.sh                       # 四方式を output_hold/<mode>/ へ生成＋「方式を無視」対照＋計測表
python3 scripts/closeness.py output_hold dev/out/ignore_hold output_frag
```

`closeness.py` の読み方：`gauge` は理想までの距離（ランダム断片ミックス＝100、完全一致＝0）、`achieved` は保持 1 回ごとに
理想が要求した移動のうち音が実際に達成した割合、`cos (chance)` は移動方向の一致と偶然水準、`change corr in-hold` は保持内の
0.5 秒ごとの変化の相関。**元に戻すには** `project.frag.<mode>.json`（断片語彙版）などの旧設定で生成するだけでよい。

## 忠実度版（fid、2026-09-17 利用者指示：A・B・C ＋ 同一素材の 2 か所同時再生 ＋ 素材 8〜12 本への対応）

保持版で「計画 → 音」の損失はなくなった（達成率 0.96〜0.99 は構造上ほぼ保証される値）。残る距離は「方式の法則 → 計画」に
あるので、そこを方式ごとに改修した。方式契約は `docs/FIDELITY_CONTRACT.md`。設定は `project.fid.<mode>.json`
（＝保持版の設定＋ `hires.voices_per_material: 2`、`hires.explore_tracks_max: 3`）、出力は `output_fid/<mode>/`。
保持版 v1 の出力は `output_hold_v1_backup/` に退避してある。

- **同一素材の 2 か所同時再生**（`hires.voices_per_material: 2`）：各素材に 2 本目のトラックを与える（同じ PCM・同じ断片語彙、
  半周ずれた位置から開始）。トラック数は 1＋2N、CSV／trace では `素材名#2`。静的ピーク上界（基準レベル B）は全トラックで数える。
- **素材 8〜12 本**：エンジンと四方式はトラック数を仮定しない（素材 10 本の fixture で四方式とも動作確認）。速度のために
  厳密評価（Gram）にトラック別キャッシュを入れ（結果はビット一致のまま）、実現器の独自ジャンプ探索を 1 確定あたり
  `explore_tracks_max` トラックに絞れるようにした（方式の計画が名指ししたトラックは常に評価する）。
- **混合を見越した断片検索** `Analyzer.fragment_candidates_mix`：他トラックが鳴らしているものを踏まえ、混合全体が目標に最も近づく
  断片を全断片から順位付けする（約 1 ms）。
- **計測の修正**：確定窓の先頭行のスペクトル変化量が 0 に落ちる癖を直した（保持モード限定）。達成率から「素材が勝手に変化する分」を
  除いた `achieved_net`（計画の介入分のうち音に現れた割合）を `closeness.py` に追加した。
- **Diffusion（A）**：保持ごとに温度を下げながら複数回の内部反復（粗＝断片、細＝音量のランジュバン・ドリフト）で計画を作る。
- **Transformer（A）**：1 手で複数トラックが動く（注意質量の大きいトラックがそれぞれ自分のトークンを選び、音量は注意の重みから
  初期化して厳密評価で微調整）。`hold_law_version: 1` で従来の 1 トークン則に戻せる。
- **VAE（B）**：潜在空間を「出せる音」の上に作り直した。制御 u＝［各トラックの音量｜鳴らす断片の PCA 座標］、厳密な配合との
  PLS-SVD で k 本の潜在軸、デコーダは z → 制御 → 厳密な配合（射影の損失は構造上ゼロ）。`hold_latent_space: xi` で従来の射影版。
- **GAN（C）**：識別器の「本物」を、乱数で作った計画から**手を加えない録音の動き**（音の近くの位置から一定音量でそのまま
  再生した窓）に替えた。特徴量に動きの項を入れ、生成器の報酬は飽和しない形（計画行での D のロジット）にした。
  `hold_positive_source: plans` で従来版。

```bash
TAG=fid OUT=output_fid COMPARE="output_hold output_frag" sh scripts/run_hold.sh     # 四方式＋対照実験＋計測表（約 25 分）
```

### fid 版の追補（2026-09-17、利用者指示の 3 点）

- **Transformer 法則 3**（`hold_law_version: 3`、既定）：「注意が向いているものが聞こえる」。ヘッドごとに z スコアで正規化してから結合し
  （記憶ヘッドがトークン数で勝つのを防ぐ）、各素材トラックの終了音量＝正規化した注意質量の `hold_level_sharpness` 乗、確率的な「動かない」
  ゲートは廃止（開放度が移動量を決める）。指標 `attention_contribution_corr`＝注意質量と厳密な寄与ベクトル c のトラック間相関。
- **Diffusion の反復抑制**：確定したジャンプ位置を音源ごと（同じ素材の 2 声は共有）に記憶し、訪問密度（幅 2 s、半減期 60 s）を断片の順位付けの
  ペナルティと連鎖の「新規性リッジ」項に入れる（`hold_repetition_weight` ほか）。場エネルギーの統計には含めない。
- **保持長と周期の掃引** `scripts/sweep_fid.py`：残差ではなく方式固有の指標で選ぶ（diffusion＝確定／ランダムの場エネルギー比、vae＝潜在追従、
  transformer＝注意→寄与の相関、gan＝D(本物の録音)−D(公開計画)）。採用規則：介入量が素材の揺れの 2 倍以上の設定のうち指標が最良のもの。
  現在の採用値との差が 2% 以内なら据え置く。

| 方式 | 保持長 1／2／4 s の指標（介入／揺れ） | 採用 | OPEN 90／150／210 s の指標 | 採用（周期） |
|---|---|---|---|---|
| diffusion | 0.37（1.9）／0.42（3.0）／0.49（2.4） | 2 s | 0.416／0.417／0.408 | 90 s（120 s、差 2% 以内で据え置き） |
| vae | 0.71（1.8）／0.79（3.2）／0.81（3.5） | 4 s | 0.794／0.812／0.806 | 150 s（180 s） |
| transformer | 0.51（1.0）／0.50（1.6）／0.51（2.5） | 4 s | 0.510／0.548／0.546 | 150 s（180 s、従来 120 s から変更） |
| gan | 0.05（2.4）／0.04（2.5）／0.15（2.1） | 2 s | 0.120／0.060／0.044 | 210 s（240 s） |

```bash
python3 scripts/sweep_fid.py vae gan --holds=1,2,4                    # 保持長の比較
python3 scripts/sweep_fid.py vae --holds=4 --opens=90,150,210         # 周期の掃引
```

### 同時に鳴る素材数の上限（2026-09-17 利用者判断）

素材を 8〜12 本に増やす前提で、**同時に鳴る素材数に上限**を設けた。すべて設定で変更できる（`project.fid.<mode>.json` の `hires`）：

| キー | fid 設定の値 | 意味 |
|---|---|---|
| `max_active_materials` | 3 | 同時に鳴ってよい素材数（同じ素材の 2 声は 1 素材と数える）。0＝上限なし |
| `min_sounding_seconds` | 2.0 | 鳴り始めた素材は最低この秒数は鳴り続ける（最小クリップ長の「聞こえる側」の対。0＝規則なし）。CONTRACT 末尾のゴール立ち上がりだけは例外 |
| `cap_hysteresis` | 0.0 | すでに鳴っている素材を順位付けで優遇する音量差（大きいほど顔ぶれが落ち着く） |
| `cap_swap_candidates` | 2 | 実現器が独自に試す入れ替え候補の数 |

- 上限は各確定（0.5 s）の**終了音量**に掛かる。0.25 s のランプで出る素材と入る素材がクロスフェードする間だけ一時的に上限を超えてよい
  （最大で上限の 2 倍、利用者判断）。鳴っている最中のジャンプは従来どおり可。
- 仕組み：`project_levels`（音量の大きい K 素材を残し、他の全声を 0 にする）を計画評価器 `plan_rows`・計画を見越した先読み・ヒント・実現器の
  音量探索のすべてに掛ける。無音の素材は「鳴っている 1 素材と入れ替わる」ことでしか入れない。**入る声は無音のうちに再生位置を変えられる**ので、
  入りの継ぎ目は聞こえない。方式は `unit.realizer_state` の `max_active_materials / material_of_track / project_levels / min_sounding_frames /
  entered_at` を見て、法則で「どの素材を鳴らすか」を選ぶ（契約：`docs/FIDELITY_CONTRACT.md` 末尾）。
- ハード検査 `hard_checks.polyphony_cap`：50 ms 格子で、ランプ外の同時発音素材数 ≤ 上限、最短の発音区間 ≥ `min_sounding_seconds`、
  発音数ごとの時間割合を記録。
- 効果（素材 10 本の fixture、上限 3）：顔ぶれの入れ替えは規則なしで毎分 76／84／82／24 回（diffusion／vae／transformer／gan、発音区間の中央値
  0.5〜1.0 s）→ `min_sounding_seconds: 2` で 40／63／38／18 回（中央値 2.0〜4.8 s）。

## 実素材 13 本の Diffusion 生成（2026-09-23 利用者指示）

利用者提供の `diffusion-materials-001.zip`（MP3 13 本、44.1 kHz／320 kbps、18〜129 s）を `materials/diffusion-001/` に展開し、
`sh scripts/prepare_inputs.sh materials/diffusion-001 inputs/diffusion-001` で WAV 化（スクリプトは `SRC_DIR OUT_DIR` 引数を受ける。引数なしは従来どおり）。
設定 `project.diff001.diffusion.json`（= fid＋上限の設定に素材 13 本、OPEN 150 s＝周期 180 s、4 周期、全長 872 s）、出力 `output_diff001/diffusion/`。
結果：BEST_EFFORT、ハード検査合格、ゴール 4 回とも完全一致、ゲージ 8.7、正味の達成率 +0.99、場エネルギー 0.21〜0.26（合法ランダム 0.47〜0.84）、
3 素材が鳴る時間 83%、顔ぶれの入れ替え 50 回／分。詳細は `docs/HANDOFF_20260917.md` §15。

同時に `latent_space/curves.py` の曲線参照（`base_values / derivatives / positions`）を、全セグメント走査から該当区間だけの走査に変えた。
全曲レンダリングとハード検査（最終段）の費用がセグメント数に比例して増えていたため（13 本・15 分で所要 52 分、うち最終段 26 分）。
出力はビット一致（従来 fixture 3 本＝`dev/compare_legacy.py`、短形式と全長の再生成）。所要 52.3 → 43.0 分（実現層 25 分は不変、最終段 26 → 18 分）。

同日、素材を 24 本に増やした `diffusion-materials-002.zip`（追加 11 本は papillon／prism の viola・violin 別ステム）でも同じ形式で生成した：
`materials/diffusion-002/` → `inputs/diffusion-002/`、設定 `project.diff002.diffusion.json`（**1 声**：2 声＝49 トラックはメモリ 13 GB 超の見込み）、
出力 `output_diff002/diffusion/`。BEST_EFFORT、ハード検査合格、場エネルギー 0.25〜0.33（合法ランダム 0.62〜1.50）、3 素材が鳴る時間 78%、
顔ぶれの入れ替え 63 回／分（発音区間の中央値 2.0 s）、所要 37 分。詳細は `docs/HANDOFF_20260917.md` §16。

## ゴール露出方針 `contract_law`（2026-09-23 夕、利用者指示「案 d」。opt-in、既定は無変更）

`form.goal_exposure.policy: "contract_law"` にすると、CONTRACT の間だけゴール音源が**法則の選べる素材**になる：実現層はトラック 0 を素材と同じ
「探索されるトラック」として扱い（音量のみ、位置ジャンプなし、`law_monotonic` で非減少）、最終立ち上がり（`hires.goal_rise_seconds`）は法則が到達した
音量から 1 へ、GOAL_HOLD は厳密のまま。Diffusion の保持連鎖は計画にゴール音量の座標を持ち、場のエネルギーは**素材の項（アンカー・ペア・三者・リッジ）を
ゴール抜きの配合**（`plan_rows` の `info["xi_materials"]`）で、**ゴール項 λ_G(1−o)² d²(ξ, ξ_goal) だけを全体の配合**で評価する。ゴール座標は雑音なしの
ドリフトだけで、絶対移動度 Δg = −d_lvl·∂E/∂g（|Δg| ≤ d_lvl）で動く。実現層が方式に渡す評価（`window_error(..., xi_materials=)`）も同じ分離。
`objective.contract_goal_weight_exponent`（既定 2.0＝従来）で CONTRACT の重み (1−o)^p の指数を変えられる（形式項と Diffusion のゴール項の両方。大きくするとゴールの出現が後ろへ寄り急になる。180 s 形式で 2:30 頃なら p≈40）。他の三方式は未対応（計画のゴール音量を
動かさない）。契約は `docs/FIDELITY_CONTRACT.md` 末尾、経過と結果は `docs/HANDOFF_20260917.md` §17。従来設定の出力はビット一致
（従来 fixture 3 本、保持経路 107 s、1 周期 T2.0）。

## 2026-09-23 のまとめと引き継ぎ

ゴール露出方針 `contract_law`（CONTRACT 中はゴール音量を Diffusion の法則が選ぶ）と収束の物差し `convergence: sparsity / presence`、指数の分離
（素材側 p、ゴール進入 q）、抜ける手（drop）を opt-in で実装。現在の状態・設定キー・結果・留保・次の候補は `docs/HANDOFF_20260923.md`（§13 が
次セッション冒頭の再出力ブロック）、詳細経緯は `docs/HANDOFF_20260917.md` §15〜§17。
