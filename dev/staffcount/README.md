# 段数分析のスクリプト（2026-09-25、docs/HANDOFF_20260924.md §12〜§13）

セッションの一時領域（scratchpad）で書いたものをそのまま置いた。パス定数 `S = "/private/tmp/claude-501/.../scratchpad"` を
このディレクトリに読み替えて使う（`units2.pkl` などの中間ファイルは `twovoice_scenarios.py` が作る）。

- `twovoice_analysis.py` / `twovoice_scenarios.py` / `twovoice_more.py`：1 段 2 声部での段数の下限・貪欲構成（§12）
- `sounding_cuts.py`：「鳴っている時間だけ」を切れ目で実現した場合（§12.1）
- `relaxed_cuts.py` / `relaxed_clip.py` / `relaxed_cap4.py`：緩和ルール A1/A2/B1/B2（§12.2）
- `voices_variants.py`：1 段の声部数 2/3/4 と luminasity の上限（§12.4）
- `prism_steps.py`：prism の DTW ステップ罰則の比較（§12.5）
- `plan_test.py` / `pack_test.py`：`scoremap.merge.plan` / `pack` の単体実行
