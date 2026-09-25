# 次セッションの最初のメッセージ（利用者がそのまま貼り付ける）

前セッションからの引き継ぎです。作業ディレクトリは `/Users/goushiyonekura/Claude/latent-space-material`（git、リモート https://github.com/goushiyonekura/latent-space-material 、ブランチ `main`、最新タグ `scoremap-v15a`）。

最初に次のことを行ってください。

1. `docs/HANDOFF_20260925.md` を全文読むこと。特に §4（正典 v15a の確定ルール）、§7（検査と再現）、§8（未達・保留中の選択肢）は細部まで把握すること。必要に応じて `docs/HANDOFF_20260924.md`（譜面マッピングの成り立ち §6、対応づけ §6.5、このセッションの経過 §11〜§13）とメモリ（`~/.claude/projects/-Users-goushiyonekura-Claude-latent-space-material/memory/`）も読むこと。
2. **最初の応答として、同文書 §10 のコードブロックの中身を一字一句そのまま出力すること**（「**引き継ぎ状況**」から「指示をお待ちします。」まで。前置きや要約は付けない）。
3. そのあと私の指示を待つこと。指示が来るまでコードの実行・変更・生成はしないこと。`python3 scripts/smoke.py` も走らせないこと。

前提の再確認：正典は `output_diff003/diffusion_1cycle_slow_law_T2_presence_keep_s3/scoremap/scoremap_keep_s3_v15a_1passage.musicxml`（v15a：1 段に同時 1 パッセージ、緩和ルールの切り出し、先頭 9 秒削除、110 秒で改ページ、ページごとに空段を非表示）。元譜面の表記を変えてよいのは §4 (k) に列挙した範囲だけで、それ以外の音符・リズム・連桁・スラーなどは改変しない。MusicXML を納品する前には必ず verify・tiecheck・`xmllint --schema`（Sibelius 同梱の XSD）を通す。報告は日本語で、実働部分と未達を明記すること。
