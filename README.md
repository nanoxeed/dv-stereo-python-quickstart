# dv-stereo-python-quickstart

iniVation のステレオイベントカメラを Python からセットアップ、表示、キャリブレーション、点群確認するためのクイックスタートです。

## セットアップ

```bash
uv sync
```

## スクリプト

### 生イベントのステレオ表示

```bash
uv run python stereo_camera_capture.py
```

左右のイベントストリームを `dv.visualization.EventVisualizer` で表示します。カメラ接続、左右同期、イベントが出ているかを最小構成で確認するためのスクリプトです。

終了するには `Esc` を押します。

### Accumulated Image のステレオ表示

```bash
uv run python accumulate_stereo_preview.py
```

左右のイベントを accumulator で画像化して表示します。キャリブレーション前に、チェスボードのエッジが見えやすい accumulation パラメータを探す用途で使います。

例:

```bash
uv run python accumulate_stereo_preview.py --accumulator generic --contribution 0.05 --decay 1000000
```

終了するには `Esc` を押します。

### ステレオキャリブレーション

キャリブレーション用チェスボードは `assets/calibration/` にあります。

- `chessboard_9x6_30mm.svg`: A4 横向きのベクター画像。印刷用はこちらを推奨します。
- `chessboard_9x6_30mm_300dpi.png`: 300dpi の PNG 版です。

```bash
uv run python stereo_calibrate.py --detector sb
```

スクリプトが開始されると、左右のカメラ画像が表示されます。
両方のカメラでチェスボードが検出され、マッチングが取れたら自動的に画像がストアされていきます。
20枚ストアが溜まったところで、一旦撮影が終了し、どの画像を採用するかの確認を行います。

確認では以下のキーで採用を決めます。
- `Space` または `k` を押すと、現在の画像を採用します。
- `d` を押すと、現在の画像を棄却します。

終了すると、キャリブレーション結果が `calibration/stereo_calibration.json` に保存されます。

### Rerun 点群ビューア

キャリブレーション結果を用いて、Disparity画像と点群を表示します。

```bash
uv run python stereo_pointcloud_rerun.py --calibration calibration/stereo_calibration.json
```
