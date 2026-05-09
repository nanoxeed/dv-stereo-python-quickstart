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

### ステレオ画像の保存

```bash
uv run python capture_stereo_images.py --count 5
```

`Space` または `s` を押すと、現在の左右 accumulated image を保存します。キャリブレーションファイルを指定した場合は、rectified image も保存します。

```bash
uv run python capture_stereo_images.py --calibration calibration/stereo_calibration.json
```

### ステレオキャリブレーション

キャリブレーション用チェスボードは `assets/calibration/` にあります。

- `chessboard_9x6_30mm.svg`: A4 横向きのベクター画像。印刷用はこちらを推奨します。
- `chessboard_9x6_30mm_300dpi.png`: 300dpi の PNG 版です。

```bash
uv run python stereo_calibrate.py --pattern chessboard --pattern-size 8x5 --square-size 30
```

`--pattern-size` は OpenCV が検出する点数です。`9x6` squares のチェスボードでは、内側コーナー数として `8x5` を指定します。

`--square-size` はデフォルトではミリメートルとして扱い、dv-processing の depth geometry 用に内部でメートルへ変換します。最初からメートルで指定したい場合は `--square-size-scale-to-meters 1.0` を追加してください。

SVG を印刷するときは、100% スケールで印刷し、`fit to page` や「用紙に合わせる」設定は無効にしてください。

このスクリプトは DV-GUI のキャリブレーションと同じ考え方で動きます。

- 2台のカメラを同期する
- イベントを accumulated image に変換する
- 左右画像でキャリブレーションパターンを検出する
- 連続フレームで安定して検出されたサンプルだけを保持する
- 必要に応じてサンプルを目視確認する
- OpenCV の mono calibration と stereo calibration を実行する
- `calibration/stereo_calibration.json` を dv-processing 用に保存する
- `calibration/opencv_calibration.npz` を OpenCV での検証用に保存する

対応パターン:

```bash
--pattern chessboard
--pattern circles
--pattern asymmetric-circles
```

### Rerun 点群ビューア

```bash
uv run python stereo_pointcloud_rerun.py --calibration calibration/stereo_calibration.json
```

`dv.SemiDenseStereoMatcher` で disparity/depth を推定し、左右 accumulated image、disparity preview、点群を Rerun に表示します。Open3D は使っていません。

よく使うオプション:

```bash
uv run python stereo_pointcloud_rerun.py \
  --calibration calibration/stereo_calibration.json \
  --max-depth-m 3.0 \
  --max-points 80000
```

## カメラ選択

多くのスクリプトでは、左右カメラを discovery index で指定できます。

```bash
--left 0 --right 1
```

カメラ名で指定することもできます。

```bash
--left DVXplorer_DXA00000 --right DVXplorer_DXA00001
```

点群ビューアは、キャリブレーションファイルに保存されたカメラ名を使ってカメラを開きます。これにより、キャリブレーションした個体とライブ接続中の個体が一致するようにしています。
