# TJA譜面生成ツールの使い方

このツールは、音源の特徴と既存のTJA譜面を使ってPPOモデルを学習し、そのモデルから新しいTJA譜面を生成する試作システムです。

基本的な流れは次のとおりです。

1. Python環境と依存パッケージを準備する
2. 音源と対応するTJA譜面を使ってモデルを学習する
3. 学習済みモデルと新しい音源からTJA譜面を生成する
4. 生成された譜面を確認し、必要に応じて手作業で調整する

## できること・現在の制約

- 音源入力形式はWAV、OGG、FLACです。
- 学習には、同じ名前（拡張子を除く）の音源と`.tja`ファイルのペアを使います。
- 生成結果は単一コースのTJAファイルです。
- 譜面の難易度は★1〜★10から指定できます。難易度制御は主にノーツ密度を基準にしており、長いリズム構造や複雑な譜面展開まで表現するものではありません。
- 学習時に短いドン／カッのパターンを既存譜面から集計し、生成時にも利用します。
- 大音符、連打、風船に対応しています。
- TJAの`#BPMCHANGE`、`#SCROLL`、正の`#DELAY`は読み書きできますが、新しい音源からこれらのギミックを自動予測する機能はありません。
- 生成後に譜面を検証し、ノーツ間隔や音源の長さなどの問題を可能な範囲で自動修正します。修正内容はコンソールに表示されます。
- 自動生成結果は完成譜面を保証するものではありません。ゲーム上で再生し、音ズレや配置を確認してください。

## 1. 動作環境の準備

Windows PowerShellでリポジトリのフォルダーに移動して、仮想環境を作成・有効化します。

```powershell
cd C:\Users\user\TJA-Gen
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

以後のコマンドは、仮想環境を有効化したPowerShellで実行してください。PowerShellの実行ポリシーにより有効化が拒否された場合は、仮想環境を有効化せず、次のように仮想環境内のPythonを直接実行できます。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe train.py
```

各コマンドの引数は、次のように確認できます。

```powershell
python train.py --help
python generate.py --help
```

## 2. 学習データを準備する

データフォルダー内に、音源と対応するTJAを同じファイル名で置きます。ファイルはサブフォルダーに分かれていても構いません。

```text
C:\Users\user\TJA-Gen-data\
├─ 曲A.ogg
├─ 曲A.tja
└─ サブフォルダー\
   ├─ 曲B.wav
   └─ 曲B.tja
```

音源とTJAの名前が一致しないペアは、PPO学習に使われません。学習時の音源形式はWAV、OGG、FLACです。

## 3. モデルを学習する

学習データの既定フォルダーは`%USERPROFILE%\TJA-Gen-data`です。既定では、対応する音源／TJAペアのうち最大8曲を使い、各音源の先頭30秒を対象として10エポック学習します。

```powershell
python train.py
```

ユーザーのデータフォルダーを指定する場合は、次のように実行します。

```powershell
python train.py --data-dir "C:\Users\user\TJA-Gen-data"
```

### データ全体を使って学習する

データフォルダー配下のすべての対応ペアを使い、各音源の全長を対象にするには、次のようにします。

```powershell
python train.py --data-dir "C:\Users\user\TJA-Gen-data" --max-tracks all --max-seconds all
```

データ全体を使うと処理時間が大きくなることがあります。まず既定設定で動作を確認してから、全件学習を実行してください。

### よく使う学習オプション

```powershell
# 最大20曲、各曲の先頭60秒、20エポックで学習
python train.py --data-dir "C:\Users\user\TJA-Gen-data" --max-tracks 20 --max-seconds 60 --epochs 20

# ★5を対象に固定して学習（指定しない場合は各エピソードで★1〜★10を選ぶ）
python train.py --level 5

# TJAパターン集計用の譜面フォルダーを別にする
python train.py --data-dir "C:\Users\user\TJA-Gen-data" --pattern-data-dir "D:\TJA-charts"
```

`--max-tracks`と`--max-seconds`には`all`を指定できます。`--level`は1〜10です。学習時の難易度を指定しない場合は、各エピソードでレベルが選ばれます。

学習中はエピソードごとの報酬、平均報酬、レベル、ノーツ数、密度、オンセット一致率などがコンソールに表示されます。既定の成果物は次のとおりです。

- `models/ppo_taiko.pt`：学習済みモデル。再学習時は同じパスのモデルが更新されます。
- `logs/training.csv`：学習メトリクス。既存CSVには追記されます。
- `graphs/training.svg`：現在の学習実行の報酬グラフ。

別の保存先を指定するには`--output`、`--log-file`、`--graph-file`を使います。

```powershell
python train.py --output "models/experiment.pt" --log-file "logs/experiment.csv" --graph-file "graphs/experiment.svg"
```

## 4. 音源からTJAを生成する

学習済みモデルが`models/ppo_taiko.pt`にある状態で、音源を指定します。

```powershell
python generate.py "C:\Users\user\TJA-Gen-data\新しい曲.ogg" --difficulty oni --level 8
```

難易度名は`Easy`、`Normal`、`Hard`、`Oni`、`Edit`から選べます。大文字・小文字は区別されません。レベルは1〜10です。

既定では、結果は`output`フォルダーに音源と同じベース名で保存されます。例えば、入力が`新しい曲.ogg`なら`output\新しい曲.tja`です。出力先を明示する場合は`--output`を指定します。

```powershell
python generate.py "C:\Users\user\TJA-Gen-data\新しい曲.ogg" --difficulty Oni --level 8 --output "C:\Users\user\Desktop\新しい曲.tja"
```

生成前にチャートを検証し、問題があれば可能な範囲で自動修正してから保存します。出力ファイルがすでに存在する場合は、検証に成功した内容で置き換えます。書き込みには一時ファイルを使うため、途中の未完成ファイルを最終出力として残しません。入力音源やモデルと同じパスへの出力は拒否されます。

TJAの`WAVE:`には音源のファイル名だけが記録され、フォルダーパスは含まれません。TJAを別の場所で使う場合は、TJAと音源を同じフォルダーに置くか、ゲーム側で音源を参照できるようにしてください。

## 5. BPMとOFFSETを指定する

`--bpm`を省略すると音源からBPMを推定します。`--offset`を省略すると`0.0`になります。どちらも個別に指定できます。

```powershell
# BPMとOFFSETを指定
python generate.py "song.ogg" --bpm 150 --offset -1.25

# BPMだけ手動指定、OFFSETは既定値
python generate.py "song.ogg" --bpm 150

# BPMは自動推定、OFFSETだけ手動指定
python generate.py "song.ogg" --offset -1.25
```

音源だけから譜面の正しいOFFSETを必ず推定できるわけではありません。譜面と音源の開始位置が合わない場合は、再生して確認したうえでOFFSETを指定してください。BPMは正の有限値、OFFSETは有限値である必要があります。

## 6. 生成オプション一覧

```text
python generate.py <音源ファイル> [オプション]
```

| オプション | 内容 | 既定値 |
|---|---|---|
| `--model` | 使用する学習済みモデル | `models/ppo_taiko.pt` |
| `--output` | 出力するTJAファイル | `output\<音源名>.tja` |
| `--difficulty` | コース名（Easy、Normal、Hard、Oni、Edit） | `Oni` |
| `--level` | 譜面レベル（1〜10） | `5` |
| `--bpm` | BPMを手動指定 | 音源から推定 |
| `--offset` | OFFSETを手動指定 | `0.0` |
| `--max-seconds` | 解析・生成に使う音源の最大秒数 | 音源全体 |
| `--seed` | 人間譜面パターンの抽選に使う乱数シード | `7` |
| `--maximum-density` | ノーツ密度の上限（ノーツ／秒） | 上限なし |
| `--device` | PyTorchの実行デバイス | `cpu` |

例えば、別のモデルで音源の先頭45秒だけを対象にするには、次のように実行します。

```powershell
python generate.py "song.wav" --model "models/experiment.pt" --max-seconds 45 --output "output\song.tja"
```

## 7. モデルに関する注意

モデルは学習したコードのState／Action仕様と一致している必要があります。StateやActionの構成が変わる更新より前に作成した古いチェックポイントは、現在のコードで読み込めない場合があります。その場合は、現在のコードでモデルを再学習してください。

生成コマンドで「モデルが見つからない」と表示された場合は、先に`python train.py`を実行するか、`--model`で存在するチェックポイントを指定してください。

## 8. テストを実行する

開発用テストは次のコマンドで実行できます。

```powershell
python -m pytest
```

すべてのテストが成功すれば、基本的な音源解析、TJAの読み書き、譜面検証・修正、学習メトリクス、生成処理のテストが通っています。
