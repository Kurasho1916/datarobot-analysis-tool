# datarobot-analysis-tool
DataRobotモデルの特徴量分析ツール
社内用の特徴量分析ダッシュボード（Streamlit製）

## 🎯 機能

- **接続診断**: DataRobotへの接続確認とプロジェクト・モデル選択
- **特徴量分析**: Feature ImpactとSHAP値の可視化
- **相関プロット**: 特徴量と予測への影響を可視化
- **SHAP分布**: 特徴量ごとのSHAP値分布を表示

## 🔐 アクセス方法

**URL**: https://[あなたのアプリURL].streamlit.app
**パスワード**: 管理者に問い合わせてください

## 📝 使い方

### 1️⃣ 接続診断モード
1. APIトークンとエンドポイントを入力
2. 「診断開始」をクリック
3. 目的変数を選択
4. プロジェクトとモデルを選択

### 2️⃣ 分析実行モード
1. 「データ読み込み」をクリック
2. Feature Impact、SHAP分析を確認

## ⚙️ 設定項目

| 項目 | 説明 |
|------|------|
| APIトークン | DataRobotのDeveloper Toolsから取得 |
| エンドポイント | 通常は `https://app.datarobot.com/api/v2` |
| モデルID | 接続診断モードで自動取得 |

## 📌 注意事項

- **パスワードは社外に漏らさないこと**
- プロキシ経由でのアクセスが必要
- データは一時的にメモリに保存（永続化なし）

## 🛠️ 技術スタック

- **Streamlit**: Webアプリケーションフレームワーク
- **DataRobot SDK**: DataRobot API接続
- **pandas/matplotlib**: データ処理と可視化
