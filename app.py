import streamlit as st
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# =============================================================================
# プロキシ・証明書設定
# =============================================================================
CA = r"C:\Users\024044\OneDrive - 株式会社ＧＳユアサ\デスクトップ\DX道場\AI道場\www.globalsign.crt"
os.environ["HTTP_PROXY"] = "http://172.17.20.158:3128"
os.environ["HTTPS_PROXY"] = "http://172.17.20.158:3128"
os.environ["REQUESTS_CA_BUNDLE"] = CA
os.environ["SSL_CERT_FILE"] = CA

# import datarobot as dr
# from datarobot.enums import INSIGHTS_SOURCES
# from datarobot import insights

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Streamlit設定
# =============================================================================
st.set_page_config(page_title="DataRobot分析ツール", layout="wide")

# =============================================================================
# Session State初期化
# =============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_impact_df' not in st.session_state:
    st.session_state.feature_impact_df = pd.DataFrame()
if 'shap_impact_df' not in st.session_state:
    st.session_state.shap_impact_df = pd.DataFrame()
if 'shap_distributions_df' not in st.session_state:
    st.session_state.shap_distributions_df = pd.DataFrame()

# =============================================================================
# サイドバー設定（最小限）
# =============================================================================
st.sidebar.header("⚙️ 接続設定")

mode = st.sidebar.radio(
    "モード選択",
    ["🔍 接続診断", "📊 分析実行"]
)

API_TOKEN = st.sidebar.text_input(
    "APIトークン", 
    value="Njk2ZGNlZTBkZWU3NzcxNzBhYjhkN2VhOk5yRFMyU3kwQzlmMlJMZ05pWWQ5am5sbzlyNVJMakZ2WEFsVU82ZjlBUG89",
    type="password",
    help="DataRobotのDeveloper Tools → APIキーから取得"
)

ENDPOINT = st.sidebar.text_input(
    "エンドポイント",
    value="https://app.datarobot.com/api/v2",
    help="通常は https://app.datarobot.com/api/v2"
)

MODEL_ID = st.sidebar.text_input(
    "モデルID",
    value="695c981386dbc28805fcd879",
    help="DataRobotプロジェクトのモデルIDを入力"
)

# =============================================================================
# モード1: 接続診断
# =============================================================================
if mode == "🔍 接続診断":
    st.title("🔍 DataRobot 接続診断")
    st.info("DataRobotへの接続状態を詳しく診断します")
    
    if st.button("🚀 診断開始", type="primary"):
        
        # ========== 1. 証明書ファイル確認 ==========
        st.header("1️⃣ 証明書ファイルの確認")
        if os.path.exists(CA):
            st.success(f"✓ 証明書ファイルが存在します")
            st.code(CA)
        else:
            st.error(f"❌ 証明書ファイルが見つかりません")
            st.code(CA)
            st.stop()
        
        # ========== 2. プロキシ接続テスト ==========
        st.header("2️⃣ プロキシ接続テスト")
        try:
            response = requests.get(
                "https://www.google.com",
                proxies={
                    "http": "http://172.17.20.158:3128",
                    "https": "http://172.17.20.158:3128"
                },
                verify=CA,
                timeout=10
            )
            st.success(f"✓ プロキシ接続成功（ステータス: {response.status_code}）")
        except Exception as e:
            st.error(f"❌ プロキシ接続エラー")
            st.exception(e)
            st.warning("プロキシ設定またはネットワーク接続を確認してください")
        
        # ========== 3. DataRobot SDK認証テスト ==========
        st.header("3️⃣ DataRobot SDK認証テスト")
        
        auth_success = False
        
        try:
            # DataRobot SDK接続
            dr.Client(endpoint=ENDPOINT, token=API_TOKEN)
            st.success("✓ DataRobot SDK接続成功")
            
            # プロジェクト取得で認証確認
            try:
                projects = dr.Project.list()
                st.success(f"✓ 認証成功（アクセス可能なプロジェクト: {len(projects)}件）")
                auth_success = True
                
                # ユーザー情報表示
                st.info("📌 認証情報が正常に機能しています")
                
            except Exception as e:
                st.error(f"❌ プロジェクト取得失敗: {e}")
                st.stop()
        
        except Exception as e:
            st.error(f"❌ SDK接続エラー")
            st.exception(e)
            
            st.warning("### 🔧 トラブルシューティング")
            st.write("1. **APIトークンを確認**")
            st.write("   - DataRobot → Developer Tools → APIキー")
            st.write("   - 「dx_python」キーが「アクティブ」になっているか確認")
            st.write("")
            st.write("2. **エンドポイントを確認**")
            st.code("https://app.datarobot.com/api/v2")
            st.write("")
            st.write("3. **プロキシ・証明書設定を確認**")
            st.code(f"証明書: {CA}")
            st.code("プロキシ: http://172.17.20.158:3128")
            
            st.stop()
        
        if not auth_success:
            st.error("❌ 認証に失敗しました")
            st.stop()
        
        # ========== 4. プロジェクト一覧取得 ==========
        st.header("4️⃣ プロジェクト一覧")
        
        if len(projects) > 0:
            st.success(f"✓ {len(projects)}件のプロジェクトにアクセス可能")
            
            project_list = []
            for proj in projects[:10]:
                project_list.append({
                    'プロジェクト名': proj.project_name,
                    'プロジェクトID': proj.id,
                    'ターゲット': proj.target,
                    '作成日': str(proj.created)[:10]
                })
            
            st.dataframe(pd.DataFrame(project_list), use_container_width=True)
        else:
            st.warning("アクセス可能なプロジェクトがありません")
            st.stop()
        
        # ========== 5. モデル取得テスト ==========
        st.header("5️⃣ モデル一覧取得")
        
        try:
            all_models = []
            
            with st.spinner(f"モデル情報を取得中... (対象: {len(projects[:5])}プロジェクト)"):
                for proj in projects[:5]:
                    try:
                        models = proj.get_models()
                        
                        for model in models[:10]:
                            all_models.append({
                                'プロジェクト名': proj.project_name,
                                'プロジェクトID': proj.id,
                                'モデルID': model.id,
                                'モデルタイプ': model.model_type,
                                'サンプル%': model.sample_pct,
                                'メトリック': getattr(model, 'metrics', {}).get(proj.metric, 'N/A') if hasattr(model, 'metrics') else 'N/A'
                            })
                    except Exception as e:
                        st.warning(f"⚠ プロジェクト '{proj.project_name}' のモデル取得失敗")
                        continue
            
            if len(all_models) > 0:
                st.success(f"✓ {len(all_models)}件のモデルを発見")
                
                models_df = pd.DataFrame(all_models)
                
                # 検索フィルター
                search_model_id = st.text_input("🔍 モデルIDで検索", value=MODEL_ID)
                
                if search_model_id:
                    filtered = models_df[models_df['モデルID'].str.contains(search_model_id, case=False)]
                    
                    if len(filtered) > 0:
                        st.success(f"✅ モデルID '{search_model_id}' が見つかりました！")
                        st.dataframe(filtered, use_container_width=True)
                        
                        st.info("このモデルは「📊 分析実行」モードで使用できます")
                    else:
                        st.warning(f"⚠ モデルID '{search_model_id}' が見つかりません")
                        st.write("利用可能なモデル一覧:")
                        st.dataframe(models_df, use_container_width=True)
                else:
                    st.dataframe(models_df, use_container_width=True)
                
                # CSVダウンロード
                csv = models_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 モデル一覧をダウンロード (CSV)",
                    data=csv,
                    file_name="datarobot_models.csv",
                    mime="text/csv"
                )
            else:
                st.warning("モデルが見つかりませんでした")
        
        except Exception as e:
            st.error("❌ モデル取得エラー")
            st.exception(e)
        
        st.success("🎉 診断完了")

# =============================================================================
# モード2: 分析実行
# =============================================================================
elif mode == "📊 分析実行":
    st.title("📊 DataRobot分析ツール")
    
    # データ読み込みボタン
    if st.button("🚀 データ読み込み", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ========== ステップ1: 接続 (20%) ==========
        status_text.info("⏳ DataRobotに接続中...")
        progress_bar.progress(20)
        
        try:
            dr.Client(endpoint=ENDPOINT, token=API_TOKEN)
            status_text.success("✓ 接続成功")
        except Exception as e:
            st.error(f"❌ 接続エラー: {e}")
            st.warning("「🔍 接続診断」モードで設定を確認してください")
            st.stop()
        
        # ========== ステップ2: モデル取得 (40%) ==========
        status_text.info(f"⏳ モデル情報取得中... (ID: {MODEL_ID})")
        progress_bar.progress(40)
        
        try:
            # 全プロジェクトからモデルを検索
            projects = dr.Project.list()
            
            model = None
            project_id = None
            
            for proj in projects:
                try:
                    models = proj.get_models()
                    for m in models:
                        if m.id == MODEL_ID:
                            model = m
                            project_id = proj.id
                            break
                    if model:
                        break
                except:
                    continue
            
            if not model:
                st.error(f"❌ モデルID '{MODEL_ID}' が見つかりません")
                st.warning("「🔍 接続診断」モードで利用可能なモデルIDを確認してください")
                st.stop()
            
            st.session_state.model = model
            st.sidebar.success(f"✓ モデル: {model.model_type}")
            st.sidebar.info(f"プロジェクトID: {project_id}")
            status_text.success("✓ モデル取得成功")
            
        except Exception as e:
            st.error(f"❌ モデル取得エラー: {e}")
            st.exception(e)
            st.stop()
        
        # ========== ステップ3: Feature Impact (60%) ==========
        status_text.info("⏳ Feature Impact取得中...")
        progress_bar.progress(60)
        
        try:
            impacts = model.get_or_request_feature_impact()
            st.session_state.feature_impact_df = pd.DataFrame(impacts).sort_values(by="impactNormalized", ascending=False)
            st.success(f"✓ Feature Impact取得完了 ({len(st.session_state.feature_impact_df)}件)")
        except Exception as e:
            st.warning(f"⚠ Feature Impact取得エラー: {e}")
            st.session_state.feature_impact_df = pd.DataFrame()
        
        # ========== ステップ4: SHAP データ (80%) ==========
        status_text.info("⏳ SHAPデータ取得中...")
        progress_bar.progress(80)
        
        # SHAP Impact
        try:
            shap_impacts_list = insights.ShapImpact.list(MODEL_ID)

            if not shap_impacts_list:
                st.info("SHAP Impactを計算中...（最大3分）")
                job = insights.ShapImpact.compute(MODEL_ID, source=INSIGHTS_SOURCES.VALIDATION, quick_compute=True)
                job.wait_for_completion(max_wait=180)
                shap_impacts_list = insights.ShapImpact.list(MODEL_ID)

            if shap_impacts_list:
                shap_impact = shap_impacts_list[0]
                shap_impact.sort('-impact_normalized')
                st.session_state.shap_impact_df = pd.DataFrame(shap_impact.shap_impacts)
                st.success(f"✓ SHAP Impact取得完了 ({len(st.session_state.shap_impact_df)}件)")
            else:
                st.session_state.shap_impact_df = pd.DataFrame()
                st.warning("⚠ SHAP Impactデータが取得できませんでした")
        except Exception as e:
            st.warning(f"⚠ SHAP Impact取得エラー: {e}")
            st.session_state.shap_impact_df = pd.DataFrame()
        
        # SHAP Distributions
        try:
            shap_dist_list = insights.ShapDistributions.list(MODEL_ID)

            if not shap_dist_list:
                st.info("SHAP Distributionsを計算中...（最大3分）")
                job = insights.ShapDistributions.compute(MODEL_ID, source=INSIGHTS_SOURCES.VALIDATION, quick_compute=True)
                job.wait_for_completion(max_wait=180)
                shap_dist_list = insights.ShapDistributions.list(MODEL_ID)

            if shap_dist_list:
                shap_dist = shap_dist_list[0]
                dist_rows = []
                for feature in shap_dist.features:
                    feature_name = feature.get('feature')
                    feature_type = feature.get('feature_type')
                    for sv in feature.get('shap_values', []):
                        dist_rows.append({
                            'feature': feature_name,
                            'feature_type': feature_type,
                            'row_index': sv.get('row_index'),
                            'prediction_value': sv.get('prediction_value'),
                            'feature_rank': sv.get('feature_rank'),
                            'feature_value': sv.get('feature_value'),
                            'shap_value': sv.get('shap_value')
                        })
                
                st.session_state.shap_distributions_df = pd.DataFrame(dist_rows)
                st.success(f"✓ SHAP Distributions取得完了 ({len(st.session_state.shap_distributions_df)}件)")
            else:
                st.session_state.shap_distributions_df = pd.DataFrame()
                st.warning("⚠ SHAP Distributionsデータが取得できませんでした")
        except Exception as e:
            st.warning(f"⚠ SHAP Distributions取得エラー: {e}")
            st.session_state.shap_distributions_df = pd.DataFrame()
        
        # ========== 完了 (100%) ==========
        progress_bar.progress(100)
        status_text.success("✓ データ取得完了")
        st.session_state.data_loaded = True
        st.rerun()
    
    # データが読み込まれている場合はグラフ表示
    if st.session_state.data_loaded:
        model = st.session_state.model
        feature_impact_df = st.session_state.feature_impact_df
        shap_impact_df = st.session_state.shap_impact_df
        shap_distributions_df = st.session_state.shap_distributions_df
        
        st.divider()
        
        # =============================================================================
        # ① 特徴量のインパクト（上位N件）
        # =============================================================================
        st.header("① 特徴量のインパクト (Permutation)")
        
        if len(feature_impact_df) > 0:
            # 最大値を実際のデータ数に制限
            max_available = len(feature_impact_df)
            
            # プロット設定（インライン）
            col1, col2 = st.columns([3, 1])
            with col1:
                num_features_impact = st.slider(
                    "表示する特徴量の数",
                    min_value=1,
                    max_value=max_available,
                    value=min(10, max_available),
                    key="impact_slider",
                    help="特徴量インパクトのグラフに表示する特徴量の数"
                )
            
            top_n_impact = feature_impact_df.nlargest(num_features_impact, 'impactNormalized').sort_values('impactNormalized')

            fig1, ax1 = plt.subplots(figsize=(10, max(6, num_features_impact * 0.4)))
            bars = ax1.barh(top_n_impact['featureName'], top_n_impact['impactNormalized'], 
                            color='steelblue', edgecolor='navy', linewidth=1.2)

            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax1.set_xlabel('Normalized Impact', fontsize=12, fontweight='bold')
            ax1.set_xlim(0, 1)
            ax1.set_title(f'Top {num_features_impact} Feature Impact (Permutation)\nModel: {model.model_type}', 
                          fontsize=14, fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            ax1.set_facecolor('#F8F9FA')
            fig1.tight_layout()
            
            st.pyplot(fig1)
            
            with st.expander("📊 データテーブルを表示"):
                st.dataframe(
                    top_n_impact[['featureName', 'impactNormalized', 'impactUnnormalized']],
                    use_container_width=True
                )
                
                # CSV ダウンロード
                csv_impact = top_n_impact.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "📥 CSVダウンロード",
                    data=csv_impact,
                    file_name=f"feature_impact_top{num_features_impact}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠ Feature Impactデータがありません")
        
        st.divider()
        
        # =============================================================================
        # ② 特徴量の作用（SHAP Impact相関プロット）
        # =============================================================================
        st.header("② 特徴量の作用 (SHAP Analysis)")
        
        # プロット表示設定
        show_shap_correlation = st.checkbox(
            "SHAP相関プロットを表示",
            value=True,
            key="show_shap_corr"
        )
        
        if len(shap_impact_df) > 0 and len(shap_distributions_df) > 0 and show_shap_correlation:
            
            # 重要度順に特徴量リストを作成
            all_shap_features = shap_impact_df.sort_values('impact_normalized', ascending=False)['feature_name'].tolist()
            
            # デフォルトはTop 10
            default_features = all_shap_features[:min(10, len(all_shap_features))]
            
            # 特徴量選択（インライン）
            col1, col2 = st.columns([4, 1])
            with col1:
                selected_features = st.multiselect(
                    "プロットする特徴量を選択（重要度順）:",
                    options=all_shap_features,
                    default=default_features,
                    help="最大20個まで選択可能です",
                    key="shap_features_select"
                )
            with col2:
                show_equation = st.checkbox(
                    "回帰式を表示",
                    value=True,
                    key="show_eq"
                )
            
            if len(selected_features) == 0:
                st.warning("⚠ 少なくとも1つの特徴量を選択してください")
            elif len(selected_features) > 20:
                st.error("❌選択できる特徴量は最大20個までです")
            else:
                # グリッドレイアウトを動的に調整
                num_selected = len(selected_features)
                
                if num_selected <= 5:
                    n_cols = num_selected
                    n_rows = 1
                elif num_selected <= 10:
                    n_cols = 5
                    n_rows = 2
                elif num_selected <= 15:
                    n_cols = 5
                    n_rows = 3
                else:
                    n_cols = 5
                    n_rows = 4

                fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
                
                # axesを1次元配列に変換
                if n_rows == 1 and n_cols == 1:
                    axes2 = [axes2]
                elif n_rows == 1 or n_cols == 1:
                    axes2 = axes2.flatten()
                else:
                    axes2 = axes2.flatten()

                for idx, feature_name in enumerate(selected_features):
                    ax = axes2[idx]
                    
                    feature_data = shap_distributions_df[shap_distributions_df['feature'] == feature_name].copy()
                    
                    if len(feature_data) > 0:
                        feature_data['feature_value_num'] = pd.to_numeric(feature_data['feature_value'], errors='coerce')
                        feature_data = feature_data.dropna(subset=['feature_value_num', 'shap_value'])
                        
                        if len(feature_data) > 10:
                            q1 = feature_data['feature_value_num'].quantile(0.05)
                            q3 = feature_data['feature_value_num'].quantile(0.95)
                            feature_data = feature_data[
                                (feature_data['feature_value_num'] >= q1) & 
                                (feature_data['feature_value_num'] <= q3)
                            ]
                            
                            # グラデーション削除 → 単色に変更
                            scatter = ax.scatter(
                                feature_data['feature_value_num'], 
                                feature_data['shap_value'],
                                color='steelblue',
                                alpha=0.6,
                                s=30,
                                edgecolors='black',
                                linewidth=0.5
                            )
                            
                            corr = feature_data['feature_value_num'].corr(feature_data['shap_value'])
                            
                            z = np.polyfit(feature_data['feature_value_num'], feature_data['shap_value'], 1)
                            p = np.poly1d(z)
                            x_line = feature_data['feature_value_num'].sort_values()
                            
                            if show_equation:
                                label_text = f'y = {z[0]:.3f}x + {z[1]:.3f}\nCorr: {corr:.3f}'
                            else:
                                label_text = f'Corr: {corr:.3f}'
                            
                            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=label_text)
                            
                            ax.set_title(f'{feature_name}', fontsize=10, fontweight='bold')
                            ax.set_xlabel('Feature Value', fontsize=9)
                            ax.set_ylabel('SHAP Value', fontsize=9)
                            ax.legend(loc='best', fontsize=8)
                            ax.grid(True, alpha=0.3)
                        else:
                            ax.text(0.5, 0.5, 'データ不足', ha='center', va='center', fontsize=12)
                            ax.set_title(f'{feature_name}', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'データなし', ha='center', va='center', fontsize=12)
                        ax.set_title(f'{feature_name}', fontsize=10)
                
                # 余分な軸を非表示
                for idx in range(num_selected, len(axes2)):
                    axes2[idx].axis('off')

                fig2.suptitle(f'Feature Impact on Prediction (SHAP Analysis)\nModel: {model.model_type}', 
                              fontsize=16, fontweight='bold', y=0.995)
                fig2.tight_layout()
                
                st.pyplot(fig2)
        
        elif not show_shap_correlation:
            st.info("ℹ️ SHAP相関プロットは非表示に設定されています")
        else:
            st.warning("⚠ SHAPデータがありません")
        
        st.divider()
        
        # =============================================================================
        # ③ SHAP分布（指標付き）
        # =============================================================================
        st.header("③ SHAP値の分布")
        
        if len(shap_distributions_df) > 0:
            feature_stats = []
            
            for feature_name in shap_distributions_df['feature'].unique():
                feature_data = shap_distributions_df[shap_distributions_df['feature'] == feature_name]
                shap_vals = feature_data['shap_value'].values
                
                stats = {
                    'feature': feature_name,
                    'mean_abs_shap': np.abs(shap_vals).mean(),
                    'std_shap': shap_vals.std(),
                    'positive_ratio': (shap_vals > 0).mean() * 100,
                    'shap_range': shap_vals.max() - shap_vals.min(),
                    'skewness': pd.Series(shap_vals).skew()
                }
                feature_stats.append(stats)
            
            stats_df = pd.DataFrame(feature_stats).sort_values('mean_abs_shap', ascending=False)
            
            # 最大値を実際のデータ数に制限
            max_available_shap = len(stats_df)
            
            # プロット設定（インライン）
            num_features_shap_dist = st.slider(
                "表示する特徴量の数",
                min_value=1,
                max_value=max_available_shap,
                value=min(10, max_available_shap),
                key="shap_dist_slider",
                help="SHAP値分布グラフに表示する特徴量の数"
            )
            
            top_n_features = stats_df.head(num_features_shap_dist)['feature'].tolist()
            top_n_data = shap_distributions_df[shap_distributions_df['feature'].isin(top_n_features)].copy()

            fig3, ax3 = plt.subplots(figsize=(14, max(10, num_features_shap_dist * 0.8)))

            for idx, feature_name in enumerate(top_n_features):
                feature_data = top_n_data[top_n_data['feature'] == feature_name]
                
                shap_values = feature_data['shap_value'].values
                feature_type = feature_data['feature_type'].iloc[0]
                
                if feature_type == 'C':
                    colors = np.array(['#AAAAAA'] * len(shap_values))
                    shap_values_to_plot = shap_values
                else:
                    try:
                        feature_values_numeric = pd.to_numeric(feature_data['feature_value'], errors='coerce')
                        valid_mask = ~feature_values_numeric.isna()
                        feature_values_numeric = feature_values_numeric[valid_mask]
                        shap_values_filtered = shap_values[valid_mask]
                        
                        if len(feature_values_numeric) > 0:
                            q1 = feature_values_numeric.quantile(0.05)
                            q3 = feature_values_numeric.quantile(0.95)
                            feature_values_clipped = feature_values_numeric.clip(q1, q3)
                            
                            if feature_values_clipped.max() != feature_values_clipped.min():
                                normalized_values = (feature_values_clipped - feature_values_clipped.min()) / \
                                                  (feature_values_clipped.max() - feature_values_clipped.min())
                            else:
                                normalized_values = np.ones(len(feature_values_clipped)) * 0.5
                            
                            cmap = plt.cm.coolwarm
                            colors = cmap(normalized_values)
                            shap_values_to_plot = shap_values_filtered
                        else:
                            colors = np.array(['#AAAAAA'] * len(shap_values))
                            shap_values_to_plot = shap_values
                    except:
                        colors = np.array(['#AAAAAA'] * len(shap_values))
                        shap_values_to_plot = shap_values
                
                if len(shap_values_to_plot) > 10:
                    try:
                        kde = gaussian_kde(shap_values_to_plot)
                        density = kde(shap_values_to_plot)
                        jitter_scale = 0.15 / (density.max() / density)
                        jitter_scale = np.clip(jitter_scale, 0.05, 0.25)
                        y_jitter = np.random.normal(0, 1, len(shap_values_to_plot)) * jitter_scale
                    except:
                        y_jitter = np.random.normal(0, 0.1, len(shap_values_to_plot))
                else:
                    y_jitter = np.random.normal(0, 0.1, len(shap_values_to_plot))
                
                y_positions = np.ones(len(shap_values_to_plot)) * idx + y_jitter
                
                ax3.scatter(
                    shap_values_to_plot,
                    y_positions,
                    c=colors,
                    alpha=0.6,
                    s=25,
                    edgecolors='none',
                    rasterized=True
                )

            ax3.axvline(x=0, color='#333333', linestyle='-', linewidth=1.5, alpha=0.7)

            y_labels = []
            for feature_name in top_n_features:
                stats = stats_df[stats_df['feature'] == feature_name].iloc[0]
                label = (f"{feature_name}\n"
                        f"Mean|SHAP|: {stats['mean_abs_shap']:.3f} "
                        f"Pos%: {stats['positive_ratio']:.1f}% "
                        f"Range: {stats['shap_range']:.3f}")
                y_labels.append(label)
            
            ax3.set_yticks(range(len(top_n_features)))
            ax3.set_yticklabels(y_labels, fontsize=9)
            ax3.set_ylim(-0.5, len(top_n_features) - 0.5)
            ax3.invert_yaxis()

            ax3.set_xlabel('SHAP value', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.2, axis='x', linestyle='--')
            ax3.set_facecolor('#FAFAFA')
            
            # タイトルをシンプルに変更
            ax3.set_title(f'SHAP分布 ({model.model_type})', 
                          fontsize=15, fontweight='bold', pad=20)

            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax3, pad=0.02, aspect=30)
            cbar.set_label('Feature value', fontsize=11, fontweight='bold')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Low', 'High'])

            fig3.tight_layout()
            st.pyplot(fig3)
            
            st.subheader(f"SHAP統計指標 (Top {num_features_shap_dist})")
            st.dataframe(
                stats_df.head(num_features_shap_dist).style.format({
                    'mean_abs_shap': '{:.4f}',
                    'std_shap': '{:.4f}',
                    'positive_ratio': '{:.2f}%',
                    'shap_range': '{:.4f}',
                    'skewness': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # CSVダウンロード
            csv_shap = stats_df.head(num_features_shap_dist).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 SHAP統計をCSVダウンロード",
                data=csv_shap,
                file_name=f"shap_statistics_top{num_features_shap_dist}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠ SHAP Distributionsデータがありません")
    
    else:
        st.info("👆 上の「🚀 データ読み込み」ボタンをクリックしてください")

        st.info("💡 モデルIDが正しいか確認するには「🔍 接続診断」モードを使用してください")
