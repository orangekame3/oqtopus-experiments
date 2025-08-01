# OQTOPUS Experiments - Implementation TODO

## 現在実装済みの実験

- [x] **Rabi** - Rabi振動測定（量子ビットキャリブレーション）
- [x] **T1** - 緩和時間測定
- [x] **T2 Echo** - エコーシーケンスによるコヒーレンス時間測定
- [x] **CHSH** - ベル不等式テストによる量子もつれ検証
- [x] **CHSH Phase Scan** - CHSH実験の位相スキャン版
- [x] **Ramsey** - 高精度周波数測定のためのRamsey干渉
- [x] **Parity Oscillation** - GHZ状態デコヒーレンス研究
- [x] **Hadamard Test** - Hadamardテスト実験
- [x] **Randomized Benchmarking** - Cliffordゲートの平均エラー率測定
- [x] **Interleaved Randomized Benchmarking** - 特定ゲートのエラー率分離評価
- [x] **Deutsch-Jozsa Algorithm** - 小規模アルゴリズムベンチマーク
- [x] **Grover Algorithm** - 量子探索アルゴリズム
- [x] **Bernstein-Vazirani Algorithm** - 隠れビット列の量子アルゴリズム

## 追加実装予定の実験

### 高優先度：キャリブレーション・ベンチマーク基盤

#### 1. ~~Randomized Benchmarking (RB)~~ ✅ **完了**
**実装優先度**: 最高 → **実装済み**
- **目的**: Cliffordゲートの平均エラー率測定
- **特徴**: SPAM誤差に対して堅牢
- **用途**: 量子ビットキャラクタリゼーション、ゲート性能評価
- **実装完了**: 
  - ✅ Standard RB（基本版）
  - ✅ Interleaved RB（特定ゲートの分離評価）
  - ✅ 完全24要素Clifford群実装
  - ✅ ゲートフィデリティ計算
  - ✅ 統合プロット機能
  - ✅ データマネージャー統合

#### 2. Quantum State Tomography (QST)
**実装優先度**: 高 → **次の優先実装候補**
- **目的**: 任意の量子状態の密度行列を完全再構成
- **用途**: GHZ、W状態やVQE状態の検証
- **実装メモ**: 
  - 1-3量子ビット対応から開始
  - パリティ振動実験との連携

#### 3. SPAM Calibration
**実装優先度**: 高
- **目的**: 測定誤差の補正行列を構築
- **用途**: 生データを物理確率に変換
- **実装メモ**: 
  - 混同行列の構築
  - 線形逆変換・ベイズ補正ベクトル

### 中優先度：高度なキャラクタリゼーション

#### 4. Quantum Process Tomography (QPT)
- **目的**: 単一・二量子ビットゲートの完全なプロセス行列再構成
- **用途**: コヒーレント・非コヒーレント誤差の完全情報取得

#### 5. Dynamical Decoupling (CPMG/XY-4)
- **目的**: パルス列による1/fおよびテレグラフノイズの抑制
- **用途**: ノイズスペクトル測定、T2 Echoの補完実験

#### 6. Crosstalk Characterization
- **目的**: 同時RBによる量子ビット間のクロストーク定量化
- **用途**: 古典・量子クロストークの評価

### 中優先度：アルゴリズム・プロトコル検証

#### 7. Quantum Teleportation
- **目的**: 3量子ビットテレポーテーションプロトコル実装
- **用途**: もつれ分散、ベル状態測定、フィードフォワードの総合テスト

#### 8. Quantum Fourier Transform (QFT) Verification
- **目的**: QFTとその逆変換の実行検証
- **用途**: 多数の制御位相ゲートの累積キャリブレーション誤差検出


### 低優先度：特殊用途・研究向け

#### 9. Quantum Volume
- **目的**: IBM標準のランダム回路性能指標
- **用途**: n量子ビット、n深度回路の実行能力評価

#### 10. Cross-Entropy Benchmarking (XEB)
- **目的**: 疑似ランダム回路の出力確率分布比較
- **用途**: Clifford空間を超えた多量子ビット忠実度測定

#### 11. Mermin-GHZ Test
- **目的**: 3量子ビット以上のベル不等式拡張
- **用途**: 多体量子もつれと非局所性の検証

#### 12. Gate-Set Tomography (GST)
- **目的**: 状態準備・測定・ゲートセットの自己整合的キャラクタリゼーション
- **用途**: QPTの拡張、完璧でない仮定なしの評価

#### 13. Leakage & Seepage Measurement
- **目的**: 計算部分空間からの漏れ（leakage）と戻り（seepage）の定量化
- **用途**: 超伝導・イオントラップ量子ビット向け

#### 14. Swap Test / Overlap Estimation
- **目的**: 補助量子ビットによる状態重複度評価
- **用途**: VQE回路検証、機械学習カーネル

## 実装戦略

### Phase 1: 基盤整備（高優先度）
1. ✅ **Randomized Benchmarking** - 完了（2025-07-26）
2. **SPAM Calibration** - 測定精度向上の基盤（次の候補）
3. **Quantum State Tomography** - 状態検証機能（次の候補）

### Phase 2: 特殊化（中優先度）
1. アルゴリズムベンチマーク群
2. 高度なキャラクタリゼーション

### Phase 3: 研究用途（低優先度）
1. 専門的測定・評価手法
2. エラー訂正準備実験

## 技術要件

- すべての実験は既存のバックエンド（OQTOPUS、Qulacs、Qiskit Aer）で動作
- 汎用ゲート操作と量子ビット読み出しのみ使用
- 並列実行、自動データ解析、可視化をサポート
- 既存の実験との一貫性を保持

## 次のアクション

1. ✅ ~~**Randomized Benchmarking**の実装から開始~~ **完了**
2. ✅ ~~既存のRabi/T1/T2実験との統合テスト~~ **完了**
3. **Quantum State Tomography (QST)** または **SPAM Calibration** の実装開始
4. 段階的に他の実験を追加

---
*最終更新: 2025-08-01*  
*RB実装完了により更新*  
*o3との相談結果に基づく提案*