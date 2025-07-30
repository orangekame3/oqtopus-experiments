## 🤖 Claude Issue Creator ヘルプ

### 📋 基本的な使い方

**Step 1: Claude分析の実行**
```
@claude このPRの変更を分析して問題があれば教えてください
```

**Step 2: 自動UI生成**
- Claudeが❌⚠️などの問題を検出すると自動的にUIが生成されます

**Step 3: Issue作成**
- チェックボックスで項目を選択
- 🚀（個別作成）または📦（まとめて作成）リアクションを追加

### 🎯 利用可能なコマンド

**直接Issue作成:**
```
/create-issue タイトル
詳細説明...

# オプション付き
/create-issue バグ修正 --labels="bug,urgent" --assign=@username
```

**Claude検出問題からの作成:**
```
/create-issues all              # 全項目を作成
/create-issues selected         # チェック済み項目を作成
/create-issues 1,3,5           # 指定番号を作成
/create-issues all --labels="priority" --assign=@reviewer
```

### ⚙️ 利用可能なオプション

- `--labels="label1,label2"` - ラベルを追加
- `--assign=@username` - ユーザーをアサイン
- `--milestone="v1.0"` - マイルストーンを設定

### 🔄 自動化フロー

1. **Claude分析** → 問題検出
2. **UI自動生成** → チェックボックス付きコメント
3. **ユーザー選択** → チェックボックス操作
4. **リアクション実行** → 🚀📦で自動Issue作成
5. **結果通知** → 作成されたIssueのリンク表示

### 📞 サポート

- `/help` または `/claude-help` - このヘルプを表示
- 問題がある場合は、ワークフローログを確認してください

---
_Claude Issue Creator v2.0 - 分離アーキテクチャ版_