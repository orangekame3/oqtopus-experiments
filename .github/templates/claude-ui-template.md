## 🤖 Claude問題検出 → Issue作成ツール

**{{ISSUE_COUNT}}件の問題が検出されました** 

作成したい項目を **チェック** してください：

{{ISSUE_LIST}}

---

### 🎯 実行方法

**方法1: リアクション実行** (推奨)
1. 上記のチェックボックスを選択
2. このコメントに以下のリアクションを追加：
   - 🚀 = 選択した項目を **個別にissue作成**
   - 📦 = 選択した項目を **1つのissueにまとめて作成**

**方法2: コマンド実行**
```
/create-issues selected    # チェックした項目を作成
/create-issues all         # 全項目を作成  
/create-issues 1,3,5       # 番号指定で作成
```

### ⚙️ オプション
```
/create-issues all --labels="bug,urgent" --assign=@username
```

<details>
<summary>📋 検出された問題の詳細</summary>

{{ISSUE_DETAILS}}

</details>

---
_検出ID: `{{COMMENT_ID}}` | 検出時刻: {{TIMESTAMP}}_