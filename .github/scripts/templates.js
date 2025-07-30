/**
 * Template definitions for GitHub Actions workflows
 * All markdown templates are defined as JavaScript template literals
 */

const TEMPLATES = {
  CLAUDE_UI: `## 🤖 Claude問題検出 → Issue作成ツール

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
\`\`\`
/create-issues selected    # チェックした項目を作成
/create-issues all         # 全項目を作成  
/create-issues 1,3,5       # 番号指定で作成
\`\`\`

### ⚙️ オプション
\`\`\`
/create-issues all --labels="bug,urgent" --assign=@username
\`\`\`

<details>
<summary>📋 検出された問題の詳細</summary>

{{ISSUE_DETAILS}}

</details>

---
_検出ID: \`{{COMMENT_ID}}\` | 検出時刻: {{TIMESTAMP}}_`,

  INDIVIDUAL_ISSUE: `## 概要
{{CONTENT}}

## コンテキスト
- **検出元**: PR #{{PR_NUMBER}}
- **検出者**: Claude Code Analysis
- **作成者**: @{{CREATOR}}
- **作成日時**: {{TIMESTAMP}}
- **元コメント**: {{COMMENT_URL}}

## 対応方針
TODO: 対応方針を記載してください

## 関連
- [ ] 問題の詳細調査
- [ ] 修正方針の検討
- [ ] 実装とテスト
- [ ] レビューと統合

---
_このissueはClaude分析結果から自動生成されました_`,

  BUNDLED_ISSUE: `## 検出された問題一覧

{{ISSUES_LIST}}

## 全体的な対応方針

- [ ] 各問題の詳細調査
- [ ] 優先度の決定
- [ ] 修正計画の策定
- [ ] 段階的な実装

## コンテキスト
- **検出元**: PR #{{PR_NUMBER}}
- **検出者**: Claude Code Analysis  
- **作成者**: @{{CREATOR}}
- **作成日時**: {{TIMESTAMP}}
- **総問題数**: {{ISSUE_COUNT}}件

## 進捗管理

各問題の解決後は、上記のチェックボックスを更新してください。

---
_このissueはClaude分析結果をまとめて自動生成されました_`,

  MANUAL_ISSUE: `## 概要
{{DESCRIPTION}}

## コンテキスト
- **作成元**: PR #{{PR_NUMBER}} - https://github.com/{{REPO_OWNER}}/{{REPO_NAME}}/pull/{{PR_NUMBER}}
- **作成者**: @{{CREATOR}}
- **作成日時**: {{TIMESTAMP}}
- **作成方法**: 手動コマンド

## 詳細
{{DETAILED_DESCRIPTION}}

---
_このissueは手動コマンドから作成されました_`,

  HELP: `## 🤖 Claude Issue Creator ヘルプ

### 📋 基本的な使い方

**Step 1: Claude分析の実行**
\`\`\`
@claude このPRの変更を分析して問題があれば教えてください
\`\`\`

**Step 2: 自動UI生成**
- Claudeが❌⚠️などの問題を検出すると自動的にUIが生成されます

**Step 3: Issue作成**
- チェックボックスで項目を選択
- 🚀（個別作成）または📦（まとめて作成）リアクションを追加

### 🎯 利用可能なコマンド

**直接Issue作成:**
\`\`\`
/create-issue タイトル
詳細説明...

# オプション付き
/create-issue バグ修正 --labels="bug,urgent" --assign=@username
\`\`\`

**Claude検出問題からの作成:**
\`\`\`
/create-issues all              # 全項目を作成
/create-issues selected         # チェック済み項目を作成
/create-issues 1,3,5           # 指定番号を作成
/create-issues all --labels="priority" --assign=@reviewer
\`\`\`

### ⚙️ 利用可能なオプション

- \`--labels="label1,label2"\` - ラベルを追加
- \`--assign=@username\` - ユーザーをアサイン
- \`--milestone="v1.0"\` - マイルストーンを設定

### 🔄 自動化フロー

1. **Claude分析** → 問題検出
2. **UI自動生成** → チェックボックス付きコメント
3. **ユーザー選択** → チェックボックス操作
4. **リアクション実行** → 🚀📦で自動Issue作成
5. **結果通知** → 作成されたIssueのリンク表示

### 📞 サポート

- \`/help\` または \`/claude-help\` - このヘルプを表示
- 問題がある場合は、ワークフローログを確認してください

---
_Claude Issue Creator v2.0 - 分離アーキテクチャ版_`
};

/**
 * Process template with variable substitution
 * @param {string} templateName - Name of the template to use
 * @param {object} variables - Object containing template variables
 * @returns {string} Processed template content
 */
function processTemplate(templateName, variables = {}) {
  let template = TEMPLATES[templateName];
  if (!template) {
    console.error(\`Template '\${templateName}' not found\`);
    return '';
  }
  
  // Replace template variables
  Object.entries(variables).forEach(([key, value]) => {
    const regex = new RegExp(\`{{\${key}}}\`, 'g');
    template = template.replace(regex, value || '');
  });
  
  return template;
}

/**
 * Generate issue list for Claude UI template
 * @param {Array} issues - Array of issue objects
 * @returns {string} Formatted issue list
 */
function generateIssueList(issues) {
  return issues.map((issue, i) => 
    \`- [ ] \${issue.emoji} **[\${issue.priority.toUpperCase()}]** \${issue.content}\`
  ).join('\\n');
}

/**
 * Generate issue details for Claude UI template
 * @param {Array} issues - Array of issue objects
 * @returns {string} Formatted issue details
 */
function generateIssueDetails(issues) {
  return issues.map((issue, i) => \`**\${i + 1}. \${issue.content}**
- タイプ: \${issue.type}
- 優先度: \${issue.priority}
\${issue.sectionContent ? \`- 詳細: \${issue.sectionContent.substring(0, 200)}...\` : ''}
\`).join('\\n');
}

/**
 * Generate bundled issues list for bundled issue template
 * @param {Array} items - Array of selected items
 * @returns {string} Formatted bundled issues list
 */
function generateBundledIssuesList(items) {
  return items.map((item, i) => \`### \${i + 1}. \${item.content}

**対応状況**: 🔲 未対応

---\`).join('\\n\\n');
}

// Make functions available globally for GitHub Actions
if (typeof global !== 'undefined') {
  global.TEMPLATES = TEMPLATES;
  global.processTemplate = processTemplate;
  global.generateIssueList = generateIssueList;
  global.generateIssueDetails = generateIssueDetails;
  global.generateBundledIssuesList = generateBundledIssuesList;
}