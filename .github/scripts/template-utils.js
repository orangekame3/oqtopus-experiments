/**
 * Template utilities for GitHub Actions workflows
 * Provides functions to load and process markdown templates
 */

/**
 * Load and process a template file with variable substitution
 * @param {string} templatePath - Path to the template file
 * @param {object} variables - Object containing template variables
 * @returns {string} Processed template content
 */
async function loadTemplate(templatePath, variables = {}) {
  try {
    // In GitHub Actions, we'll read the template content directly
    // This is a placeholder for the actual template loading logic
    const fs = require('fs');
    const path = require('path');
    
    const fullPath = path.join(process.cwd(), templatePath);
    let content = fs.readFileSync(fullPath, 'utf8');
    
    // Replace template variables
    Object.entries(variables).forEach(([key, value]) => {
      const regex = new RegExp(`{{${key}}}`, 'g');
      content = content.replace(regex, value || '');
    });
    
    return content;
  } catch (error) {
    console.error(`Failed to load template ${templatePath}:`, error);
    return '';
  }
}

/**
 * Generate issue list for Claude UI template
 * @param {Array} issues - Array of issue objects
 * @returns {string} Formatted issue list
 */
function generateIssueList(issues) {
  return issues.map((issue, i) => 
    `- [ ] ${issue.emoji} **[${issue.priority.toUpperCase()}]** ${issue.content}`
  ).join('\n');
}

/**
 * Generate issue details for Claude UI template
 * @param {Array} issues - Array of issue objects
 * @returns {string} Formatted issue details
 */
function generateIssueDetails(issues) {
  return issues.map((issue, i) => `**${i + 1}. ${issue.content}**
- タイプ: ${issue.type}
- 優先度: ${issue.priority}
${issue.sectionContent ? `- 詳細: ${issue.sectionContent.substring(0, 200)}...` : ''}
`).join('\n');
}

/**
 * Generate bundled issues list for bundled issue template
 * @param {Array} items - Array of selected items
 * @returns {string} Formatted bundled issues list
 */
function generateBundledIssuesList(items) {
  return items.map((item, i) => `### ${i + 1}. ${item.content}

**対応状況**: 🔲 未対応

---`).join('\n\n');
}

// Export functions for use in GitHub Actions
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    loadTemplate,
    generateIssueList,
    generateIssueDetails,
    generateBundledIssuesList
  };
}

// For GitHub Actions inline script usage
global.templateUtils = {
  loadTemplate,
  generateIssueList,
  generateIssueDetails,
  generateBundledIssuesList
};