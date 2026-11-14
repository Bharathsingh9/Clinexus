import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

const SRC_DIR = path.resolve(__dirname, '../../src');

// Forbidden promotional / validation terms per SPEC 7, Appendix A15, Rule R11, and Step 4.5
// Note: "not approved" in the mandatory privacy notice is permitted, but positive claims of approval are strictly forbidden.
const FORBIDDEN_TERMS = [
  { pattern: /(?<!not\s+)\bapproved\b/i, label: 'approved' },
  { pattern: /\bvalidated\b/i, label: 'validated' },
  { pattern: /\bcorrect\b/i, label: 'correct' },
  { pattern: /\bsafe\b/i, label: 'safe' },
  { pattern: /\baccurate\b/i, label: 'accurate' },
  { pattern: /\bdiagnosis\s+confirmed\b/i, label: 'diagnosis confirmed' },
];

function getSourceFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...getSourceFiles(fullPath));
    } else if (
      (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx')) &&
      !entry.name.endsWith('.d.ts')
    ) {
      files.push(fullPath);
    }
  }
  return files;
}

describe('Copy Rules Enforcement (Step 4.5 / A15 / R11)', () => {
  const files = getSourceFiles(SRC_DIR);

  it('scans all frontend source files and finds no forbidden clinical claim terms', () => {
    const violations: { file: string; line: number; term: string; content: string }[] = [];

    for (const filePath of files) {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');

      lines.forEach((lineText, idx) => {
        // Strip code comments if any, but scan all string literals and JSX text
        for (const { pattern, label } of FORBIDDEN_TERMS) {
          if (pattern.test(lineText)) {
            violations.push({
              file: path.relative(SRC_DIR, filePath),
              line: idx + 1,
              term: label,
              content: lineText.trim(),
            });
          }
        }
      });
    }

    if (violations.length > 0) {
      const details = violations
        .map((v) => `  ${v.file}:${v.line} matched "${v.term}": "${v.content}"`)
        .join('\n');
      expect.fail(
        `Found ${violations.length} forbidden copy violations in frontend/src:\n${details}`
      );
    }

    expect(violations).toHaveLength(0);
  });
});
