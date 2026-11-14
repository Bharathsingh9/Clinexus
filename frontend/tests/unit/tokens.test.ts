import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('Design System Tokens (A15 / tokens.css)', () => {
  const tokensPath = path.resolve(__dirname, '../../src/styles/tokens.css');
  const tokensContent = fs.readFileSync(tokensPath, 'utf-8');

  it('defines the 4px spacing scale', () => {
    expect(tokensContent).toContain('--space-1: 4px');
    expect(tokensContent).toContain('--space-2: 8px');
    expect(tokensContent).toContain('--space-3: 12px');
    expect(tokensContent).toContain('--space-4: 16px');
    expect(tokensContent).toContain('--space-8: 32px');
  });

  it('defines semantic colors matching SPEC 7 and A15', () => {
    expect(tokensContent).toContain('--color-status-completed');
    expect(tokensContent).toContain('--color-status-partial');
    expect(tokensContent).toContain('--color-status-failed');
    expect(tokensContent).toContain('--color-status-abstained');
    expect(tokensContent).toContain('--color-status-human');
  });

  it('strictly contains no green colors for clinical validity', () => {
    // Strip comments to inspect actual CSS rules and color declarations
    const cssWithoutComments = tokensContent.replace(/\/\*[\s\S]*?\*\//g, '');
    expect(cssWithoutComments).not.toMatch(/\bgreen\b/i);
    expect(cssWithoutComments).not.toMatch(/#00ff00|#008000|#22c55e|#16a34a|#15803d/i);
  });

  it('defines dark theme via prefers-color-scheme', () => {
    expect(tokensContent).toContain('@media (prefers-color-scheme: dark)');
  });

  it('respects prefers-reduced-motion', () => {
    expect(tokensContent).toContain('@media (prefers-reduced-motion: reduce)');
  });
});
