import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'node:fs';
import path from 'node:path';

// Intercept network at fetch boundary returning [SYNTHETIC] data only
test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/**', async (route) => {
    const url = route.request().url();
    const method = route.request().method();

    if (url.includes('/auth/login') && method === 'POST') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'synthetic-jwt-token-clinician',
          token_type: 'bearer',
          role: 'clinician',
        }),
      });
    }

    if (url.includes('/health/dependencies') && method === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'HEALTHY',
          database: { status: 'UP', engine: 'PostgreSQL 16' },
          qdrant: {
            status: 'UP',
            collection: 'evidence_biobert_mnli_snli_scinli_scitail_mednli_stsb_v1',
            points_count: 85139,
          },
          llm: {
            status: 'UP',
            provider: 'gemini',
            model: 'gemini-1.5-flash',
            api_key_present: true,
          },
          ncbi: { status: 'UP', email_configured: true },
        }),
      });
    }

    if (url.includes('/analyze') && method === 'POST') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          case_id: 'c4444444-4444-4444-4444-444444444444',
          execution_id: 'e4444444-4444-4444-4444-444444444444',
          status: 'PENDING',
        }),
      });
    }

    if (
      url.includes('/executions/e4444444-4444-4444-4444-444444444444/agents') &&
      method === 'GET'
    ) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            agent_name: 'Clarifier',
            status: 'COMPLETED',
            started_at: '2026-11-14T11:00:01Z',
            completed_at: '2026-11-14T11:00:03Z',
            duration_ms: 2000,
            error_code: null,
          },
          {
            agent_name: 'RAG Analyzer',
            status: 'COMPLETED',
            started_at: '2026-11-14T11:00:03Z',
            completed_at: '2026-11-14T11:00:06Z',
            duration_ms: 3000,
            error_code: null,
          },
          {
            agent_name: 'Safety gate',
            status: 'FAILED',
            started_at: '2026-11-14T11:00:06Z',
            completed_at: '2026-11-14T11:00:07Z',
            duration_ms: 1000,
            error_code: 'VERIFICATION_FAILED',
          },
        ]),
      });
    }

    if (url.includes('/executions/e4444444-4444-4444-4444-444444444444') && method === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'e4444444-4444-4444-4444-444444444444',
          case_id: 'c4444444-4444-4444-4444-444444444444',
          status: 'FAILED',
          started_at: '2026-11-14T11:00:01Z',
          completed_at: '2026-11-14T11:00:07Z',
          error_code: 'VERIFICATION_FAILED',
          error_message: 'Safety gate verification failed',
        }),
      });
    }

    if (url.includes('/cases') && method === 'POST') {
      return route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'c4444444-4444-4444-4444-444444444444',
          status: 'SUBMITTED',
          urgent: true,
          urgent_banner: 'Acute cardiac symptoms reported',
          created_at: '2026-11-14T11:00:00Z',
        }),
      });
    }

    if (url.includes('/cases') && method === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 'c1111111-1111-1111-1111-111111111111',
            created_at: '2026-11-14T10:00:00Z',
            status: 'READY_FOR_REVIEW',
            urgent: true,
            latest_execution_status: 'COMPLETED',
          },
          {
            id: 'c2222222-2222-2222-2222-222222222222',
            created_at: '2026-11-14T09:30:00Z',
            status: 'ABSTAINED',
            urgent: false,
            latest_execution_status: 'ABSTAINED',
          },
          {
            id: 'c3333333-3333-3333-3333-333333333333',
            created_at: '2026-11-14T08:15:00Z',
            status: 'FAILED',
            urgent: false,
            latest_execution_status: 'FAILED',
          },
        ]),
      });
    }

    return route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ error: { code: 'NOT_FOUND', message: 'Not found', details: null } }),
    });
  });
});

test.describe('Clinexus Frontend Phase 6 Stage B E2E Flow', () => {
  const screenshotsDir = path.resolve(process.cwd(), 'screenshots');

  test.beforeAll(() => {
    if (!fs.existsSync(screenshotsDir)) {
      fs.mkdirSync(screenshotsDir, { recursive: true });
    }
  });

  test('smoke flow: login -> dashboard -> submit -> analysis with axe audits and screenshots', async ({
    page,
  }, testInfo) => {
    const projectName = testInfo.project.name;

    // 1. LOGIN PAGE
    await page.goto('/login');
    await expect(page.getByRole('heading', { name: 'Clinexus' })).toBeVisible();

    // Axe audit on Login Page
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const loginAxe = await new AxeBuilder({ page: page as any }).analyze();
    const loginViolations = loginAxe.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );
    expect(loginViolations).toEqual([]);

    // Capture screenshot: Login
    await page.screenshot({
      path: path.join(screenshotsDir, `${projectName}-login.png`),
      fullPage: true,
    });

    // Fill login credentials
    await page.getByLabel('Email address').fill('clinician@hospital.org');
    await page.getByLabel('Password').fill('ValidPassword123!');
    await page.getByRole('button', { name: 'Sign in' }).click();

    // 2. DASHBOARD PAGE
    await expect(page).toHaveURL('/');
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByText('PostgreSQL Database')).toBeVisible();

    // Axe audit on Dashboard Page
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const dashboardAxe = await new AxeBuilder({ page: page as any }).analyze();
    const dashboardViolations = dashboardAxe.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );
    expect(dashboardViolations).toEqual([]);

    // Capture screenshot: Dashboard
    await page.screenshot({
      path: path.join(screenshotsDir, `${projectName}-dashboard.png`),
      fullPage: true,
    });

    // 3. CASE SUBMISSION PAGE
    await page.getByRole('button', { name: 'New case submission' }).click();
    await expect(page).toHaveURL('/submit');
    await expect(page.getByRole('heading', { name: 'Clinical Case Submission' })).toBeVisible();

    // Fill clinical case text
    const caseInput = page.getByLabel(/Clinical Narrative/i);
    await caseInput.fill(
      'A 62-year-old male presents with acute retrosternal chest pain radiating to the left arm and diaphoresis.'
    );

    // Axe audit on Submission Page
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const submitAxe = await new AxeBuilder({ page: page as any }).analyze();
    const submitViolations = submitAxe.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );
    expect(submitViolations).toEqual([]);

    // Capture screenshot: Submission before submit
    await page.screenshot({
      path: path.join(screenshotsDir, `${projectName}-submit.png`),
      fullPage: true,
    });

    // Submit case
    await page.getByRole('button', { name: 'Submit case' }).click();

    // Verify triage banner appears
    await expect(page.getByRole('alert')).toContainText('Acute cardiac symptoms reported');

    // Click "Start analysis"
    await page.getByRole('button', { name: 'Start analysis' }).click();

    // 4. ANALYSIS WORKFLOW PAGE
    await expect(page).toHaveURL(
      '/cases/c4444444-4444-4444-4444-444444444444/analysis/e4444444-4444-4444-4444-444444444444'
    );
    await expect(
      page.getByRole('heading', { name: 'Multi-Agent Analysis Workflow' })
    ).toBeVisible();
    await expect(page.getByText('Clinical claim verification criteria not met')).toBeVisible();

    // Axe audit on Analysis Page
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const analysisAxe = await new AxeBuilder({ page: page as any }).analyze();
    const analysisViolations = analysisAxe.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );
    expect(analysisViolations).toEqual([]);

    // Capture screenshot: Analysis
    await page.screenshot({
      path: path.join(screenshotsDir, `${projectName}-analysis.png`),
      fullPage: true,
    });
  });
});
