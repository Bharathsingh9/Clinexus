import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CaseSubmissionPage } from '../../src/pages/CaseSubmissionPage';
import { AuthProvider } from '../../src/context/AuthContext';
import { SCHEMA_MAX_CHARS } from '../../src/api/schema_constants';

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>{ui}</AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('CaseSubmissionPage (UI-2)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  it('renders free-text narrative field, schema max character counter, and privacy notice', () => {
    renderWithProviders(<CaseSubmissionPage />);

    // Privacy notice
    expect(
      screen.getByText(
        'Do not enter real patient identifiers. This research system is not approved for real patient data.'
      )
    ).toBeInTheDocument();

    // Narrative field
    const textarea = screen.getByLabelText(/Clinical Narrative/i);
    expect(textarea).toBeInTheDocument();

    // Character counter displays schema maximum
    expect(screen.getByText(new RegExp(`0 / ${SCHEMA_MAX_CHARS} characters`))).toBeInTheDocument();
  });

  it('computes missing-information hints strictly from empty structured fields', () => {
    renderWithProviders(<CaseSubmissionPage />);

    expect(
      screen.getByText(
        /Structured context fields currently unprovided: Age, Biological sex, Symptom duration/i
      )
    ).toBeInTheDocument();

    // Fill in age
    fireEvent.change(screen.getByLabelText(/Patient Age/i), { target: { value: '45' } });
    expect(
      screen.getByText(
        /Structured context fields currently unprovided: Biological sex, Symptom duration/i
      )
    ).toBeInTheDocument();

    // Fill in sex and duration
    fireEvent.change(screen.getByLabelText(/Biological Sex/i), { target: { value: 'Female' } });
    fireEvent.change(screen.getByLabelText(/Symptom Duration/i), { target: { value: '3 days' } });

    // When all structured fields filled, hints box disappears
    expect(
      screen.queryByText(/Structured context fields currently unprovided/i)
    ).not.toBeInTheDocument();
  });

  it('enforces client-side validation for empty and over-length narrative', async () => {
    renderWithProviders(<CaseSubmissionPage />);

    const submitBtn = screen.getByRole('button', { name: 'Submit case' });
    fireEvent.click(submitBtn);

    expect(await screen.findByText('Case narrative is required.')).toBeInTheDocument();

    const textarea = screen.getByLabelText(/Clinical Narrative/i);
    // Overlength test
    fireEvent.change(textarea, { target: { value: 'x'.repeat(SCHEMA_MAX_CHARS + 5) } });
    fireEvent.click(submitBtn);

    expect(
      await screen.findByText(
        new RegExp(`Case text exceeds maximum allowed length of ${SCHEMA_MAX_CHARS}`)
      )
    ).toBeInTheDocument();
  });

  it('submits case narrative, disables double submit, displays returned urgent banner, and passes fresh Idempotency-Key to analyze', async () => {
    const mockCaseResponse = {
      id: 'case-test-1234-5678',
      status: 'DRAFT',
      urgent: true,
      urgent_banner:
        'Possible urgent presentation: do not delay appropriate urgent human assessment',
      created_at: '2026-11-14T10:00:00Z',
    };

    const mockAnalyzeResponse = {
      execution_id: 'exec-test-9999',
      status: 'QUEUED',
      case_id: 'case-test-1234-5678',
    };

    let postCasesCalls = 0;
    let postAnalyzeCalls = 0;
    let analyzeIdempotencyKey: string | null = null;

    const fetchMock = vi.fn().mockImplementation((url: string, init: RequestInit) => {
      if (url.includes('/cases') && !url.includes('/analyze')) {
        postCasesCalls++;
        return Promise.resolve({
          ok: true,
          status: 201,
          json: async () => mockCaseResponse,
        });
      }
      if (url.includes('/analyze')) {
        postAnalyzeCalls++;
        analyzeIdempotencyKey = (init.headers as Record<string, string>)['Idempotency-Key'];
        return Promise.resolve({
          ok: true,
          status: 202,
          json: async () => mockAnalyzeResponse,
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders(<CaseSubmissionPage />);

    const textarea = screen.getByLabelText(/Clinical Narrative/i);
    fireEvent.change(textarea, {
      target: { value: 'Patient presents with acute severe chest pain and diaphoresis.' },
    });

    const submitBtn = screen.getByRole('button', { name: 'Submit case' });
    fireEvent.click(submitBtn);

    // Double submit prevented
    expect(submitBtn).toBeDisabled();

    // Success view rendered
    expect(await screen.findByText('Case registered successfully')).toBeInTheDocument();
    expect(screen.getByText('case-test-1234-5678')).toBeInTheDocument();

    // Urgent banner shown prominently with exact returned text
    expect(
      screen.getByText(
        'Possible urgent presentation: do not delay appropriate urgent human assessment'
      )
    ).toBeInTheDocument();

    expect(postCasesCalls).toBe(1);

    // Start analysis
    const startAnalysisBtn = screen.getByRole('button', { name: /Start analysis/i });
    fireEvent.click(startAnalysisBtn);

    await waitFor(() => {
      expect(postAnalyzeCalls).toBe(1);
      expect(analyzeIdempotencyKey).toBeTruthy();
      // Ensure Idempotency-Key is a valid UUID format
      expect(analyzeIdempotencyKey).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
    });
  });

  it('displays 422 server validation error without echoing user patient text', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Validation failed',
          details: [{ loc: ['body', 'raw_text'], msg: 'String should have at least 1 character' }],
        },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders(<CaseSubmissionPage />);

    const textarea = screen.getByLabelText(/Clinical Narrative/i);
    const sensitivePatientText = 'Sensitive confidential clinical note 999';
    fireEvent.change(textarea, { target: { value: sensitivePatientText } });

    fireEvent.click(screen.getByRole('button', { name: 'Submit case' }));

    const errorBanner = await screen.findByRole('alert');
    expect(errorBanner).toHaveTextContent('String should have at least 1 character');
    // Ensure the sensitive text is NOT echoed in error output
    expect(errorBanner.textContent).not.toContain(sensitivePatientText);
  });
});
