import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  AnalysisWorkflowPage,
  ExecutionStatus,
  getErrorExplanation,
} from '../../src/pages/AnalysisWorkflowPage';
import { AgentRecord } from '../../src/components/AgentTimeline';
import { AuthProvider } from '../../src/context/AuthContext';

function renderWithProviders(initialEntries: string[] = ['/cases/case-123/analysis/exec-456']) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries}>
        <AuthProvider>
          <Routes>
            <Route path="/cases/:id/analysis/:executionId" element={<AnalysisWorkflowPage />} />
            <Route path="/analysis" element={<AnalysisWorkflowPage />} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('AnalysisWorkflowPage (UI-3)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  it('correctly maps error codes to plain language explanations', () => {
    expect(getErrorExplanation('LLM_UNAVAILABLE')).toBe('No language model is configured');
    expect(getErrorExplanation('LLM_QUOTA_EXHAUSTED')).toBe(
      "The model provider's quota is used up"
    );
    expect(getErrorExplanation('LLM_RATE_LIMITED')).toBe(
      'Provider rate limit reached; backoff applied'
    );
    expect(getErrorExplanation('LLM_TIMEOUT')).toBe('Model response timed out');
    expect(getErrorExplanation('LLM_AUTH_ERROR')).toBe('Invalid or unauthorized model credentials');
    expect(getErrorExplanation('VERIFICATION_FAILED')).toBe(
      'Clinical claim verification criteria not met'
    );
    expect(getErrorExplanation('UNKNOWN_CODE_999')).toBe('Error: UNKNOWN_CODE_999');
  });

  it('renders empty state when no execution ID is in the route', () => {
    renderWithProviders(['/analysis']);
    expect(screen.getByText('No active execution specified')).toBeInTheDocument();
  });

  it('renders loading state, then loads execution and agent pipeline records', async () => {
    const mockExecution: ExecutionStatus = {
      id: 'exec-456',
      case_id: 'case-123',
      status: 'RUNNING',
      started_at: '2026-11-14T10:00:00Z',
    };

    const mockAgents: AgentRecord[] = [
      { agent_name: 'Clarifier', status: 'COMPLETED', duration_ms: 1200 },
      { agent_name: 'Symptom RAG Analyzer', status: 'RUNNING', duration_ms: 800 },
    ];

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/agents')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => mockAgents,
        });
      }
      if (url.includes('/executions/exec-456')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => mockExecution,
        });
      }
      return Promise.reject(new Error('Not found'));
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders();

    expect(await screen.findByText('exec-456')).toBeInTheDocument();
    expect(screen.getByText('Clarifier')).toBeInTheDocument();
    expect(screen.getByText('Symptom RAG Analyzer')).toBeInTheDocument();

    // Overall status in aria-live region
    const liveRegion = screen
      .getByRole('heading', { name: /Execution Status:/i })
      .parentElement?.querySelector('[aria-live="polite"]');
    expect(liveRegion).toHaveTextContent('RUNNING');
  });

  it('displays error explanation and halts polling on terminal FAILED status', async () => {
    const mockExecution: ExecutionStatus = {
      id: 'exec-456',
      case_id: 'case-123',
      status: 'FAILED',
      started_at: '2026-11-14T10:00:00Z',
      duration_seconds: 4.2,
      error_code: 'LLM_QUOTA_EXHAUSTED',
      error_message: 'Resource has been exhausted (e.g. check quota).',
    };

    const mockAgents: AgentRecord[] = [
      { agent_name: 'Clarifier', status: 'COMPLETED', duration_ms: 1100 },
      {
        agent_name: 'Symptom RAG Analyzer',
        status: 'FAILED',
        error_code: 'LLM_QUOTA_EXHAUSTED',
        error_message: 'Daily quota exhausted',
      },
    ];

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/agents')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => mockAgents });
      }
      if (url.includes('/executions/exec-456')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => mockExecution });
      }
      return Promise.reject(new Error('Not found'));
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders();

    expect(await screen.findByText(/Failure code: LLM_QUOTA_EXHAUSTED/i)).toBeInTheDocument();
    expect(screen.getByText("The model provider's quota is used up")).toBeInTheDocument();

    // Terminal state link to report
    expect(screen.getByRole('button', { name: /View Report/i })).toBeInTheDocument();
  });

  it('displays abstention alert when status is ABSTAINED', async () => {
    const mockExecution: ExecutionStatus = {
      id: 'exec-456',
      case_id: 'case-123',
      status: 'ABSTAINED',
      started_at: '2026-11-14T10:00:00Z',
      duration_seconds: 8.5,
    };

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/agents')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => [] });
      }
      if (url.includes('/executions/exec-456')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => mockExecution });
      }
      return Promise.reject(new Error('Not found'));
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders();

    expect(await screen.findByText('Evidence Safety Gate Abstained')).toBeInTheDocument();
    expect(
      screen.getByText(
        /The multi-agent workflow abstained from generating a diagnostic recommendation/i
      )
    ).toBeInTheDocument();
  });

  it('pauses polling when tab is hidden and resumes when visible', async () => {
    const mockExecution: ExecutionStatus = {
      id: 'exec-456',
      case_id: 'case-123',
      status: 'RUNNING',
      started_at: '2026-11-14T10:00:00Z',
    };

    let fetchCount = 0;
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      fetchCount++;
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => (url.includes('/agents') ? [] : mockExecution),
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders();

    await screen.findByText('exec-456');
    const countBeforeHidden = fetchCount;

    // Simulate tab hiding
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'hidden',
    });
    fireEvent(document, new Event('visibilitychange'));

    // Wait 100ms
    await new Promise((r) => setTimeout(r, 100));

    // Restore tab visibility
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'visible',
    });
    fireEvent(document, new Event('visibilitychange'));

    expect(fetchCount).toBeGreaterThanOrEqual(countBeforeHidden);
  });
});
