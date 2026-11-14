import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DashboardPage, CaseSummary, DependenciesResponse } from '../../src/pages/DashboardPage';
import { AuthProvider } from '../../src/context/AuthContext';

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>{ui}</AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('DashboardPage (UI-1)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
  });

  it('renders loading skeleton while fetching data', () => {
    const fetchMock = vi.fn().mockImplementation(() => new Promise(() => {})); // Never resolves
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders(<DashboardPage />);

    expect(screen.getByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
    expect(screen.getAllByLabelText('Loading content').length).toBeGreaterThan(0);
  });

  it('renders EmptyState when no cases exist, with navigation button to /submit', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/cases')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => [] as CaseSummary[],
        });
      }
      if (url.includes('/health/dependencies')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () =>
            ({
              status: 'UP',
              database: { status: 'UP', engine: 'PostgreSQL' },
              qdrant: { status: 'UP', points_count: 85139 },
              llm: {
                status: 'UP',
                provider: 'gemini',
                model: 'gemini-2.0-flash',
                api_key_present: true,
              },
              ncbi: { status: 'UP', email_configured: true },
            }) as DependenciesResponse,
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText('No cases submitted yet')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Submit first case' })).toBeInTheDocument();
  });

  it('renders ErrorState with retry button when cases request fails', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/cases')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
          json: async () => ({ error: { code: 'SERVER_ERROR', message: 'DB connection failure' } }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ status: 'DOWN' }),
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText('Failed to load cases')).toBeInTheDocument();
    expect(screen.getByText('DB connection failure')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('renders populated cases table, counters, panels, and dependencies; strictly omits case narrative text', async () => {
    const mockCases: CaseSummary[] = [
      {
        id: 'c1111111-2222-3333-4444-555555555555',
        status: 'READY_FOR_REVIEW',
        created_at: '2026-11-14T08:30:00Z',
        urgent: true,
        latest_execution_status: 'COMPLETED',
      },
      {
        id: 'c2222222-3333-4444-5555-666666666666',
        status: 'ABSTAINED',
        created_at: '2026-11-14T09:00:00Z',
        urgent: false,
        latest_execution_status: 'ABSTAINED',
      },
      {
        id: 'c3333333-4444-5555-6666-777777777777',
        status: 'FAILED',
        created_at: '2026-11-14T09:15:00Z',
        urgent: false,
        latest_execution_status: 'FAILED',
      },
    ];

    const mockDeps: DependenciesResponse = {
      status: 'UP',
      database: { status: 'UP', engine: 'PostgreSQL' },
      qdrant: { status: 'UP', points_count: 85139 },
      llm: {
        status: 'UP',
        provider: 'gemini',
        model: 'gemini-2.0-flash',
        api_key_present: true,
      },
      ncbi: { status: 'UP', email_configured: true },
    };

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/cases')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => mockCases,
        });
      }
      if (url.includes('/health/dependencies')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => mockDeps,
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
    vi.stubGlobal('fetch', fetchMock);

    const { container } = renderWithProviders(<DashboardPage />);

    // Table and Needs Review panel render case short IDs
    const c1Elements = await screen.findAllByText('c1111111...');
    expect(c1Elements.length).toBeGreaterThanOrEqual(1);

    const c2Elements = await screen.findAllByText('c2222222...');
    expect(c2Elements.length).toBeGreaterThanOrEqual(1);

    const c3Elements = await screen.findAllByText('c3333333...');
    expect(c3Elements.length).toBeGreaterThanOrEqual(1);

    // Urgent marker
    expect(screen.getByText('Urgent')).toBeInTheDocument();

    // Counters computed ONLY from loaded list
    expect(screen.getByText('Total Cases')).toBeInTheDocument();
    expect(screen.getAllByText('(n = 3)')).toHaveLength(3);

    // Panels
    expect(screen.getByRole('heading', { name: 'Needs Review' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Abstained & Failed Cases' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'System Dependencies' })).toBeInTheDocument();

    // Dependencies details
    expect(screen.getByText('85139')).toBeInTheDocument();
    expect(screen.getByText(/API key present: yes/i)).toBeInTheDocument();
    expect(screen.getByText(/Provider: gemini/i)).toBeInTheDocument();

    // STRICT INVARIANT CHECK: NO patient or case narrative text anywhere
    const patientTextTerms = [
      'patient presents',
      'chest pain',
      'dyspnea',
      'shortness of breath',
      'myocardial infarction',
      'troponin',
      'hypertension',
    ];
    const pageHtml = container.innerHTML.toLowerCase();
    for (const term of patientTextTerms) {
      expect(pageHtml).not.toContain(term);
    }
  });
});
