import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { PlusCircle, AlertOctagon, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { Card } from '../components/Card';
import { DataTable, Column } from '../components/DataTable';
import { StatusBadge } from '../components/StatusBadge';
import { MetricTile } from '../components/MetricTile';
import { Button } from '../components/Button';
import { Skeleton } from '../components/Skeleton';
import { EmptyState } from '../components/EmptyState';
import { ErrorState } from '../components/ErrorState';
import styles from './DashboardPage.module.css';

export interface CaseSummary {
  id: string;
  status: string;
  created_at: string;
  urgent: boolean;
  latest_execution_status?: string | null;
}

export interface DependenciesResponse {
  status: string;
  database: { status: string; engine?: string };
  qdrant: { status: string; collection?: string; points_count?: number };
  llm: { status: string; provider?: string; model?: string; api_key_present?: boolean };
  ncbi: { status: string; email_configured?: boolean };
}

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  const {
    data: cases,
    isLoading: isCasesLoading,
    isError: isCasesError,
    error: casesError,
    refetch: refetchCases,
  } = useQuery<CaseSummary[]>({
    queryKey: ['cases'],
    queryFn: () => apiClient<CaseSummary[]>('/cases'),
  });

  const {
    data: dependencies,
    isLoading: isDepsLoading,
    isError: isDepsError,
    error: depsError,
    refetch: refetchDeps,
  } = useQuery<DependenciesResponse>({
    queryKey: ['dependencies'],
    queryFn: () => apiClient<DependenciesResponse>('/health/dependencies'),
  });

  if (isCasesLoading || isDepsLoading) {
    return (
      <div className={styles.page}>
        <div className={styles.header}>
          <h1>Dashboard</h1>
        </div>
        <div className={styles.metricsGrid}>
          <Skeleton height="100px" count={3} />
        </div>
        <Card title="Recent Cases">
          <Skeleton height="200px" count={1} />
        </Card>
      </div>
    );
  }

  if (isCasesError) {
    return (
      <div className={styles.page}>
        <div className={styles.header}>
          <h1>Dashboard</h1>
        </div>
        <ErrorState
          title="Failed to load cases"
          message={casesError instanceof Error ? casesError.message : 'Unable to reach API'}
          onRetry={() => refetchCases()}
        />
      </div>
    );
  }

  const loadedCases = cases || [];

  // Compute counters ONLY from the loaded cases list (never invent statistics)
  const totalLoaded = loadedCases.length;
  const needsReviewCases = loadedCases.filter((c) =>
    ['READY_FOR_REVIEW', 'EVIDENCE_REVIEW_REQUIRED'].includes(c.status)
  );
  const abstainedOrFailedCases = loadedCases.filter((c) =>
    ['ABSTAINED', 'FAILED', 'CANCELLED'].includes(c.status)
  );

  const columns: Column<CaseSummary>[] = [
    {
      key: 'id',
      header: 'Case ID',
      render: (row) => (
        <span className="mono-id" title={row.id}>
          {row.id.slice(0, 8)}...
        </span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
    },
    {
      key: 'urgent',
      header: 'Triage Flag',
      render: (row) =>
        row.urgent ? (
          <span className={styles.urgentMarker}>
            <AlertOctagon size={12} aria-hidden="true" />
            Urgent
          </span>
        ) : (
          <span style={{ color: 'var(--color-text-muted)' }}>Standard</span>
        ),
    },
    {
      key: 'latest_execution_status',
      header: 'Latest Execution Status',
      render: (row) =>
        row.latest_execution_status ? (
          <StatusBadge status={row.latest_execution_status} />
        ) : (
          <span style={{ color: 'var(--color-text-muted)', fontSize: 'var(--font-size-xs)' }}>
            None
          </span>
        ),
    },
    {
      key: 'created_at',
      header: 'Created',
      render: (row) => (
        <span
          className="num-tabular"
          style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}
        >
          {new Date(row.created_at).toLocaleString()}
        </span>
      ),
    },
  ];

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1>Dashboard</h1>
          <p>Real-time system overview and recent clinical cases</p>
        </div>
        <Button variant="primary" onClick={() => navigate('/submit')}>
          <PlusCircle size={16} aria-hidden="true" />
          <span>New case submission</span>
        </Button>
      </div>

      {/* Metrics Row: computed ONLY from the loaded cases list */}
      <div className={styles.metricsGrid}>
        <MetricTile
          label="Total Cases"
          value={totalLoaded}
          n={totalLoaded}
          hint="in the loaded list"
        />
        <MetricTile
          label="Needs Review"
          value={needsReviewCases.length}
          n={totalLoaded}
          hint="in the loaded list (READY_FOR_REVIEW / EVIDENCE_REVIEW_REQUIRED)"
        />
        <MetricTile
          label="Abstained or Failed"
          value={abstainedOrFailedCases.length}
          n={totalLoaded}
          hint="in the loaded list (ABSTAINED / FAILED)"
        />
      </div>

      {/* Recent Cases Table */}
      <Card
        title="Recent Cases"
        headerAction={
          <Button
            variant="secondary"
            onClick={() => refetchCases()}
            aria-label="Refresh cases list"
            style={{ minHeight: '32px', padding: 'var(--space-1) var(--space-3)' }}
          >
            <RefreshCw size={14} aria-hidden="true" />
            <span>Refresh</span>
          </Button>
        }
      >
        {loadedCases.length === 0 ? (
          <EmptyState
            title="No cases submitted yet"
            description="Submit a new clinical case narrative to initiate multi-agent research analysis."
            action={
              <Button variant="primary" onClick={() => navigate('/submit')}>
                Submit first case
              </Button>
            }
          />
        ) : (
          <DataTable
            columns={columns}
            data={loadedCases}
            keyExtractor={(row) => row.id}
            ariaLabel="Recent cases table"
          />
        )}
      </Card>

      {/* Split Panels: Needs Review, Abstained/Failed, Dependencies */}
      <div className={styles.panelsGrid}>
        {/* Needs Review Panel */}
        <Card title="Needs Review">
          {needsReviewCases.length === 0 ? (
            <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--font-size-sm)' }}>
              No cases currently requiring clinician review in the loaded list.
            </p>
          ) : (
            <div>
              {needsReviewCases.map((c) => (
                <div key={c.id} className={styles.panelItem}>
                  <div>
                    <span className="mono-id">{c.id.slice(0, 8)}...</span>
                    <div
                      style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}
                    >
                      {new Date(c.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <StatusBadge status={c.status} />
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Abstained and Failed Panel */}
        <Card title="Abstained & Failed Cases">
          {abstainedOrFailedCases.length === 0 ? (
            <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--font-size-sm)' }}>
              No abstained or failed cases in the loaded list.
            </p>
          ) : (
            <div>
              {abstainedOrFailedCases.map((c) => (
                <div key={c.id} className={styles.panelItem}>
                  <div>
                    <span className="mono-id">{c.id.slice(0, 8)}...</span>
                    <div
                      style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}
                    >
                      {c.status === 'ABSTAINED'
                        ? 'Evidence safety gate abstained'
                        : 'Analysis execution halted with error'}
                    </div>
                  </div>
                  <StatusBadge status={c.status} />
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Recent Executions Panel */}
        <Card title="Recent Executions">
          <p
            style={{
              fontSize: 'var(--font-size-xs)',
              color: 'var(--color-text-muted)',
              marginBottom: 'var(--space-3)',
            }}
          >
            Execution deep-links are not available from the API (per CP-10, execution IDs are not
            exposed on GET /cases). Latest execution statuses derived from cases list:
          </p>
          {loadedCases.filter((c) => c.latest_execution_status).length === 0 ? (
            <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--font-size-sm)' }}>
              No recent executions found in loaded cases.
            </p>
          ) : (
            <div>
              {loadedCases
                .filter((c) => c.latest_execution_status)
                .slice(0, 5)
                .map((c) => (
                  <div key={`exec-${c.id}`} className={styles.panelItem}>
                    <div>
                      <span className="mono-id">Case {c.id.slice(0, 8)}...</span>
                      <div
                        style={{
                          fontSize: 'var(--font-size-xs)',
                          color: 'var(--color-text-muted)',
                        }}
                      >
                        Status: {c.latest_execution_status}
                      </div>
                    </div>
                    <span
                      style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}
                    >
                      Not available from the API
                    </span>
                  </div>
                ))}
            </div>
          )}
        </Card>

        {/* System Dependencies Readiness Panel */}
        <Card
          title="System Dependencies"
          headerAction={
            <Button
              variant="secondary"
              onClick={() => refetchDeps()}
              aria-label="Refresh dependency status"
              style={{ minHeight: '32px', padding: 'var(--space-1) var(--space-3)' }}
            >
              <RefreshCw size={14} aria-hidden="true" />
            </Button>
          }
        >
          {isDepsError ? (
            <p style={{ color: 'var(--color-status-failed)', fontSize: 'var(--font-size-sm)' }}>
              Error fetching dependency status:{' '}
              {depsError instanceof Error ? depsError.message : 'Unknown'}
            </p>
          ) : dependencies ? (
            <div className={styles.depsList}>
              <div className={styles.depItem}>
                <div>
                  <strong>PostgreSQL Database</strong>
                  <div className={styles.depMeta}>Relational store & audit trail</div>
                </div>
                <StatusBadge status={dependencies.database.status} />
              </div>

              <div className={styles.depItem}>
                <div>
                  <strong>Qdrant Vector DB</strong>
                  <div className={styles.depMeta}>
                    Points:{' '}
                    <span className="num-tabular">
                      {dependencies.qdrant.points_count != null
                        ? dependencies.qdrant.points_count
                        : 'N/A'}
                    </span>
                  </div>
                </div>
                <StatusBadge status={dependencies.qdrant.status} />
              </div>

              <div className={styles.depItem}>
                <div>
                  <strong>LLM Provider</strong>
                  <div className={styles.depMeta}>
                    Provider: {dependencies.llm.provider || 'None'} | Model:{' '}
                    {dependencies.llm.model || 'None'} | API key present:{' '}
                    {dependencies.llm.api_key_present ? 'yes' : 'no'}
                  </div>
                </div>
                <StatusBadge status={dependencies.llm.status} />
              </div>

              <div className={styles.depItem}>
                <div>
                  <strong>NCBI PubMed Scanner</strong>
                  <div className={styles.depMeta}>
                    Live search probe | Email configured:{' '}
                    {dependencies.ncbi.email_configured ? 'yes' : 'no'}
                  </div>
                </div>
                <StatusBadge status={dependencies.ncbi.status} />
              </div>
            </div>
          ) : null}
        </Card>
      </div>
    </div>
  );
};
