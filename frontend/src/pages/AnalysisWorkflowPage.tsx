import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { FileText, ArrowLeft, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { AgentTimeline, AgentRecord } from '../components/AgentTimeline';
import { StatusBadge } from '../components/StatusBadge';
import { AlertBanner } from '../components/AlertBanner';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Skeleton } from '../components/Skeleton';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import styles from './AnalysisWorkflowPage.module.css';

export interface ExecutionStatus {
  id: string;
  case_id: string;
  status: string;
  started_at: string;
  completed_at?: string | null;
  duration_seconds?: number | null;
  error_code?: string | null;
  error_message?: string | null;
}

const TERMINAL_STATUSES = ['COMPLETED', 'ABSTAINED', 'FAILED', 'CANCELLED'];

export const ERROR_EXPLANATIONS: Record<string, string> = {
  LLM_UNAVAILABLE: 'No language model is configured',
  LLM_QUOTA_EXHAUSTED: "The model provider's quota is used up",
  LLM_RATE_LIMITED: 'Provider rate limit reached; backoff applied',
  LLM_TIMEOUT: 'Model response timed out',
  LLM_AUTH_ERROR: 'Invalid or unauthorized model credentials',
  VERIFICATION_FAILED: 'Clinical claim verification criteria not met',
};

export function getErrorExplanation(code?: string | null): string | null {
  if (!code) return null;
  return ERROR_EXPLANATIONS[code] || `Error: ${code}`;
}

export const AnalysisWorkflowPage: React.FC = () => {
  const { id: caseId, executionId } = useParams<{ id: string; executionId: string }>();
  const navigate = useNavigate();

  // Track tab visibility to pause polling when tab is hidden
  const [isTabVisible, setIsTabVisible] = useState(
    typeof document !== 'undefined' ? document.visibilityState === 'visible' : true
  );

  useEffect(() => {
    const handleVisibility = () => {
      setIsTabVisible(document.visibilityState === 'visible');
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, []);

  const {
    data: execution,
    isLoading: isExecLoading,
    isError: isExecError,
    error: execError,
    refetch: refetchExecution,
  } = useQuery<ExecutionStatus>({
    queryKey: ['execution', executionId],
    queryFn: () => apiClient<ExecutionStatus>(`/executions/${executionId}`),
    enabled: Boolean(executionId),
    refetchInterval: (query) => {
      if (!isTabVisible) return false;
      const status = query.state.data?.status?.toUpperCase();
      if (status && TERMINAL_STATUSES.includes(status)) {
        return false;
      }
      // Backoff from 2000ms to 5000ms based on fetch count
      const count = query.state.dataUpdateCount;
      return count > 5 ? 5000 : 2000;
    },
  });

  const {
    data: agents = [],
    isLoading: isAgentsLoading,
    refetch: refetchAgents,
  } = useQuery<AgentRecord[]>({
    queryKey: ['execution-agents', executionId],
    queryFn: () => apiClient<AgentRecord[]>(`/executions/${executionId}/agents`),
    enabled: Boolean(executionId),
    refetchInterval: () => {
      if (!isTabVisible) return false;
      const status = execution?.status?.toUpperCase();
      if (status && TERMINAL_STATUSES.includes(status)) {
        return false;
      }
      return 2500;
    },
  });

  if (!executionId) {
    return (
      <div className={styles.page}>
        <h1>Analysis Workflow</h1>
        <EmptyState
          title="No active execution specified"
          description="Select a case from the dashboard or submit a new case to start an analysis workflow."
          action={
            <Button variant="primary" onClick={() => navigate('/')}>
              Go to Dashboard
            </Button>
          }
        />
      </div>
    );
  }

  if (isExecLoading) {
    return (
      <div className={styles.page}>
        <h1>Analysis Workflow</h1>
        <Card>
          <Skeleton height="80px" count={1} />
        </Card>
        <Card title="Agent Execution Pipeline">
          <Skeleton height="150px" count={3} />
        </Card>
      </div>
    );
  }

  if (isExecError || !execution) {
    return (
      <div className={styles.page}>
        <h1>Analysis Workflow</h1>
        <ErrorState
          title="Failed to load execution status"
          message={execError instanceof Error ? execError.message : 'Execution not found'}
          onRetry={() => {
            refetchExecution();
            refetchAgents();
          }}
        />
      </div>
    );
  }

  const isTerminal = TERMINAL_STATUSES.includes(execution.status.toUpperCase());
  const errorExplanation = getErrorExplanation(execution.error_code);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <button
            type="button"
            onClick={() => navigate('/')}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-primary)',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 'var(--space-1)',
              fontSize: 'var(--font-size-sm)',
              marginBottom: 'var(--space-2)',
              padding: 0,
            }}
          >
            <ArrowLeft size={16} aria-hidden="true" />
            <span>Back to Dashboard</span>
          </button>
          <h1>Multi-Agent Analysis Workflow</h1>
          <p>
            Execution <span className="mono-id">{execution.id}</span> for Case{' '}
            <span className="mono-id">{execution.case_id || caseId}</span>
          </p>
        </div>

        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          <Button
            variant="secondary"
            onClick={() => {
              refetchExecution();
              refetchAgents();
            }}
            aria-label="Refresh workflow status"
          >
            <RefreshCw size={14} aria-hidden="true" />
            <span>Refresh</span>
          </Button>

          {isTerminal && (
            <Button
              variant="primary"
              onClick={() => navigate(`/cases/${execution.case_id || caseId}/report`)}
            >
              <FileText size={16} aria-hidden="true" />
              <span>View Report</span>
            </Button>
          )}
        </div>
      </header>

      {/* Current Overall Status with aria-live */}
      <section className={styles.statusCard} aria-labelledby="exec-status-heading">
        <div className={styles.statusInfo}>
          <h2 id="exec-status-heading" style={{ fontSize: 'var(--font-size-base)' }}>
            Execution Status:
          </h2>
          <div aria-live="polite" aria-atomic="true">
            <StatusBadge status={execution.status} />
          </div>
        </div>

        <div className={styles.timing}>
          <span>
            Started:{' '}
            <span className="mono-id">{new Date(execution.started_at).toLocaleTimeString()}</span>
          </span>
          {execution.duration_seconds != null && (
            <span style={{ marginLeft: 'var(--space-4)' }}>
              Duration:{' '}
              <span className="num-tabular">{execution.duration_seconds.toFixed(1)}s</span>
            </span>
          )}
        </div>
      </section>

      {/* Failure explanation in plain language */}
      {execution.error_code && (
        <AlertBanner variant="urgent" title={`Failure code: ${execution.error_code}`}>
          {errorExplanation || execution.error_message || 'The execution encountered an error.'}
        </AlertBanner>
      )}

      {/* Abstention notification */}
      {execution.status.toUpperCase() === 'ABSTAINED' && (
        <AlertBanner variant="abstained" title="Evidence Safety Gate Abstained">
          The multi-agent workflow abstained from generating a diagnostic recommendation due to
          insufficient, conflicting, or unverified retrieved evidence.
        </AlertBanner>
      )}

      {/* Agent Timeline */}
      <Card title="Agent Execution Pipeline">
        {isAgentsLoading && agents.length === 0 ? (
          <Skeleton height="80px" count={4} />
        ) : (
          <AgentTimeline agents={agents} />
        )}
      </Card>

      {/* Terminal State Notice */}
      {isTerminal && (
        <aside className={styles.terminalNotice} role="note">
          <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
            Execution completed. Notice: Outputs are research findings generated by multi-agent
            retrieval and do not constitute clinical validation.
          </p>
        </aside>
      )}
    </div>
  );
};
