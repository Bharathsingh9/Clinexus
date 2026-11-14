import React from 'react';
import { StatusBadge } from './StatusBadge';
import styles from './AgentTimeline.module.css';

export interface AgentRecord {
  agent_name: string;
  status: string;
  started_at?: string | null;
  duration_ms?: number | null;
  error_code?: string | null;
  error_message?: string | null;
}

interface AgentTimelineProps {
  agents: AgentRecord[];
  className?: string;
}

export const AgentTimeline: React.FC<AgentTimelineProps> = ({ agents, className = '' }) => {
  if (agents.length === 0) {
    return <div className={className}>No agent execution records available yet.</div>;
  }

  return (
    <div
      className={`${styles.container} ${className}`}
      role="feed"
      aria-label="Agent Execution Timeline"
    >
      {agents.map((agent, index) => {
        const isCompleted = agent.status.toUpperCase() === 'COMPLETED';
        const isFailed = ['FAILED', 'ERROR', 'CANCELLED'].includes(agent.status.toUpperCase());
        const isRunning = ['RUNNING', 'ANALYZING', 'IN_PROGRESS'].includes(
          agent.status.toUpperCase()
        );

        let circleClass = styles.circle;
        if (isCompleted) circleClass += ` ${styles.circleCompleted}`;
        else if (isFailed) circleClass += ` ${styles.circleFailed}`;
        else if (isRunning) circleClass += ` ${styles.circleRunning}`;

        return (
          <article
            key={`${agent.agent_name}-${index}`}
            className={styles.agentCard}
            aria-labelledby={`agent-name-${index}`}
          >
            <div className={styles.stepIndicator} aria-hidden="true">
              <div className={circleClass}>{index + 1}</div>
            </div>
            <div className={styles.content}>
              <div className={styles.header}>
                <span id={`agent-name-${index}`} className={styles.agentName}>
                  {agent.agent_name}
                </span>
                <StatusBadge status={agent.status} />
              </div>
              <div className={styles.meta}>
                {agent.started_at && (
                  <span>
                    Started:{' '}
                    <span className="mono-id">
                      {new Date(agent.started_at).toLocaleTimeString()}
                    </span>
                  </span>
                )}
                {agent.duration_ms != null && (
                  <span>
                    Duration:{' '}
                    <span className="num-tabular">{(agent.duration_ms / 1000).toFixed(2)}s</span>
                  </span>
                )}
                {agent.error_code && (
                  <span>
                    Code: <span className="mono-id">{agent.error_code}</span>
                  </span>
                )}
              </div>
              {agent.error_message && (
                <div className={styles.errorBox} role="alert">
                  <strong>Error:</strong> {agent.error_message}
                </div>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
};
