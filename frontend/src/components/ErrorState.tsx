import React from 'react';
import { AlertCircle } from 'lucide-react';
import styles from './ErrorState.module.css';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'An error occurred',
  message,
  onRetry,
  className = '',
}) => {
  return (
    <div className={`${styles.errorState} ${className}`} role="alert" aria-live="assertive">
      <AlertCircle className={styles.icon} aria-hidden="true" />
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.message}>{message}</p>
      {onRetry && (
        <div className={styles.action}>
          <button
            type="button"
            onClick={onRetry}
            style={{
              padding: 'var(--space-2) var(--space-4)',
              background: 'var(--color-bg-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
              minHeight: 'var(--touch-target-min)',
              fontWeight: 'var(--font-weight-medium)',
            }}
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
};
