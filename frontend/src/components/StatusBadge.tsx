import React from 'react';
import styles from './StatusBadge.module.css';

export type StatusVariant = 'completed' | 'partial' | 'failed' | 'abstained' | 'human';

interface StatusBadgeProps {
  status: string;
  variant?: StatusVariant;
  className?: string;
}

export function getStatusVariant(status: string): StatusVariant {
  const s = status.toUpperCase();
  if (
    [
      'FAILED',
      'CANCELLED',
      'CONTRADICTED',
      'DOWN',
      'URGENT',
      'UNAVAILABLE',
      'TIMEOUT',
      'ERROR',
    ].some((k) => s.includes(k))
  ) {
    return 'failed';
  }
  if (s === 'ABSTAINED' || s.includes('ABSTAIN')) {
    return 'abstained';
  }
  if (
    [
      'PARTIAL',
      'READY_FOR_REVIEW',
      'EVIDENCE_REVIEW_REQUIRED',
      'REVISION_REQUESTED',
      'UNVERIFIED',
      'PENDING',
      'WARNING',
    ].some((k) => s.includes(k))
  ) {
    return 'partial';
  }
  if (['HUMAN', 'NOTE', 'REVIEWED', 'CLINICIAN'].some((k) => s.includes(k))) {
    return 'human';
  }
  // Default completed / neutral blue-gray
  return 'completed';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, variant, className = '' }) => {
  const activeVariant = variant ?? getStatusVariant(status);
  const variantClass = styles[activeVariant] || styles.completed;

  return (
    <span className={`${styles.badge} ${variantClass} ${className}`} role="status">
      {status}
    </span>
  );
};
