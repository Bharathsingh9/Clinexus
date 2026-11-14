import React from 'react';
import styles from './MetricTile.module.css';

interface MetricTileProps {
  label: string;
  value?: number | string | null;
  n?: number | null;
  unit?: string;
  hint?: string;
  className?: string;
}

export const MetricTile: React.FC<MetricTileProps> = ({
  label,
  value,
  n,
  unit,
  hint,
  className = '',
}) => {
  const isEvaluated = value !== undefined && value !== null;

  return (
    <div className={`${styles.tile} ${className}`}>
      <div className={styles.header}>
        <span className={styles.label}>{label}</span>
        {n !== undefined && n !== null && (
          <span
            className={`${styles.sampleSize} num-tabular`}
            aria-label={`Sample size n equals ${n}`}
          >
            (n = {n})
          </span>
        )}
      </div>
      <div className={styles.valueContainer}>
        {isEvaluated ? (
          <>
            <span className={`${styles.value} num-tabular`}>{value}</span>
            {unit && <span className={styles.hint}>{unit}</span>}
          </>
        ) : (
          <span className={styles.notEvaluated}>Not evaluated</span>
        )}
      </div>
      {hint && <div className={styles.hint}>{hint}</div>}
    </div>
  );
};
