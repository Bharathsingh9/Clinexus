import React from 'react';
import styles from './Skeleton.module.css';

interface SkeletonProps {
  width?: string;
  height?: string;
  borderRadius?: string;
  className?: string;
  count?: number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = '1rem',
  borderRadius,
  className = '',
  count = 1,
}) => {
  return (
    <div
      aria-busy="true"
      aria-label="Loading content"
      style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}
    >
      {Array.from({ length: count }).map((_, idx) => (
        <span
          key={idx}
          className={`${styles.skeleton} ${className}`}
          style={{ width, height, borderRadius }}
        />
      ))}
    </div>
  );
};
