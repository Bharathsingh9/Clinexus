import React from 'react';
import { AlertTriangle, AlertOctagon, Info } from 'lucide-react';
import styles from './AlertBanner.module.css';

export type AlertBannerVariant = 'urgent' | 'abstained' | 'info';

interface AlertBannerProps {
  variant: AlertBannerVariant;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

const ICONS: Record<AlertBannerVariant, React.ComponentType<{ className?: string }>> = {
  urgent: AlertOctagon,
  abstained: AlertTriangle,
  info: Info,
};

export const AlertBanner: React.FC<AlertBannerProps> = ({
  variant,
  title,
  children,
  className = '',
}) => {
  const IconComponent = ICONS[variant];
  const role = variant === 'urgent' ? 'alert' : 'status';

  return (
    <div
      className={`${styles.banner} ${styles[variant]} ${className}`}
      role={role}
      aria-live={variant === 'urgent' ? 'assertive' : 'polite'}
    >
      <IconComponent className={styles.icon} aria-hidden="true" />
      <div className={styles.content}>
        {title && <div className={styles.title}>{title}</div>}
        <div>{children}</div>
      </div>
    </div>
  );
};
