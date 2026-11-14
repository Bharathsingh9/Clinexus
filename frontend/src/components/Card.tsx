import React from 'react';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  title?: React.ReactNode;
  headerAction?: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export const Card: React.FC<CardProps> = ({
  children,
  title,
  headerAction,
  footer,
  className = '',
  bodyClassName = '',
}) => {
  return (
    <article className={`${styles.card} ${className}`}>
      {(title || headerAction) && (
        <header className={styles.header}>
          {typeof title === 'string' ? <h3>{title}</h3> : title}
          {headerAction && <div>{headerAction}</div>}
        </header>
      )}
      <div className={`${styles.body} ${bodyClassName}`}>{children}</div>
      {footer && <footer className={styles.footer}>{footer}</footer>}
    </article>
  );
};
