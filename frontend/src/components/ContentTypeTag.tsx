import React from 'react';
import { User, BookOpen, Cpu, AlertTriangle, FileEdit } from 'lucide-react';
import styles from './ContentTypeTag.module.css';

export type ContentType =
  | 'patient-provided'
  | 'retrieved-evidence'
  | 'ai-interpretation'
  | 'unverified-claim'
  | 'human-note';

interface ContentTypeTagProps {
  type: ContentType;
  className?: string;
}

const CONFIG: Record<
  ContentType,
  { label: string; icon: React.ComponentType<{ className?: string }>; styleClass: string }
> = {
  'patient-provided': {
    label: 'Patient-provided',
    icon: User,
    styleClass: styles.patient,
  },
  'retrieved-evidence': {
    label: 'Retrieved evidence',
    icon: BookOpen,
    styleClass: styles.evidence,
  },
  'ai-interpretation': {
    label: 'AI interpretation',
    icon: Cpu,
    styleClass: styles.ai,
  },
  'unverified-claim': {
    label: 'Unverified claim',
    icon: AlertTriangle,
    styleClass: styles.unverified,
  },
  'human-note': {
    label: 'Human note',
    icon: FileEdit,
    styleClass: styles.human,
  },
};

export const ContentTypeTag: React.FC<ContentTypeTagProps> = ({ type, className = '' }) => {
  const item = CONFIG[type];
  if (!item) return null;
  const IconComponent = item.icon;

  return (
    <span
      className={`${styles.tag} ${item.styleClass} ${className}`}
      aria-label={`Content type: ${item.label}`}
    >
      <IconComponent className={styles.icon} aria-hidden="true" />
      <span>{item.label}</span>
    </span>
  );
};
