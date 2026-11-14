import React from 'react';
import { X } from 'lucide-react';
import styles from './Toast.module.css';

export interface ToastMessage {
  id: string;
  message: string;
  type?: 'info' | 'error' | 'success';
}

interface ToastProps {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

export const Toast: React.FC<ToastProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null;

  return (
    <div
      className={styles.toastContainer}
      role="region"
      aria-live="polite"
      aria-label="Notifications"
    >
      {toasts.map((toast) => {
        let borderClass = '';
        if (toast.type === 'error') borderClass = styles.toastError;
        else if (toast.type === 'success') borderClass = styles.toastSuccess;

        return (
          <div key={toast.id} className={`${styles.toast} ${borderClass}`} role="status">
            <span>{toast.message}</span>
            <button
              type="button"
              className={styles.closeButton}
              onClick={() => onDismiss(toast.id)}
              aria-label="Dismiss notification"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        );
      })}
    </div>
  );
};
