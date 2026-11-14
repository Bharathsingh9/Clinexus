import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { FormField } from '../components/FormField';
import { Button } from '../components/Button';
import { AlertBanner } from '../components/AlertBanner';
import { ApiError } from '../api/client';
import styles from './LoginPage.module.css';

export const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [emailError, setEmailError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [generalError, setGeneralError] = useState('');
  const [rateLimitError, setRateLimitError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const validate = (): boolean => {
    let isValid = true;
    setEmailError('');
    setPasswordError('');
    setGeneralError('');
    setRateLimitError('');

    if (!email.trim()) {
      setEmailError('Email is required');
      isValid = false;
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setEmailError('Enter a valid email address');
      isValid = false;
    }

    if (!password) {
      setPasswordError('Password is required');
      isValid = false;
    }

    return isValid;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    setIsLoading(true);
    setGeneralError('');
    setRateLimitError('');

    try {
      await login(email.trim(), password);
      const from = (location.state as { from?: { pathname?: string } })?.from?.pathname || '/';
      navigate(from, { replace: true });
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 429 || err.code === 'RATE_LIMITED') {
          setRateLimitError('Too many attempts, wait a minute');
        } else {
          // Generic failure message that never reveals which field was wrong
          setGeneralError('Invalid email or password');
        }
      } else {
        setGeneralError('Invalid email or password');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.loginCard}>
        <div className={styles.header}>
          <h1 className={styles.title}>Clinexus</h1>
          <p className={styles.subtitle}>Sign in with your clinical research credentials</p>
        </div>

        {rateLimitError && (
          <AlertBanner variant="urgent" title="Rate limit reached">
            {rateLimitError}
          </AlertBanner>
        )}

        {generalError && !rateLimitError && (
          <AlertBanner variant="urgent" title="Authentication failed">
            {generalError}
          </AlertBanner>
        )}

        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          <FormField id="login-email" label="Email address" error={emailError} required>
            {(fieldProps) => (
              <input
                {...fieldProps}
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="clinician@hospital.org"
                disabled={isLoading}
              />
            )}
          </FormField>

          <FormField id="login-password" label="Password" error={passwordError} required>
            {(fieldProps) => (
              <input
                {...fieldProps}
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
              />
            )}
          </FormField>

          <Button
            type="submit"
            variant="primary"
            isLoading={isLoading}
            loadingText="Signing in..."
            style={{ width: '100%', marginTop: 'var(--space-2)' }}
          >
            Sign in
          </Button>
        </form>

        <footer className={styles.disclaimer}>
          Research tool. Not a medical device. Outputs require independent clinical judgment.
        </footer>
      </div>
    </div>
  );
};
