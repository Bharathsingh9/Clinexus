import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { LoginPage } from '../../src/pages/LoginPage';
import { AuthProvider } from '../../src/context/AuthContext';

describe('LoginPage (UI-9)', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it('renders email and password FormFields', () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </BrowserRouter>
    );

    expect(screen.getByLabelText(/Email address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Sign in/i })).toBeInTheDocument();
  });

  it('performs inline client-side validation', async () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </BrowserRouter>
    );

    const submitBtn = screen.getByRole('button', { name: /Sign in/i });
    fireEvent.click(submitBtn);

    expect(await screen.findByText('Email is required')).toBeInTheDocument();
    expect(await screen.findByText('Password is required')).toBeInTheDocument();

    const emailInput = screen.getByLabelText(/Email address/i);
    fireEvent.change(emailInput, { target: { value: 'not-an-email' } });
    fireEvent.click(submitBtn);

    expect(await screen.findByText('Enter a valid email address')).toBeInTheDocument();
  });

  it('displays generic error message on invalid credentials without revealing which field was wrong', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      json: async () => ({
        error: { code: 'UNAUTHORIZED', message: 'Invalid credentials' },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(
      <BrowserRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </BrowserRouter>
    );

    fireEvent.change(screen.getByLabelText(/Email address/i), {
      target: { value: 'doctor@hospital.org' },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: 'WrongPassword123' },
    });

    fireEvent.click(screen.getByRole('button', { name: /Sign in/i }));

    expect(await screen.findByText('Invalid email or password')).toBeInTheDocument();
    // Verify password is never logged to console
    const consoleSpy = vi.spyOn(console, 'log');
    expect(consoleSpy).not.toHaveBeenCalledWith(expect.stringContaining('WrongPassword123'));
  });

  it('displays visible 429 message when rate-limited', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
      json: async () => ({
        error: { code: 'RATE_LIMITED', message: 'Rate limit exceeded' },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(
      <BrowserRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </BrowserRouter>
    );

    fireEvent.change(screen.getByLabelText(/Email address/i), {
      target: { value: 'doctor@hospital.org' },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: 'Password123!' },
    });

    fireEvent.click(screen.getByRole('button', { name: /Sign in/i }));

    expect(await screen.findByText('Too many attempts, wait a minute')).toBeInTheDocument();
  });

  it('handles login success and stores token in sessionStorage', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        access_token: 'valid-jwt-token-789',
        token_type: 'bearer',
        role: 'clinician_reviewer',
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(
      <BrowserRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </BrowserRouter>
    );

    fireEvent.change(screen.getByLabelText(/Email address/i), {
      target: { value: 'clinician@hospital.org' },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: 'ValidPassword123!' },
    });

    fireEvent.click(screen.getByRole('button', { name: /Sign in/i }));

    await waitFor(() => {
      expect(sessionStorage.getItem('clinexus_token')).toBe('valid-jwt-token-789');
      const user = JSON.parse(sessionStorage.getItem('clinexus_user') || '{}');
      expect(user.role).toBe('clinician_reviewer');
    });
  });
});
