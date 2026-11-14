import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  apiClient,
  ApiError,
  setStoredToken,
  getStoredToken,
  clearStoredAuth,
} from '../../src/api/client';

describe('API Client (apiClient)', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it('sets and retrieves token from sessionStorage only', () => {
    setStoredToken('test-token-xyz');
    expect(getStoredToken()).toBe('test-token-xyz');
    expect(sessionStorage.getItem('clinexus_token')).toBe('test-token-xyz');

    clearStoredAuth();
    expect(getStoredToken()).toBeNull();
    expect(sessionStorage.getItem('clinexus_token')).toBeNull();
  });

  it('includes Authorization header when token is stored', async () => {
    setStoredToken('auth-jwt-123');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ success: true }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await apiClient<{ success: boolean }>('/health');
    expect(result).toEqual({ success: true });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/v1/health');
    expect(init.headers['Authorization']).toBe('Bearer auth-jwt-123');
    expect(init.headers['Content-Type']).toBe('application/json');
  });

  it('includes Idempotency-Key header when provided', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 'queued' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await apiClient('/cases/123/analyze', {
      method: 'POST',
      idempotencyKey: 'idemp-uuid-456',
    });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers['Idempotency-Key']).toBe('idemp-uuid-456');
  });

  it('maps server error shape {error: {code, message, details}} to ApiError', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Case text exceeds limit',
          details: { max_chars: 20000 },
        },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiClient('/cases')).rejects.toThrowError(ApiError);
    try {
      await apiClient('/cases');
    } catch (err: unknown) {
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(400);
      expect(apiErr.code).toBe('INVALID_REQUEST');
      expect(apiErr.message).toBe('Case text exceeds limit');
      expect(apiErr.details).toEqual({ max_chars: 20000 });
    }
  });

  it('clears session and triggers redirect on 401 response', async () => {
    setStoredToken('expired-jwt');
    sessionStorage.setItem('clinexus_user', JSON.stringify({ email: 'dr@hospital.org' }));

    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      json: async () => ({
        error: {
          code: 'UNAUTHORIZED',
          message: 'Session expired',
        },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiClient('/cases')).rejects.toThrowError('Session expired');
    expect(sessionStorage.getItem('clinexus_token')).toBeNull();
    expect(sessionStorage.getItem('clinexus_user')).toBeNull();
  });

  it('does not retry on 4xx errors', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({
        error: { code: 'VALIDATION_ERROR', message: 'Invalid payload' },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiClient('/cases')).rejects.toThrowError(ApiError);
    // Verified: exactly one fetch call, no retry on 4xx
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
