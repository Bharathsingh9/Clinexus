import { paths } from './schema';

const DEFAULT_BASE_URL = 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  public readonly code: string;
  public readonly status: number;
  public readonly details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function getApiBaseUrl(): string {
  const envUrl = (import.meta as unknown as { env?: { VITE_API_BASE_URL?: string } }).env
    ?.VITE_API_BASE_URL;
  return envUrl || DEFAULT_BASE_URL;
}

export function getStoredToken(): string | null {
  try {
    return sessionStorage.getItem('clinexus_token');
  } catch {
    return null;
  }
}

export function setStoredToken(token: string | null): void {
  try {
    if (token) {
      sessionStorage.setItem('clinexus_token', token);
    } else {
      sessionStorage.removeItem('clinexus_token');
    }
  } catch {
    // sessionStorage not available
  }
}

export function clearStoredAuth(): void {
  try {
    sessionStorage.removeItem('clinexus_token');
    sessionStorage.removeItem('clinexus_user');
  } catch {
    // sessionStorage not available
  }
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
  idempotencyKey?: string;
}

export async function apiClient<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = 15000, idempotencyKey, headers = {}, ...rest } = options;
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${path}`;

  const requestHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(headers as Record<string, string>),
  };

  const token = getStoredToken();
  if (token) {
    requestHeaders['Authorization'] = `Bearer ${token}`;
  }

  if (idempotencyKey) {
    requestHeaders['Idempotency-Key'] = idempotencyKey;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...rest,
      headers: requestHeaders,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.status === 401) {
      clearStoredAuth();
      if (
        typeof window !== 'undefined' &&
        window.location &&
        window.location.pathname !== '/login' &&
        !(import.meta as unknown as { env?: { MODE?: string } }).env?.MODE?.includes('test')
      ) {
        window.location.href = '/login';
      }
      throw new ApiError(401, 'UNAUTHORIZED', 'Session expired or invalid credentials');
    }

    if (!response.ok) {
      let errorBody: { error?: { code?: string; message?: string; details?: unknown } } | null =
        null;
      try {
        errorBody = await response.json();
      } catch {
        // Not a JSON response
      }

      const code = errorBody?.error?.code || `HTTP_${response.status}`;
      const message = errorBody?.error?.message || response.statusText || 'API request failed';
      const details = errorBody?.error?.details;

      throw new ApiError(response.status, code, message, details);
    }

    if (response.status === 204) {
      return undefined as unknown as T;
    }

    return (await response.json()) as T;
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    if (err instanceof ApiError) {
      throw err;
    }
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError(408, 'REQUEST_TIMEOUT', `Request timed out after ${timeoutMs}ms`);
    }
    throw new ApiError(500, 'NETWORK_ERROR', err instanceof Error ? err.message : 'Network error');
  }
}

export type Paths = paths;
