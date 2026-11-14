import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiClient, clearStoredAuth, getStoredToken, setStoredToken } from '../api/client';

export interface UserInfo {
  email: string;
  role: string;
}

interface AuthContextType {
  user: UserInfo | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

function getInitialUser(): UserInfo | null {
  try {
    const raw = sessionStorage.getItem('clinexus_user');
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [user, setUser] = useState<UserInfo | null>(getInitialUser());

  useEffect(() => {
    const currentToken = getStoredToken();
    if (!currentToken && user) {
      setUser(null);
    }
  }, [user]);

  const login = async (email: string, password: string) => {
    interface LoginResponse {
      access_token: string;
      token_type: string;
      role: string;
    }

    const data = await apiClient<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });

    setStoredToken(data.access_token);
    const userInfo: UserInfo = { email, role: data.role };
    try {
      sessionStorage.setItem('clinexus_user', JSON.stringify(userInfo));
    } catch {
      // sessionStorage unavailable
    }
    setToken(data.access_token);
    setUser(userInfo);
  };

  const logout = () => {
    clearStoredAuth();
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: Boolean(token),
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
