import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './context/AuthContext';
import { RouteGuard } from './components/RouteGuard';
import { AppShell } from './components/AppShell';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { CaseSubmissionPage } from './pages/CaseSubmissionPage';
import { AnalysisWorkflowPage } from './pages/AnalysisWorkflowPage';
import { ComingSoonPage } from './pages/ComingSoonPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />

          <Route
            path="/"
            element={
              <RouteGuard>
                <AppShell>
                  <DashboardPage />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/submit"
            element={
              <RouteGuard>
                <AppShell>
                  <CaseSubmissionPage />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/cases/:id/analysis/:executionId"
            element={
              <RouteGuard>
                <AppShell>
                  <AnalysisWorkflowPage />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/analysis"
            element={
              <RouteGuard>
                <AppShell>
                  <AnalysisWorkflowPage />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/cases/:id/report"
            element={
              <RouteGuard>
                <AppShell>
                  <ComingSoonPage pageName="Explainable Report" />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/report"
            element={
              <RouteGuard>
                <AppShell>
                  <ComingSoonPage pageName="Explainable Report" />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/evidence"
            element={
              <RouteGuard>
                <AppShell>
                  <ComingSoonPage pageName="Evidence Explorer" />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/review"
            element={
              <RouteGuard>
                <AppShell>
                  <ComingSoonPage pageName="Human Review" />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route
            path="/evaluations"
            element={
              <RouteGuard>
                <AppShell>
                  <ComingSoonPage pageName="Evaluation Dashboard" />
                </AppShell>
              </RouteGuard>
            }
          />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </QueryClientProvider>
  );
};
