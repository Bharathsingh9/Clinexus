import React from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { EmptyState } from '../components/EmptyState';
import { Button } from '../components/Button';

interface ComingSoonPageProps {
  pageName: string;
}

export const ComingSoonPage: React.FC<ComingSoonPageProps> = ({ pageName }) => {
  const navigate = useNavigate();
  const { id } = useParams<{ id?: string }>();

  return (
    <div>
      <h1>{pageName}</h1>
      <div style={{ marginTop: 'var(--space-6)' }}>
        <EmptyState
          title={`${pageName} - Coming in the next stage`}
          description={
            id
              ? `Stage C will provide the detailed ${pageName.toLowerCase()} interface for case ${id}.`
              : `This page is scheduled for Stage C / Stage D and is not yet built in Stage B.`
          }
          action={
            <Button variant="primary" onClick={() => navigate('/')}>
              Return to Dashboard
            </Button>
          }
        />
      </div>
    </div>
  );
};
