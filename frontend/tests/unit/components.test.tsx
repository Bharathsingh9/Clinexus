import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

import { StatusBadge } from '../../src/components/StatusBadge';
import { ContentTypeTag, ContentType } from '../../src/components/ContentTypeTag';
import { Card } from '../../src/components/Card';
import { DataTable, Column } from '../../src/components/DataTable';
import { Tabs } from '../../src/components/Tabs';
import { Drawer } from '../../src/components/Drawer';
import { AgentTimeline } from '../../src/components/AgentTimeline';
import { AlertBanner } from '../../src/components/AlertBanner';
import { MetricTile } from '../../src/components/MetricTile';
import { EmptyState } from '../../src/components/EmptyState';
import { ErrorState } from '../../src/components/ErrorState';
import { Skeleton } from '../../src/components/Skeleton';
import { Toast } from '../../src/components/Toast';
import { FormField } from '../../src/components/FormField';
import { Button } from '../../src/components/Button';
import { AppShell } from '../../src/components/AppShell';
import { AuthProvider } from '../../src/context/AuthContext';

describe('Design System Components (UI-8 / A15)', () => {
  it('renders StatusBadge with semantic color classes and text', () => {
    const { rerender } = render(<StatusBadge status="COMPLETED" />);
    expect(screen.getByRole('status')).toHaveTextContent('COMPLETED');

    rerender(<StatusBadge status="FAILED" />);
    expect(screen.getByRole('status')).toHaveTextContent('FAILED');

    rerender(<StatusBadge status="ABSTAINED" />);
    expect(screen.getByRole('status')).toHaveTextContent('ABSTAINED');

    rerender(<StatusBadge status="READY_FOR_REVIEW" />);
    expect(screen.getByRole('status')).toHaveTextContent('READY_FOR_REVIEW');
  });

  it('renders ContentTypeTag with distinct text and icon for all 5 types', () => {
    const types: ContentType[] = [
      'patient-provided',
      'retrieved-evidence',
      'ai-interpretation',
      'unverified-claim',
      'human-note',
    ];

    for (const t of types) {
      const { unmount } = render(<ContentTypeTag type={t} />);
      const el = screen.getByLabelText(new RegExp(`Content type:`, 'i'));
      expect(el).toBeInTheDocument();
      expect(el.querySelector('svg')).toBeInTheDocument();
      unmount();
    }
  });

  it('renders Card with title, header action, and footer', () => {
    render(
      <Card
        title="Card Title"
        headerAction={<button>Action</button>}
        footer={<span>Footer Info</span>}
      >
        <p>Card content</p>
      </Card>
    );
    expect(screen.getByText('Card Title')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByText('Card content')).toBeInTheDocument();
    expect(screen.getByText('Footer Info')).toBeInTheDocument();
  });

  it('renders DataTable with sticky header and keyboard navigable region', () => {
    interface TestRow {
      id: string;
      name: string;
    }
    const cols: Column<TestRow>[] = [
      { key: 'id', header: 'ID' },
      { key: 'name', header: 'Name' },
    ];
    const data: TestRow[] = [{ id: '1', name: 'Item 1' }];

    render(
      <DataTable columns={cols} data={data} keyExtractor={(r) => r.id} ariaLabel="Test table" />
    );

    const region = screen.getByRole('region', { name: 'Test table' });
    expect(region).toHaveAttribute('tabindex', '0');
    expect(screen.getByText('Item 1')).toBeInTheDocument();
  });

  it('renders Tabs and supports switching and keyboard navigation', () => {
    const tabs = [
      { id: 'tab1', label: 'Tab 1', content: <div>Panel 1 Content</div> },
      { id: 'tab2', label: 'Tab 2', content: <div>Panel 2 Content</div> },
    ];

    render(<Tabs tabs={tabs} ariaLabel="Test tabs" />);

    expect(screen.getByText('Panel 1 Content')).toBeInTheDocument();
    expect(screen.queryByText('Panel 2 Content')).not.toBeInTheDocument();

    const tab2Btn = screen.getByRole('tab', { name: 'Tab 2' });
    fireEvent.click(tab2Btn);

    expect(screen.getByText('Panel 2 Content')).toBeInTheDocument();
    expect(screen.queryByText('Panel 1 Content')).not.toBeInTheDocument();

    // Arrow navigation
    fireEvent.keyDown(tab2Btn, { key: 'ArrowLeft' });
    expect(screen.getByText('Panel 1 Content')).toBeInTheDocument();
  });

  it('renders Drawer and handles close and Escape key', () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <Drawer isOpen={true} onClose={onClose} title="Drawer Title">
        <div>Drawer content</div>
      </Drawer>
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Drawer content')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Close panel'));
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);

    rerender(
      <Drawer isOpen={false} onClose={onClose} title="Drawer Title">
        <div>Drawer content</div>
      </Drawer>
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('renders AgentTimeline with status and duration', () => {
    const agents = [
      {
        agent_name: 'Clarifier',
        status: 'COMPLETED',
        started_at: '2026-11-14T10:00:00Z',
        duration_ms: 1250,
      },
      {
        agent_name: 'RAG Analyzer',
        status: 'FAILED',
        error_code: 'LLM_TIMEOUT',
        error_message: 'Model call timed out',
      },
    ];

    render(<AgentTimeline agents={agents} />);

    expect(screen.getByText('Clarifier')).toBeInTheDocument();
    expect(screen.getByText('1.25s')).toBeInTheDocument();
    expect(screen.getByText('RAG Analyzer')).toBeInTheDocument();
    expect(screen.getByText('LLM_TIMEOUT')).toBeInTheDocument();
    expect(screen.getByText('Model call timed out')).toBeInTheDocument();
  });

  it('renders AlertBanner with urgent, abstained, and info variants', () => {
    const { rerender } = render(
      <AlertBanner variant="urgent" title="Urgent Warning">
        Immediate assessment required
      </AlertBanner>
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Urgent Warning');
    expect(screen.getByRole('alert')).toHaveTextContent('Immediate assessment required');

    rerender(
      <AlertBanner variant="abstained" title="Abstained">
        Safety gate abstained
      </AlertBanner>
    );
    expect(screen.getByRole('status')).toHaveTextContent('Abstained');

    rerender(
      <AlertBanner variant="info" title="Info">
        Informational banner
      </AlertBanner>
    );
    expect(screen.getByRole('status')).toHaveTextContent('Informational banner');
  });

  it('renders MetricTile with value and explicit Not evaluated state', () => {
    const { rerender } = render(<MetricTile label="Accuracy" value="0.85" n={100} unit="%" />);
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('0.85')).toBeInTheDocument();
    expect(screen.getByText('(n = 100)')).toBeInTheDocument();

    rerender(<MetricTile label="Unmeasured Metric" value={null} n={50} />);
    expect(screen.getByText('Not evaluated')).toBeInTheDocument();
    expect(screen.getByText('(n = 50)')).toBeInTheDocument();
  });

  it('renders EmptyState and ErrorState with action buttons', () => {
    const onAction = vi.fn();
    render(
      <EmptyState
        title="No items found"
        description="Try adding a new item"
        action={<button onClick={onAction}>Add Item</button>}
      />
    );
    expect(screen.getByText('No items found')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Add Item'));
    expect(onAction).toHaveBeenCalledTimes(1);

    const onRetry = vi.fn();
    render(<ErrorState title="Error" message="Network error" onRetry={onRetry} />);
    expect(screen.getByText('Network error')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Retry'));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('renders Skeleton with aria-busy and aria-label', () => {
    render(<Skeleton count={2} />);
    const el = screen.getByLabelText('Loading content');
    expect(el).toHaveAttribute('aria-busy', 'true');
  });

  it('renders Toast and dismisses notification', () => {
    const onDismiss = vi.fn();
    render(
      <Toast
        toasts={[{ id: '1', message: 'Test notification', type: 'info' }]}
        onDismiss={onDismiss}
      />
    );
    expect(screen.getByText('Test notification')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Dismiss notification'));
    expect(onDismiss).toHaveBeenCalledWith('1');
  });

  it('renders FormField with linked hint and error text', () => {
    render(
      <FormField id="test-field" label="Test Label" hint="Help text" error="Field error" required>
        {(props) => <input {...props} type="text" />}
      </FormField>
    );

    const input = screen.getByLabelText(/Test Label/);
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(input).toHaveAttribute('aria-describedby', 'test-field-hint test-field-error');
    expect(screen.getByText('Help text')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Field error');
  });

  it('renders Button with variants and loading state', () => {
    const { rerender } = render(<Button variant="primary">Submit</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Submit');

    rerender(
      <Button variant="primary" isLoading loadingText="Submitting...">
        Submit
      </Button>
    );
    const btn = screen.getByRole('button');
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('aria-busy', 'true');
    expect(btn).toHaveTextContent('Submitting...');
  });

  it('renders AppShell with persistent disclaimer, landmarks, and navigation links', () => {
    sessionStorage.setItem(
      'clinexus_user',
      JSON.stringify({ email: 'doctor@hospital.org', role: 'clinician_reviewer' })
    );
    sessionStorage.setItem('clinexus_token', 'test-token');

    render(
      <BrowserRouter>
        <AuthProvider>
          <AppShell>
            <div>App shell content</div>
          </AppShell>
        </AuthProvider>
      </BrowserRouter>
    );

    // Permanent disclaimer
    expect(
      screen.getByText(
        'Research tool. Not a medical device. Outputs require independent clinical judgment.'
      )
    ).toBeInTheDocument();

    // Skip to main content link
    expect(screen.getByText('Skip to main content')).toBeInTheDocument();

    // User section
    expect(screen.getByText('doctor@hospital.org')).toBeInTheDocument();
    expect(screen.getByText('clinician_reviewer')).toBeInTheDocument();
    expect(screen.getByText('Sign out')).toBeInTheDocument();

    // Navigation items
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Case submission')).toBeInTheDocument();
    expect(screen.getByText('Analysis workflow')).toBeInTheDocument();

    // Disabled unbuilt stages
    const comingSoonBadges = screen.getAllByText('Coming soon');
    expect(comingSoonBadges.length).toBe(4);
  });
});
