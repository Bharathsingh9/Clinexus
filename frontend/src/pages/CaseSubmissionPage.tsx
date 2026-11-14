import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert, Play, CheckCircle2 } from 'lucide-react';
import { apiClient, ApiError } from '../api/client';
import { SCHEMA_MAX_CHARS, SCHEMA_MIN_CHARS } from '../api/schema_constants';
import { FormField } from '../components/FormField';
import { Button } from '../components/Button';
import { AlertBanner } from '../components/AlertBanner';
import { StatusBadge } from '../components/StatusBadge';
import styles from './CaseSubmissionPage.module.css';

interface CreateCaseResult {
  id: string;
  status: string;
  urgent: boolean;
  urgent_banner?: string | null;
  created_at: string;
}

interface AnalyzeResponse {
  execution_id: string;
  status: string;
  case_id: string;
}

export const CaseSubmissionPage: React.FC = () => {
  const navigate = useNavigate();

  // Form states
  const [rawText, setRawText] = useState('');
  const [rawTextError, setRawTextError] = useState('');
  const [formError, setFormError] = useState('');

  // Structured fields (optional context guidance)
  const [patientAge, setPatientAge] = useState('');
  const [patientSex, setPatientSex] = useState('');
  const [symptomDuration, setSymptomDuration] = useState('');

  // Submission & analysis states
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false);
  const [createdCase, setCreatedCase] = useState<CreateCaseResult | null>(null);
  const [analysisError, setAnalysisError] = useState('');

  // Missing-information hints computed ONLY from empty structured fields (never from parsing text)
  const missingStructuredHints: string[] = [];
  if (!patientAge.trim()) missingStructuredHints.push('Age');
  if (!patientSex.trim()) missingStructuredHints.push('Biological sex');
  if (!symptomDuration.trim()) missingStructuredHints.push('Symptom duration');

  const validate = (): boolean => {
    setRawTextError('');
    setFormError('');

    const trimmed = rawText.trim();
    if (!trimmed || trimmed.length < SCHEMA_MIN_CHARS) {
      setRawTextError('Case narrative is required.');
      return false;
    }
    if (trimmed.length > SCHEMA_MAX_CHARS) {
      setRawTextError(
        `Case text exceeds maximum allowed length of ${SCHEMA_MAX_CHARS} characters.`
      );
      return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate() || isSubmitting) return;

    setIsSubmitting(true);
    setFormError('');

    try {
      // POST /cases accepts CreateCaseRequest with raw_text
      const result = await apiClient<CreateCaseResult>('/cases', {
        method: 'POST',
        body: JSON.stringify({ raw_text: rawText.trim() }),
      });
      setCreatedCase(result);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 422) {
          // Field-level error from API without echoing user text
          const details = Array.isArray(err.details)
            ? err.details.map((d: { msg?: string }) => d.msg || 'Invalid field').join('; ')
            : err.message;
          setFormError(`Validation error from server: ${details}`);
        } else {
          setFormError(err.message || 'Failed to submit clinical case');
        }
      } else {
        setFormError('Failed to submit clinical case');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStartAnalysis = async () => {
    if (!createdCase || isStartingAnalysis) return;

    setIsStartingAnalysis(true);
    setAnalysisError('');

    try {
      // Fresh Idempotency-Key per attempt
      const idempotencyKey = crypto.randomUUID();
      const res = await apiClient<AnalyzeResponse>(`/cases/${createdCase.id}/analyze`, {
        method: 'POST',
        idempotencyKey,
      });

      // Navigate to Analysis workflow page with executionId
      navigate(`/cases/${createdCase.id}/analysis/${res.execution_id}`);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setAnalysisError(err.message || 'Failed to trigger workflow execution');
      } else {
        setAnalysisError('Failed to trigger workflow execution');
      }
    } finally {
      setIsStartingAnalysis(false);
    }
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Clinical Case Submission</h1>
        <p>Submit clinical presentation narrative for multi-agent evidence synthesis</p>
      </header>

      {/* Mandatory Privacy Notice */}
      <aside className={styles.privacyBanner} role="note" aria-label="Privacy notice">
        <ShieldAlert size={16} aria-hidden="true" style={{ color: 'var(--color-primary)' }} />
        <span>
          Do not enter real patient identifiers. This research system is not approved for real
          patient data.
        </span>
      </aside>

      {/* Success View */}
      {createdCase ? (
        <div className={styles.successCard} role="status">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <CheckCircle2 size={24} style={{ color: 'var(--color-primary)' }} aria-hidden="true" />
            <h2>Case registered successfully</h2>
          </div>

          <div>
            <p>
              Case ID: <span className="mono-id">{createdCase.id}</span>
            </p>
            <div style={{ marginTop: 'var(--space-1)' }}>
              Status: <StatusBadge status={createdCase.status} />
            </div>
          </div>

          {/* Urgent Banner returned by API shown prominently */}
          {createdCase.urgent && createdCase.urgent_banner && (
            <AlertBanner variant="urgent" title="Urgent Clinical Presentation Flag">
              {createdCase.urgent_banner}
            </AlertBanner>
          )}

          {analysisError && (
            <AlertBanner variant="urgent" title="Analysis error">
              {analysisError}
            </AlertBanner>
          )}

          <div className={styles.actions}>
            <Button
              variant="primary"
              onClick={handleStartAnalysis}
              isLoading={isStartingAnalysis}
              loadingText="Starting analysis..."
            >
              <Play size={16} aria-hidden="true" />
              <span>Start analysis</span>
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                setCreatedCase(null);
                setRawText('');
                setPatientAge('');
                setPatientSex('');
                setSymptomDuration('');
              }}
              disabled={isStartingAnalysis}
            >
              Submit another case
            </Button>
          </div>
        </div>
      ) : (
        /* Submission Form */
        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          {formError && (
            <AlertBanner variant="urgent" title="Submission failed">
              {formError}
            </AlertBanner>
          )}

          {/* Narrative Free-Text Field */}
          <FormField
            id="case-narrative"
            label="Clinical Narrative"
            hint="Describe presentation, clinical findings, history, and physical examination without identifying details"
            error={rawTextError}
            required
          >
            {(fieldProps) => (
              <div>
                <textarea
                  {...fieldProps}
                  rows={8}
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  placeholder="Enter narrative clinical description..."
                  disabled={isSubmitting}
                />
                <div
                  className={`${styles.charCounter} ${
                    rawText.length > SCHEMA_MAX_CHARS * 0.9 ? styles.charCounterWarn : ''
                  }`}
                  aria-live="polite"
                >
                  <span className="num-tabular">
                    {rawText.length} / {SCHEMA_MAX_CHARS} characters
                  </span>
                </div>
              </div>
            )}
          </FormField>

          {/* Structured Fields Section */}
          <div className={styles.structuredSection}>
            <div className={styles.structuredTitle}>
              <span>Structured Context Fields (Optional Guidance)</span>
              <span className={styles.notAvailableNotice}>
                Not available from the API (per CP-10: POST /cases accepts raw narrative only)
              </span>
            </div>

            <FormField id="patient-age" label="Patient Age" hint="Age in years (guidance only)">
              {(fieldProps) => (
                <input
                  {...fieldProps}
                  type="text"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  placeholder="e.g. 54"
                  disabled={isSubmitting}
                />
              )}
            </FormField>

            <FormField
              id="patient-sex"
              label="Biological Sex"
              hint="Male, female, or unrecorded (guidance only)"
            >
              {(fieldProps) => (
                <input
                  {...fieldProps}
                  type="text"
                  value={patientSex}
                  onChange={(e) => setPatientSex(e.target.value)}
                  placeholder="e.g. Male"
                  disabled={isSubmitting}
                />
              )}
            </FormField>

            <FormField
              id="symptom-duration"
              label="Symptom Duration"
              hint="Duration of primary complaint (guidance only)"
            >
              {(fieldProps) => (
                <input
                  {...fieldProps}
                  type="text"
                  value={symptomDuration}
                  onChange={(e) => setSymptomDuration(e.target.value)}
                  placeholder="e.g. 2 hours"
                  disabled={isSubmitting}
                />
              )}
            </FormField>

            {/* Missing-information hints computed ONLY from empty structured fields */}
            {missingStructuredHints.length > 0 && (
              <div className={styles.hintsBox} role="status">
                <strong>Notice:</strong> Structured context fields currently unprovided:{' '}
                {missingStructuredHints.join(', ')}.
              </div>
            )}
          </div>

          <div className={styles.actions}>
            <Button
              type="submit"
              variant="primary"
              isLoading={isSubmitting}
              loadingText="Submitting..."
              disabled={isSubmitting}
            >
              Submit case
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={() => navigate('/')}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
          </div>
        </form>
      )}
    </div>
  );
};
