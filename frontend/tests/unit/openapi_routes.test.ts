import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('OpenAPI Route Conformance', () => {
  it('confirms every API endpoint path used by frontend exists in docs/openapi.json', () => {
    const openapiPath = path.resolve(__dirname, '../../../docs/openapi.json');
    const openapi = JSON.parse(fs.readFileSync(openapiPath, 'utf-8'));
    const declaredPaths: string[] = Object.keys(openapi.paths);

    // List of all endpoint routes invoked by frontend client calls (with /api/v1 prefix)
    const clientEndpointTemplates = [
      '/api/v1/auth/login',
      '/api/v1/cases',
      '/api/v1/cases/{id}',
      '/api/v1/cases/{id}/analyze',
      '/api/v1/executions/{id}',
      '/api/v1/executions/{id}/agents',
      '/api/v1/health/dependencies',
    ];

    for (const endpoint of clientEndpointTemplates) {
      expect(
        declaredPaths,
        `Expected endpoint "${endpoint}" to exist in docs/openapi.json`
      ).toContain(endpoint);
    }
  });
});
