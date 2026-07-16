# Security

## Reporting a vulnerability

Do not open a public GitHub issue for a suspected vulnerability. Report AWS
service vulnerabilities through the
[AWS vulnerability reporting process](https://aws.amazon.com/security/vulnerability-reporting/).
For issues limited to this sample, contact the repository maintainers privately.

## Deployment guidance

- Use a dedicated, least-privilege SageMaker execution role.
- Restrict S3 model prefixes and enable encryption and access logging as
  required by your organization.
- Pin model revisions and serving images for production releases.
- Do not pass access tokens through committed files or command history.
- Consider VPC-only endpoints and network isolation for regulated workloads.
- Treat prompts, retrieved data, tool calls, and generated output as untrusted.
- Evaluate model quality, safety, and tool-use behavior for the intended use
  case before exposing an endpoint.
