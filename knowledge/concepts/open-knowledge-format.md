---
type: Reference
title: Open Knowledge Format v0.2
description: A source-linked guide to OKF bundle structure, concept metadata, provenance, lifecycle, and OpenWiki 0.2 integration.
tags: [okf, documentation, openwiki, provenance]
status: draft
generated:
  by: okf-skill/0.2
  at: 2026-09-23T14:11:42+00:00
sources:
  - id: okf-v02-spec
    resource: https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md
    title: Open Knowledge Format v0.2 specification
  - id: catalog-okf-spec
    resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
    title: OKF specification copy in knowledge-catalog
  - id: google-cloud-introduction
    resource: https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/
    title: Google Cloud introduction to OKF
  - id: openwiki-02-article
    resource: https://www.langchain.com/blog/openwiki-0-2-adds-okf-support
    title: OpenWiki 0.2 adds OKF support
  - id: okf-home
    resource: https://okf.md/
    title: OKF overview site
---

# Open Knowledge Format v0.2

Open Knowledge Format (OKF) is a portable, human- and agent-readable way to package knowledge as Markdown files with YAML frontmatter. A bundle is a directory tree; it can live inside a larger repository, and it does not require a particular runtime or schema registry.[^okf-v02-spec]

Google Cloud's introduction describes the original v0.1 launch and the goal of sharing context across otherwise separate systems. For current v0.2 structure and metadata, use the versioned specification as the authority.[^google-cloud-introduction][^okf-v02-spec] The supplied `knowledge-catalog/okf/SPEC.md` copy also identifies itself as version 0.2; this note uses the specification in the dedicated `open-knowledge-format` repository for normative v0.2 details.[^catalog-okf-spec][^okf-v02-spec]

## Bundle and concept files

A bundle organizes concepts in directories according to the producer's domain. Each concept is a Markdown document with YAML frontmatter. `type` is the only required frontmatter key; `title`, `description`, and `tags` are recommended descriptive metadata. Type names are not centrally registered, so consumers should tolerate types they do not recognize.[^okf-v02-spec]

`index.md` is a reserved filename for a directory listing, and `log.md` is reserved for update history. Both are optional. Links between concepts are ordinary Markdown links; bundle-root paths beginning with `/` are the stable, recommended form.[^okf-v02-spec]

## Provenance, trust, and lifecycle

The optional `sources` field records materials behind a concept. Give sources stable `id` values when individual statements cite them, then use Markdown footnotes with those IDs so attribution stays attached to claims even when the source list is reordered.[^okf-v02-spec]

`generated` records who or what produced the current content and when. It is not a verification event: `verified` is separate and should only be added when a verifier and confirmation event are known. `status` can mark a concept as `draft`, `stable`, or `deprecated`; without `verified`, a concept is unverified, not rejected.[^okf-v02-spec]

## OpenWiki 0.2

The OpenWiki 0.2 announcement describes generated and updated wikis using OKF-style frontmatter, directory indexes, and update logs. Treat that as OpenWiki integration guidance while retaining the v0.2 specification as the canonical format contract.[^openwiki-02-article][^okf-v02-spec]

There is a filename discrepancy in the announcement: it refers to `logs.md`, while the OKF v0.2 specification reserves the singular `log.md`. Use `log.md` when naming a standards-conforming OKF log file.[^openwiki-02-article][^okf-v02-spec]

## Sources

For an additional overview, see the [OKF site].[^okf-home]

[^okf-v02-spec]: [Open Knowledge Format v0.2 specification](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md).
[^catalog-okf-spec]: [OKF specification copy in knowledge-catalog](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md).
[^google-cloud-introduction]: [Google Cloud introduction to OKF](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/).
[^openwiki-02-article]: [OpenWiki 0.2 adds OKF support](https://www.langchain.com/blog/openwiki-0-2-adds-okf-support).
[^okf-home]: [OKF overview site](https://okf.md/).
