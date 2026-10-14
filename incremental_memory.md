# Incremental memory

This file records user corrections and revision requests for generated work in this repository. Read it before starting work and update it whenever a correction or requested revision is addressed.

Each entry records the date, scope, feedback, change made, and a concrete rule for future work. Refine an existing entry for the same lesson instead of duplicating it. Keep preferences scoped to the context in which they were requested.

## 2026-10-14 — Cloudflare as the sole production host

- **Scope:** Site URL, base path, crawler policy, README, page-view counter origin, and deployment workflow.
- **Feedback:** Use the existing default Cloudflare address and stop maintaining the GitHub Pages address; commit and push the migration.
- **Change made:** Set the canonical origin to `https://llm-101.core-computings-llm101.workers.dev` with root base path, updated site links and robots.txt, aligned the counter origin, and removed the GitHub Pages deployment workflow.
- **Rule for future work:** Use the confirmed production hostname rather than guessing a Pages project name. The site is served from `/`; the confirmed host determines the deployment platform. Keep canonical URLs, sitemap, crawler policy, and counter CORS origin aligned. Updating the counter source still requires a separate Worker deployment.

- **Follow-up feedback (2026-10-14):** Switch website hosting from Workers to Cloudflare Pages at `https://llm-101.pages.dev`. Updated the canonical origin, links, crawler policy, counter CORS origin, and Pages build instructions. The optional counter remains a separate Worker; the root base path stays unchanged. Use the Pages address for future website deployments.

## 2026-10-13 — Search discovery and crawler configuration

- **Scope:** Sitemap configuration, deployment history, homepage metadata, and robots.txt for LLM 101.
- **Feedback:** Prepare the site for Google Search Console with a sitemap and crawler files.
- **Change made:** Configured the existing automatic sitemap explicitly with Git-based modification dates, enabled full-history checkout, added a homepage description and submission instructions, excluded the two title-only chapters from indexing until they have content, and added a project robots.txt copy plus the effective host-root policy in the root site's repository.
- **Rule for future work:** Keep sitemaps generated from actual routes. On a GitHub Pages project site, robots.txt must be published at the host root to control crawling. Verify public URLs and distinguish crawlability from Google's decision to index or rank a page.

## 2026-10-13 — Part 2 title

- **Scope:** Part 2 display title in `sidebars.js` and `README.md`.
- **Feedback:** Rename Part 2 to GPT2.
- **Change made:** Updated the sidebar category and README title to `Part 2 — GPT2`.
- **Rule for future work:** Use the requested spelling `GPT2` for the Part 2 title. This naming preference does not require changing chapter titles, technical mentions of GPT-2, or existing URLs.

## 2026-10-09 — Repository instructions and correction tracking

- **Scope:** `AGENTS.md`, `svg_specs.md`, and `incremental_memory.md`.
- **Feedback:** Keep `AGENTS.md` concise, move the full SVG requirements into a separate file, and persist the lessons from every user correction or requested revision.
- **Change made:** Moved the existing SVG specification unchanged to `svg_specs.md`, linked it from `AGENTS.md`, and added mandatory instructions to read and update this memory file.
- **Rule for future work:** Use `AGENTS.md` for repository-wide instructions and links to detailed specifications. Whenever the user corrects or asks to revise generated work, complete the revision and record its feedback, change, and prevention rule in `incremental_memory.md` before responding.

## 2026-10-10 — Explain the existing symmetric quantization implementation

- **Scope:** Part 4, Chapter 1, Linear Symmetric Quantization.
- **Feedback:** Add mathematical formulas and an SVG explanation based on the chapter's existing code.
- **Change made:** Added formulas for the shared scale, zero-point, rounding, clipping, reconstruction, and error bound. Added a diagram using the same 3×3 weights and a magnified rounding example, including explanations of the all-zero case and ties-to-even rounding.
- **Rule for future work:** When explaining this implementation, derive the formulas and diagrams from its actual conventions: one scale per tensor, codes in `[-127, 127]`, zero-point 0, `np.rint` rounding, and scale 1 for an all-zero tensor. Verify illustrated values against the executable example and distinguish rounded display values from computed outputs.
- **Follow-up feedback (2026-10-10):** The mathematical explanation was too verbose. Condensed it to the scale, quantization/reconstruction formulas, brief implementation notes, and the error bound. Removed repeated definitions, the extra worked error example, and the outlier discussion.
- **Writing rule for this section:** Let the formulas and code carry the explanation. Use short connecting sentences and essential symbol definitions, without restating each formula or repeating examples already shown in the code.
- **Follow-up feedback (2026-10-10):** Remove the generated SVG. Deleted the diagram asset, its Markdown reference, and the diagram-specific introduction. This supersedes the earlier request for a diagram in this section.
- **Visual rule for this section:** Keep the current explanation in formulas and code. Do not reintroduce the removed SVG unless requested. This preference applies to this section, not to every diagram in the repository.
- **Follow-up feedback (2026-10-10):** Add a conceptual SVG before section 1 to explain floating-point values mapping to discrete INT8 levels. Added a new number-line illustration with a many-to-one mapping, leaving the removed matrix walkthrough out.
- **Placement and scope:** Use the introductory figure to teach the general quantization idea before the implementation-specific section. Distinguish integer codes from the real values they represent, label any illustrative scale, and avoid implying that floating-point values are mathematically continuous.
- **Follow-up feedback (2026-10-10):** The introductory figure was not direct enough and had too much text. Replaced the INT8 cards with a second number line, connected the floating-point samples to integer ticks, and removed explanatory sentences inside the SVG.
- **Visual rule for the introduction:** Prefer two aligned number lines with direct mapping arrows. Let positions, discrete ticks, and many-to-one arrows explain quantization. Keep visible text to axis names, numbers, and the example scale.
- **Follow-up feedback (2026-10-10):** Make the floating-point line longer than the INT8 line and include larger input magnitudes. Changed the illustration to a 960-unit floating-point axis and a 560-unit INT8 axis, with inputs −1200, −800, 300, and 1400 mapping to −2, −2, 1, and 3 using scale 500.
- **Geometry rule for this figure:** Show the floating-point examples above a visibly shorter INT8 axis. Use a consistent, explicitly labeled scale. When mixed magnitudes require non-proportional spacing for readability, label it schematic and mark omitted INT8 intervals with axis breaks.
- **Follow-up feedback (2026-10-10):** Using only values in the hundreds and thousands was too extreme. The requested example should mix fractions, single-digit values, tens, and hundreds. Updated it to −120, −24, −3.6, 0.3, 0.7, 4.2, and 120, with scale 2 and the two fractions sharing code 0.
- **Example-selection rule:** A request to include larger values means broaden the range of examples, not replace all small values with large ones. Preserve mixed magnitudes and avoid overcorrecting toward one extreme.
- **Follow-up feedback (2026-10-10):** Add simple explanatory text below the diagram. Added a short paragraph covering division by the shared scale, rounding to an INT8 code, precision loss when inputs share a code, and approximate reconstruction.
- **Caption rule for this figure:** Keep the SVG minimal and place a brief plain-language explanation directly below it. Explain the mapping and its consequence without repeating every example or adding a long derivation.
- **Follow-up feedback (2026-10-10):** Add a separate diagram inside Linear Symmetric Quantization using the same number-line design, specifically to explain “symmetric.” Added mirrored input/output ranges, paired positive and negative mappings, zero-to-zero mapping, and equal distances from zero to the endpoints, with a brief caption.
- **Follow-up feedback (2026-10-10):** Mirrored sample extrema hid what symmetric quantization does to asymmetric data. Changed the observed data to span −0.8 to 1.6, extended the quantization range to [−1.6, 1.6], and distinguished the added boundary with dashed lines and hollow points. Updated the shared scale to 1.6/127 and verified the actual samples map to −64, 0, 64, and 127.
- **Example rule for symmetry:** Use unequal magnitudes for the observed minimum and maximum to explain symmetric quantization. Visually distinguish actual samples from the range extended using the largest absolute value, so readers see that the quantization range is symmetric even when the data is not.
- **Symmetry rule for this section:** The newly requested diagram is authorized alongside the introductory figure. Explain symmetry through mirrored geometry and a shared scale, using `[-127, 127]` to match the code. Distinguish symmetry of the quantization range from symmetry of the weight distribution, and keep the retired matrix walkthrough removed.

## 2026-10-10 — Center each display formula on its own line

- **Scope:** Mathematical exposition in repository chapters, first applied to Part 4, Chapter 1.
- **Feedback:** When two formulas appear on the same line, place them on separate lines and center each one.
- **Change made:** Split the maximum-magnitude and scale definitions into separate display blocks, and did the same for quantization and reconstruction.
- **Rule for future work:** Give each distinct display formula its own centered line. Use separate `$$` blocks instead of placing independent formulas side by side with spacing commands such as `\qquad`.
