# SVG Diagram Specification for LLM 101

## Purpose and scope

This file defines the required visual system for every instructional SVG in this repository. It applies when creating, extending, or repairing diagrams under `docs/**/assets/**/*.svg`.

The diagrams are not decorative illustrations. They are technical teaching tools for explaining tensors, model architecture, token flow, attention, GPU execution, memory traffic, and training. A diagram must make the computation easier to understand at a glance before it adds detail.

Use existing diagrams as the visual baseline:

- `docs/part-1-transformer/assets/chapter-4/self-attention-flow.svg`
- `docs/part-1-transformer/assets/chapter-4/kv-cache.svg`
- `docs/part-2-train-a-gpt/assets/chapter-2/gpt2-architecture.svg`
- `docs/part-2-train-a-gpt/assets/chapter-14/ddp-training-flow.svg`
- `docs/part-3-llm-must-knows/assets/chapter-3/kv-cache-incremental-decoding.svg`
- `docs/part-3-llm-must-knows/assets/chapter-4/tiled-gemm.svg`

Do not copy another diagram blindly. Reuse this system, adapt the information architecture to the concept, and preserve the semantic color meaning defined below.

## Core design principles

1. **Concept first.** Show the central operation or data movement before secondary implementation details.
2. **One stable reading path.** A reader should follow the flow primarily left to right or top to bottom. Do not make the eye zigzag between unrelated areas.
3. **Visual explanation over prose.** Put long explanations in the Markdown surrounding the diagram. SVG labels must be short and functional.
4. **Whitespace is required.** Prefer a taller diagram over smaller text, tighter boxes, or crowded arrows.
5. **Semantic consistency.** The same type of thing must retain the same color and visual treatment throughout one diagram and, when practical, across the repository.
6. **Accurate computation.** Matrix orientations, tensor shapes, index ranges, cache contents, and aggregation directions must match the code and accompanying Markdown.
7. **Accessible standalone asset.** Each SVG must contain an informative `<title>` and `<desc>`, then use `role="img"` and `aria-labelledby` to reference them.

## Canvas and responsive sizing

- Use a `viewBox` on every SVG. If explicit `width` and `height` attributes are used, they must match the viewBox aspect ratio.
- Use a wide teaching canvas for most diagrams, normally **1000 to 1400 units wide**.
- Use height freely. Vertical diagrams are preferred for multi-stage processes, repeated blocks, cache lifecycles, and any composition that would otherwise make text too small.
- Typical starting sizes are:
  - compact flow: `1200 × 600` to `1200 × 850`
  - multi-panel flow: `1200 × 900` to `1280 × 1500`
  - deep architecture or step-by-step execution: `1000 × 1800` or taller
- The content must have an outer margin of at least **32 units**. Use **44 to 56 units** for large diagrams.
- If the diagram has a full background card, use a pale near-white fill such as `#fbfdf8` or `#fcfdf9`, with a subtle border. It must render cleanly in both site light mode and dark mode.
- Never shrink the primary text below **14 SVG units** merely to fit more content. Re-layout or increase the canvas height instead.

## Typography

Use the following baseline style unless a diagram has a clearly justified need to differ:

```svg
<style>
  text {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    fill: #243028;
  }
  .section { font-size: 22px; font-weight: 700; }
  .label   { font-size: 18px; font-weight: 680; }
  .small   { font-size: 16px; fill: #586c5d; }
  .tiny    { font-size: 14px; fill: #63766a; }
  .math    { font-family: Georgia, "Times New Roman", serif; font-size: 20px; font-weight: 650; }
</style>
```

- Use `.section` only for meaningful stage labels inside a large multi-stage figure. Do not add a decorative title that repeats the chapter heading.
- Use `.label` for card titles, node names, and short state labels.
- Use `.small` for one-line clarifications and tensor shapes.
- Use `.tiny` only for secondary annotations. It must never contain the sole explanation of a key concept.
- Use `.math` for compact equations, tensor notation, and index expressions. Keep mathematical text separate from arrows and box edges.
- Center text only when it belongs to a centered card or grid. Left-align prose-like labels within wide panels.
- Keep every label inside its parent card with at least **14 units** of padding on all sides. Increase padding for larger type.
- Never let a label cross a border, overlap another label, overlap an arrow, or depend on clipping to fit.
- Use Unicode subscripts and superscripts such as `xₜ`, `C₀₀`, and `Σₖ` when they remain readable. Do not insert tiny explanatory prose into a matrix cell.

## Palette and semantic color system

Use muted, low-saturation colors. Do not introduce bright primary colors, gradients, shadows, or unrelated color families without a semantic reason.

| Meaning | Fill | Border or line | Typical use |
| --- | --- | --- | --- |
| Neutral surface | `#ffffff` or `#f6f9fb` | `#d1dfeb` or `#dbe3df` | explanatory panel, unclassified object |
| Input, activation, token, query | `#e9f2fa` or `#eaf3fb` | `#8fb3d2` or `#4e789f` | input tensors, embeddings, query-like flow |
| Weight, key, lookup, memory read | `#fff0dc` or `#fff4e3` | `#d99c48` or `#c78331` | weights, key/value read, alternate operand |
| Output, aggregate, cache written, complete state | `#edf7e6`, `#eaf5e1`, or `#f3faeb` | `#9dbb88` or `#557f40` | output tensors, shared result, completed region |
| Warning, excluded, unavailable | `#f2f5f8` or `#fffaf2` | `#cbd7df` or `#ecd7ae` | crossed-out or inactive values only |

Rules:

- Blue represents the primary input or activation path.
- Orange represents the second operand, weights, or memory-read path.
- Green represents produced output, aggregation, saved state, or a completed result.
- Gray represents neutral framing or intentionally unavailable information.
- Use the same fill and border pairing for equivalent objects within a figure.
- A selected matrix tile, row, column, or completed output must use both a stronger fill and a stronger border. Do not rely on fill color alone.
- Use at most four semantic color families in one figure. Use white or neutral surfaces for structure instead of inventing additional colors.

## Cards, panels, and containers

- Use rounded rectangles for conceptual nodes and panels.
- Standard node corner radius is **12 to 16 units**. Large stage panels use **18 to 24 units**. Matrix cells use **6 to 10 units** only when rounded corners improve readability.
- Standard card borders are **1.5 to 2 units**. Highlighted selected states may use **2.5 to 4 units**.
- A panel must group one coherent stage, such as `Prefill`, `Decode`, `Load tiles`, or `All-reduce`.
- Every panel requires internal padding. Do not place labels, arrows, or small grids directly against panel edges.
- Use nested panels sparingly. A reader should still be able to see which container is the main stage.
- Do not use drop shadows. Depth should come from spacing, pale panel fills, and border contrast.
- Avoid large empty cards that contain only a single short label unless they represent a meaningful device, component, or memory location.

## Arrows, lines, and connectors

Arrows are part of the explanation. They must be visually precise and never compete with the content.

- Use connector strokes between **2.0 and 2.4 units**. Use round line caps and round joins where possible.
- Use arrowheads around **5 to 8 units**. The arrowhead must be visibly smaller than the meaningful line segment.
- The arrowhead cannot obscure the line. There must be a clear visible shaft before every arrowhead.
- Start and end every arrow outside the content area of a card. Leave at least **12 to 18 units** between the arrow endpoint and any text inside the target card.
- Do not let a line or arrowhead overlap a border, a matrix grid, a formula, a label, or another connector.
- Do not route an arrow through a card unless the line itself is the object being explained.
- Keep a minimum **16-unit** separation between parallel lines, and a minimum **24-unit** separation where lines would otherwise cross.
- Prefer straight horizontal or vertical connectors. Use elbow connectors only to avoid a collision or to represent an intentional branch or merge.
- When two inputs merge, merge their lines in open whitespace before entering the target. Do not place the merge point on top of the target card or label.
- If several arrows share a destination, use a visible fan-in arrangement or a dedicated operation card. Never stack arrowheads on top of one another.
- Use colored arrows only when the color maps to the data category, such as blue for activations and orange for weights. Otherwise use a neutral green-gray stroke such as `#516559`.
- Arrow labels must sit beside a line with ample clearance. Never lay text directly over a line.
- Do not include arrows with no clear source and target. Remove redundant arrows that communicate the same relation twice.

Recommended marker pattern:

```svg
<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
  <path d="M0,0 L0,8 L8,4 z" fill="#516559"/>
</marker>
```

Use a smaller marker, around `5 × 5`, only for dense local details after confirming that the line remains visible at rendered document width.

## Diagram grammar

Choose the visual grammar that matches the concept instead of forcing every topic into the same flowchart.

### Data-flow diagrams

Use for tokenization, next-token generation, attention, cache read and write, forward pass, and training loops.

- Start with the input representation, then show the operation, then the output representation.
- Show state that persists across steps as a separate cache, memory, or accumulator card.
- Use numbered stages only when the sequence itself is essential. Otherwise use simple spatial order.
- If a process repeats, draw one full step and use a concise repeat indicator such as `repeat for each K tile`. Do not redraw identical details unnecessarily.

### Tensor and matrix diagrams

Use for embeddings, logits, attention weights, normalization, GEMM, and batch construction.

- State the tensor shape near the tensor, not in a distant legend.
- Ensure matrix multiplication orientation is correct. If a transposed view is used for teaching clarity, label it explicitly.
- Highlight only the row, column, tile, or element that participates in the current step.
- Use a readable grid, normally four to six visible cells per side for an abstract illustration. Use ellipses rather than drawing an unreadable full-size matrix.
- Preserve index consistency. For example, `A₀₁ × B₁₀` contributes to `C₀₀` in a tiled matrix product.
- For a completed output region, use a green fill and thicker green outline to distinguish it from regions that are merely present.

### Architecture diagrams

Use for Transformer blocks, GPT-2, multi-head attention, and repeated layers.

- Place the main forward path vertically for deep stacks and horizontally for a short single block.
- Draw only the first and final repeated block in detail when repetition is not the teaching target. Use a clear ellipsis between them.
- Show residual or skip connections as a separate bypass path that visibly merges at an add operation. Do not imply residual addition only with prose.
- Use an explicit merge or add node for summed paths. Keep both incoming arrows visible until the merge.
- Label tensor shape changes at boundaries where the shape changes, not at every repeated layer.

### GPU and systems diagrams

Use for GEMM, Flash Attention, compilation, GPU memory traffic, DDP, and data loading.

- Represent compute units as clearly named cards, such as `GPU thread`, `workgroup`, or `GPU 0`.
- Represent data locations as separate regions, such as `global memory`, `shared memory`, `HBM`, or `cache`.
- Use the diagram to show reuse, movement, synchronization, or reduction. Do not replace it with a generic chip icon.
- A tile-based diagram must make clear which data is loaded, where it is stored, what is reused, and what output is accumulated.
- In distributed training diagrams, distinguish local data, local gradients, collective communication, and synchronized parameters with separate visual regions.

## Density and spacing rules

- One major idea per panel. If a panel needs more than three lines of explanatory text, move that explanation to Markdown or split the panel.
- Leave at least **28 to 40 units** between major cards or stages.
- Leave at least **20 to 28 units** between a card and a nearby connector that does not attach to it.
- In a compact grid, maintain at least **12 units** between individual labels or cells when rendered at the intended page width.
- When a diagram becomes dense, use one of these solutions in this order:
  1. increase the SVG height
  2. split the flow into stacked panels
  3. remove repeated prose and redundant arrows
  4. move explanatory detail to the Markdown
  5. only then consider a small, limited reduction in type size
- Never solve density by allowing labels to overlap, reducing key text below 14 units, or shrinking arrow shafts until they disappear.

## Required implementation structure

Every new SVG must include:

```svg
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 WIDTH HEIGHT"
     role="img"
     aria-labelledby="title desc">
  <title id="title">Short descriptive title</title>
  <desc id="desc">A concise explanation of the concept and important relationships shown.</desc>
  <defs>
    <!-- markers and shared styles -->
  </defs>
  <!-- background, stages, nodes, connectors, and annotations -->
</svg>
```

- Keep definitions near the top and use semantic CSS classes such as `.label`, `.small`, `.math`, `.arrow`, `.blue`, and `.orange`.
- Use explicit IDs only when they are locally unique in the SVG.
- Keep related shapes adjacent in source order. A future editor should be able to identify a panel, its labels, and its connectors as one block.
- Add XML comments to separate major stages in large diagrams, for example `<!-- prefill -->`, `<!-- decode -->`, or `<!-- workgroup 0 -->`.
- Do not embed external fonts, raster screenshots, JavaScript, or remote resources.
- Prefer native SVG rectangles, paths, text, and groups. Use an embedded raster image only when the source material cannot be expressed as a diagram and the user explicitly requests it.

## Content constraints

- Diagram language must match the chapter language, which is normally English.
- Do not add a figure title inside the SVG unless it names a stage or is necessary to distinguish multiple panels. The Markdown heading and caption already provide context.
- Avoid generic filler labels such as `Step 1`, `Code note`, or `magic happens here`.
- Do not mention implementation artifacts such as “the first shader” when the pedagogical concept is general. Use general terms such as `GPU thread`, `workgroup`, `shared memory`, or `matrix tile`.
- Do not use semicolons in diagram prose. Use sentences or commas instead.
- Do not claim a computation is parallel, cached, fused, or reused unless the diagram accurately shows the relevant dimensions or data path.
- Explain a new abstraction with one concrete example where helpful, but do not duplicate the full example in text and graphic form.

## Mandatory preflight checklist

Before finishing an SVG task, verify all items below.

### Semantic correctness

- [ ] Tensor dimensions, labels, indices, and operation directions match the chapter code and Markdown.
- [ ] Every color has a consistent meaning.
- [ ] Every highlighted tile, row, column, or output is the exact region involved in the shown operation.
- [ ] Repeated layers, cache values, reductions, and skip connections are represented accurately.

### Geometry and legibility

- [ ] No text overlaps another text node, a border, a matrix grid, an arrow, or a formula.
- [ ] No text extends outside a parent card or outside the SVG canvas.
- [ ] Every arrow has a visible shaft and a small readable head.
- [ ] No arrow, line, or arrowhead overlaps a card boundary or its text content.
- [ ] Connectors have unambiguous sources and targets.
- [ ] Panels have sufficient whitespace and are not artificially compressed.
- [ ] The visual hierarchy remains clear when rendered at normal documentation-column width.

### Repository validation

- [ ] Inspect the actual local Docusaurus page, not only the raw SVG file.
- [ ] Inspect both site light mode and dark mode when the diagram has a visible background or low-contrast colors.
- [ ] For a long or complex figure, scroll through the full figure and inspect every panel at rendered size.
- [ ] Run `git diff --check`.
- [ ] Run `npm run build`.

If any checklist item fails, re-layout the SVG. The correct fix is normally more space, fewer words, clearer connector routing, or a more focused panel. Do not accept a diagram that is technically present but visually difficult to read.
