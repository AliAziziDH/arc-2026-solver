# 🎨 Image Generation Prompts for ARC-AGI Solver

This document contains ready-to-use prompts for generating the visual assets needed for the project's README and gallery. Use these with Midjourney, DALL-E 3, Bing Image Creator, or any AI image generation tool.

---

## 1. Banner Image (`docs/images/banner.png`)

**Purpose:** Main banner at the top of the README.

**Prompt:**
> A futuristic abstract banner for an AI research project about ARC-AGI visual grid reasoning, dark background, glowing neon blue and orange grid puzzles, matrix style, high-tech, cinematic lighting, 8k resolution.

**Alternative (more detailed):**
> A wide cinematic banner showing a 3D holographic grid puzzle floating in dark space, with glowing blue and orange cells transforming and rotating, neural network connections in the background, futuristic AI research lab aesthetic, ultra-detailed, 8k, wide aspect ratio 16:9.

---

## 2. Grid Puzzle Transformation (`docs/images/solved_rotation.png`)

**Purpose:** Show a solved ARC task example (input → output transformation).

**Prompt:**
> A clean infographic showing a 2D colorful grid puzzle transformation, pixel art style, bright vibrant colors on a dark minimalist UI, showing input grid turning into output grid, tech documentation style.

**Alternative:**
> A side-by-side comparison of two 2D pixel grids, left side labeled "INPUT" and right side labeled "OUTPUT", showing a rotation transformation, clean flat design, dark background with neon accents, professional tech documentation style.

---

## 3. Architecture Diagram (`docs/images/architecture.png`)

**Purpose:** Visual representation of the solver architecture.

**Prompt:**
> A professional software architecture diagram for an AI solver system, showing a central orchestrator connected to a beam search engine and an LLM lifeline module, with core primitives at the bottom, clean flat design with blue and orange color scheme, dark background, tech documentation style, high resolution.

---

## 4. Logo / Icon (`docs/images/logo.png`)

**Purpose:** Small icon/logo for the project.

**Prompt:**
> A minimal modern logo for an AI puzzle solver, featuring a stylized 3x3 grid with a glowing brain or neural node in the center, flat vector style, blue and orange gradient on dark background, clean, professional, scalable.

---

## 5. Gallery: Object Extraction (`docs/images/gallery_extract.png`)

**Purpose:** Show the `extract_largest` primitive in action.

**Prompt:**
> A pixel art visualization showing a grid with multiple colored objects, with the largest object highlighted and extracted to a separate grid, clean educational infographic style, dark background, vibrant colors, tech documentation aesthetic.

---

## 6. Gallery: Gravity Down (`docs/images/gallery_gravity.png`)

**Purpose:** Show the `gravity_down` primitive in action.

**Prompt:**
> A pixel art visualization showing colored blocks falling to the bottom of a grid, like gravity simulation, before and after comparison, clean educational infographic style, dark background, vibrant colors, tech documentation aesthetic.

---

## 7. Gallery: Color Replacement (`docs/images/gallery_color.png`)

**Purpose:** Show the `replace_color` primitive in action.

**Prompt:**
> A pixel art visualization showing a grid where one color is being replaced by another, with color mapping arrows, before and after comparison, clean educational infographic style, dark background, vibrant colors, tech documentation aesthetic.

---

## 8. Gallery: Symmetrize (`docs/images/gallery_symmetrize.png`)

**Purpose:** Show the `symmetrize_hv` macro in action.

**Prompt:**
> A pixel art visualization showing a grid being symmetrized both horizontally and vertically, with mirror reflection lines, before and after comparison, clean educational infographic style, dark background, vibrant colors, tech documentation aesthetic.

---

## Usage Instructions

1. **Midjourney:** Use `/imagine` command and paste the prompt
2. **DALL-E 3:** Paste the prompt in ChatGPT or the DALL-E interface
3. **Bing Image Creator:** Paste the prompt at bing.com/create
4. **Stable Diffusion:** Use the prompt with a suitable model (e.g., SDXL)

After generating, save the images to `docs/images/` and update the README to reference them:

```markdown
![Banner](docs/images/banner.png)
```

---

## Recommended Image Sizes

| Image | Recommended Size | Aspect Ratio |
|-------|-----------------|--------------|
| `banner.png` | 1920×480 | 4:1 |
| `solved_rotation.png` | 1024×512 | 2:1 |
| `architecture.png` | 1600×900 | 16:9 |
| `logo.png` | 512×512 | 1:1 |
| Gallery images | 800×400 | 2:1 |