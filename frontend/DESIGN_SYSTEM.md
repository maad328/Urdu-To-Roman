# Design System - Urdu Translation App

## Color Palette

### Dark Theme Foundation
Our unified dark theme provides excellent contrast and accessibility while maintaining visual appeal.

#### Base Colors
```css
--bg: #0b0f1a;                /* Page background (dark navy) */
--bg-gradient-from: #1b164a;  /* Gradient start (deep purple) */
--bg-gradient-to: #ee4b6a;    /* Gradient end (coral pink) */
--surface: #121727;           /* Card/panel backgrounds */
--surface-2: #1a2140;         /* Elevated/hover surfaces */
--border: #2a3156;            /* Border color */
```

#### Interactive Colors
```css
--primary: #8b5cf6;           /* Primary violet */
--primary-foreground: #ffffff; /* Text on primary */
--primary-hover: #7c3aed;     /* Primary hover state */
--primary-light: #a78bfa;     /* Light primary variant */
```

#### Semantic Colors
```css
--accent: #22d3ee;            /* Cyan accents/highlights */
--success: #10b981;           /* Success states */
--warning: #f59e0b;           /* Warning states */
--error: #ef4444;             /* Error states */
```

#### Typography
```css
--text: #e6e9f5;              /* Primary text */
--text-secondary: #c4c8d4;    /* Secondary text */
--muted: #a6b0cf;             /* Muted/disabled text */
--text-muted: #9ca3af;        /* Alternative muted */
```

## Usage Guidelines

### Buttons
- **Primary Actions**: Use `--primary` with `--primary-foreground`
- **Secondary Actions**: Use `--surface-2` with `--text`
- **Success Actions**: Use `--success` with `--primary-foreground`
- **Error Actions**: Use `--error` with `--primary-foreground`

### Cards & Surfaces
- **Main Cards**: `--surface` background with `--border`
- **Elevated Cards**: `--surface-2` background for hover/focus
- **Text**: Use `--text` for headings, `--muted` for labels

### Focus States
- **Focus Ring**: 2px `--accent` (cyan)
- **Focus Ring Offset**: 2px `--bg` for proper contrast
- **Hover**: 8% color lightening effect
- **Active**: 12% color darkening effect

## Component Examples

### Translation Cards
```css
.input-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
}
```

### Model Information
```css
.info-item {
  background: var(--surface-2);
  border: 1px solid var(--border);
  color: var(--text);
}

.info-item label {
  color: var(--muted);
}
```

### Interactive Elements
```css
.translate-btn {
  background: linear-gradient(135deg, var(--primary), var(--primary-hover));
  color: var(--primary-foreground);
}

.copy-btn {
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border);
}
```

## Accessibility

### WCAG AA Compliance
- Text contrast ratios ≥ 4.5:1 against backgrounds
- Focus indicators clearly visible with 2px cyan outline
- Reduced motion support for animations
- High contrast mode support

### Color Blindness Support
- Primary reliance on violet/cyan combination
- Semantic colors use different hues (green success, red error, orange warning)
- No color-only information conveyance

## Customization

### Changing the Theme
Update CSS variables in `:root` to customize colors:

```css
:root {
  /* Change primary color */
  --primary: #6366f1;          /* Indigo instead of violet */
  
  /* Adjust background gradient */
  --bg-gradient-from: #1e293b; /* Slate start */
  --bg-gradient-to: #dc2626;   /* Red end */
  
  /* Modify accent color */
  --accent: #10b981;           /* Green accent */
}
```

### Adding Light Mode
To add light mode support, create a media query or class:

```css
@media (prefers-color-scheme: light) {
  :root {
    --bg: #ffffff;
    --surface: #f8fafc;
    --surface-2: #f1f5f9;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
  }
}
```

## Best Practices

1. **Always use CSS variables** - Never hard-code colors
2. **Test contrast ratios** - Ensure accessibility compliance
3. **Use semantic naming** - `--success` not `--green`
4. **Maintain consistency** - Use the same border radius values
5. **Progressive enhancement** - Fallbacks for older browsers

## Migration Notes

All hard-coded hex colors and RGB values have been replaced with the token system. The design now features:

- Unified dark theme with consistent color grading
- Radial gradient background for visual depth
- Enhanced shadows and glass morphism effects
- Improved accessibility and focus states
- Consistent border radius (rounded-xl standard)
- WCAG AA compliant contrast ratios