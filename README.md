# Zubr Website

A static website for Zubr, an AI-driven plant detection company focused on agriculture and land management. This site is designed to be hosted on GitHub Pages.

## About Zubr

Zubr builds AI-driven plant detection tools for agriculture and land managers. We combine handheld near-infrared (NIR) analyzers for hay bale screening with drone-based hyperspectral mapping for noxious weed detection (e.g., invasive annual grasses like ventenata). We help hay producers, ranchers, and weed programs detect contamination early, map infestations, and prioritize treatment.

## Site Structure

The website includes the following pages:

- **Home** (`index.html`) - Landing page with hero, benefits, and how it works
- **Products** (`products.html`) - HayGuard™, FieldScan™, and SpectraCloud™ products
- **Technology** (`technology.html`) - Technical details on detection methods
- **Use Cases** (`use-cases.html`) - Real-world applications
- **Pricing** (`pricing.html`) - Pilot, Pro, and Enterprise tiers
- **About** (`about.html`) - Mission, team, and partners
- **Resources** (`resources.html`) - Technical brief and sample outputs
- **FAQ** (`faq.html`) - Frequently asked questions
- **Contact** (`contact.html`) - Contact form

## Running Locally

This is a static site with no build step required. To run locally:

1. Clone or download the repository
2. Open `index.html` in your web browser

For a local development server (optional):

```bash
# Using Python 3
python -m http.server 8000

# Using Node.js with npx
npx serve
```

Then visit `http://localhost:8000` (or the port shown).

## Deployment via GitHub Pages

This site is designed to be deployed via GitHub Pages:

1. Push the code to a GitHub repository
2. Go to Settings → Pages
3. Under "Source", select the branch (e.g., `main`) and root folder (`/`)
4. Click Save
5. The site will be available at `https://[username].github.io/[repo-name]/`

## File Structure

```
├── index.html              # Home page
├── products.html           # Products page
├── technology.html         # Technology page
├── use-cases.html          # Use cases page
├── pricing.html            # Pricing page
├── about.html              # About page
├── resources.html          # Resources page
├── faq.html                # FAQ page
├── contact.html            # Contact page
├── logo.png                # Company logo (add your own)
├── technical-brief.pdf     # Technical brief (add your own)
├── README.md               # This file
└── assets/
    ├── css/
    │   └── styles.css      # Main stylesheet
    ├── js/
    │   └── main.js         # JavaScript for navigation and FAQ
    └── images/             # Image assets (add as needed)
```

## Customization

### Logo
Replace `logo.png` in the root directory with your company logo. The logo is used:
- In the navigation header (top-left)
- As the favicon

### Technical Brief
Add your technical brief PDF as `technical-brief.pdf` in the root directory.

### Colors
Edit the CSS variables in `assets/css/styles.css` to change the color scheme:

```css
:root {
    --color-primary: #2a7f62;      /* Main accent color */
    --color-primary-dark: #1f5f49;
    --color-primary-light: #3a9f7a;
    --color-secondary: #1a4d3e;
    /* ... */
}
```

### Content
Edit the HTML files directly to update text content, add images, or modify structure.

## Browser Support

This site uses modern CSS (Grid, Flexbox, CSS Variables) and is compatible with:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome for Android)

## Accessibility

The site includes:
- Skip link for keyboard navigation
- Semantic HTML structure
- ARIA labels where appropriate
- Sufficient color contrast
- Focus indicators for interactive elements
- Alt text placeholders for images

## License

© 2025 Zubr. All rights reserved.
