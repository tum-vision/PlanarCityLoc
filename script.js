const header = document.querySelector('.site-header');
new ResizeObserver(() => document.documentElement.style.setProperty('--header-offset', `${header.offsetHeight + 20}px`)).observe(header);

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll('.reveal').forEach((item) => revealObserver.observe(item));

document.querySelector('#copy-bib').addEventListener('click', async (event) => {
  const text = document.querySelector('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(text);
    event.currentTarget.textContent = 'Copied';
    window.setTimeout(() => { event.currentTarget.textContent = 'Copy BibTeX'; }, 1600);
  } catch {
    event.currentTarget.textContent = 'Select text to copy';
  }
});
