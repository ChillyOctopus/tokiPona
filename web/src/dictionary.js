// The dictionary page's random-word card: a word from the page, toki pona on one side and its English on the other.
// Tapping the card turns it over; Another word picks a new word and keeps whichever side is showing. The side is
// remembered between visits when the browser allows it. Built with textContent only.

const card = document.getElementById('random');
const flip = document.getElementById('flip');
const sideLabel = document.getElementById('flip-side');
const text = document.getElementById('flip-text');
const hint = document.getElementById('flip-hint');
const another = document.getElementById('another');
const link = document.getElementById('random-link');
// Only words most speakers use, so the card teaches the everyday vocabulary
const words = [...document.querySelectorAll('article.word[data-usage="core"], article.word[data-usage="common"]')].map((article) => ({
  word: article.id,
  english: article.dataset.brief,
}));

let english = false;
try {
  english = localStorage.getItem('random-side') === 'english';
} catch {
  // storage blocked: start on the toki pona side
}
let current;

function show() {
  sideLabel.textContent = english ? 'English' : 'toki pona';
  text.textContent = english ? current.english : current.word;
  text.className = english ? 'flip-text flip-text--english' : 'flip-text';
  hint.textContent = english ? 'Tap the card to see the toki pona.' : 'Tap the card to see the English.';
  link.textContent = `See ${current.word} in full`;
  link.href = `#${encodeURIComponent(current.word)}`;
}

function pick() {
  let next;
  do next = words[Math.floor(Math.random() * words.length)];
  while (words.length > 1 && next === current);
  current = next;
  show();
}

flip.addEventListener('click', () => {
  english = !english;
  try {
    localStorage.setItem('random-side', english ? 'english' : 'toki-pona');
  } catch {
    // not remembered; it still turns over
  }
  show();
});
another.addEventListener('click', pick);

if (words.length) {
  pick();
  card.hidden = false;
}

// The round button at the bottom right, as on jistevenson.com: it appears once you've scrolled about a screen down and
// takes you back to the top, smoothly unless you've asked for reduced motion. Keyboard focus goes back to the top too.
const toTop = document.getElementById('back-to-top');
const showToTop = () => toTop.toggleAttribute('data-shown', scrollY > innerHeight);
addEventListener('scroll', showToTop, { passive: true });
showToTop();
toTop.addEventListener('click', () => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
  document.querySelector('.top a')?.focus({ preventScroll: true });
});
