const railItems = document.querySelectorAll('.rail-item');
const formFields = document.getElementById('form-fields');
const intakeForm = document.getElementById('intake-form');
const runBtn = document.getElementById('run-btn');
const activeTitle = document.getElementById('active-title');
const tierBadge = document.getElementById('active-tier-badge');
const tierNote = document.getElementById('active-tier-note');

const resultEmpty = document.getElementById('result-empty');
const resultContent = document.getElementById('result-content');
const resultLoading = document.getElementById('result-loading');
const resultYesNo = document.getElementById('result-yesno');
const resultConfidence = document.getElementById('result-confidence-value');
const resultTierNote = document.getElementById('result-tier-note');
const gaugeNeedle = document.getElementById('gauge-needle');

let activeKey = null;

function tierClass(tier) {
  return { 1: 't1', 2: 't2', 3: 't3' }[tier] || '';
}

function renderFields(key) {
  const cfg = CANCERS[key];
  formFields.innerHTML = '';

  cfg.fields.forEach(field => {
    const wrap = document.createElement('div');
    wrap.className = 'field';

    const label = document.createElement('label');
    label.className = 'field-label';
    label.textContent = field.label;
    label.htmlFor = `f-${field.name}`;
    wrap.appendChild(label);

    let input;
    if (field.type === 'select') {
      input = document.createElement('select');
      field.options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        input.appendChild(o);
      });
    } else {
      input = document.createElement('input');
      input.type = 'number';
      input.step = field.step || 'any';
      input.value = 0;
    }
    input.id = `f-${field.name}`;
    input.name = field.name;
    input.required = true;
    wrap.appendChild(input);

    formFields.appendChild(wrap);
  });

  runBtn.disabled = false;
}

function selectCancer(key) {
  activeKey = key;
  const cfg = CANCERS[key];

  railItems.forEach(btn => btn.classList.toggle('active', btn.dataset.key === key));

  activeTitle.textContent = cfg.label;
  tierBadge.textContent = `TIER ${cfg.tier}`;
  tierBadge.className = `tier-badge show ${tierClass(cfg.tier)}`;
  tierNote.textContent = cfg.tier_note;

  renderFields(key);
  resetResult();
}

function resetResult() {
  resultContent.classList.add('hidden');
  resultLoading.classList.add('hidden');
  resultEmpty.classList.remove('hidden');
  gaugeNeedle.style.transform = 'rotate(-90deg)';
}

function setNeedle(confidencePct) {
  // gauge spans -90deg (0%) to +90deg (100%)
  const angle = -90 + (confidencePct / 100) * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;
}

railItems.forEach(btn => {
  btn.addEventListener('click', () => selectCancer(btn.dataset.key));
});

intakeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!activeKey) return;

  const values = {};
  new FormData(intakeForm).forEach((val, key) => { values[key] = val; });

  resultEmpty.classList.add('hidden');
  resultContent.classList.add('hidden');
  resultLoading.classList.remove('hidden');

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cancer: activeKey, values })
    });
    const data = await res.json();

    resultLoading.classList.add('hidden');

    if (data.error) {
      resultEmpty.classList.remove('hidden');
      resultEmpty.querySelector('p').textContent = `Error: ${data.error}`;
      return;
    }

    resultYesNo.textContent = data.prediction;
    resultYesNo.className = `result-yesno ${data.prediction === 'Yes' ? 'yes' : 'no'}`;
    resultConfidence.textContent = `${data.confidence}%`;
    resultTierNote.textContent = `Tier ${data.tier}: ${data.tier_note}`;

    resultContent.classList.remove('hidden');
    setNeedle(data.confidence);
  } catch (err) {
    resultLoading.classList.add('hidden');
    resultEmpty.classList.remove('hidden');
    resultEmpty.querySelector('p').textContent = 'Could not reach the server.';
  }
});

// select first module by default
const firstKey = Object.keys(CANCERS)[0];
if (firstKey) selectCancer(firstKey);
