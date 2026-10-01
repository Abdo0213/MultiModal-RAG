document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const imageInput = document.getElementById('imageInput');
  const dropPrompt = document.getElementById('dropPrompt');
  const previewWrapper = document.getElementById('previewWrapper');
  const imagePreview = document.getElementById('imagePreview');
  const btnRemoveImage = document.getElementById('btnRemoveImage');
  const textQuery = document.getElementById('textQuery');
  const chipButtons = document.querySelectorAll('.chip-btn');
  const activeQueryMode = document.getElementById('activeQueryMode');

  const settingsToggle = document.getElementById('settingsToggle');
  const settingsBody = document.getElementById('settingsBody');
  const settingsChevron = document.getElementById('settingsChevron');
  const fusionMode = document.getElementById('fusionMode');
  const alphaSlider = document.getElementById('alphaSlider');
  const alphaValue = document.getElementById('alphaValue');
  const topKImages = document.getElementById('topKImages');
  const topKTexts = document.getElementById('topKTexts');

  const btnSubmit = document.getElementById('btnSubmit');
  const submitSpinner = document.getElementById('submitSpinner');
  const btnSubmitText = document.getElementById('btnSubmitText');

  const emptyAnswerPlaceholder = document.getElementById('emptyAnswerPlaceholder');
  const answerText = document.getElementById('answerText');
  const totalLatencyBadge = document.getElementById('totalLatencyBadge');
  const dbStatusPill = document.getElementById('dbStatusPill');
  const dbStatusText = document.getElementById('dbStatusText');

  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const imageGallery = document.getElementById('imageGallery');
  const chunksList = document.getElementById('chunksList');
  const imageCountBadge = document.getElementById('imageCountBadge');
  const textCountBadge = document.getElementById('textCountBadge');

  const lightboxModal = document.getElementById('lightboxModal');
  const modalImage = document.getElementById('modalImage');
  const modalCaption = document.getElementById('modalCaption');
  const btnCloseModal = document.getElementById('btnCloseModal');

  let currentImageFile = null;

  // 1. Initial Status Check
  checkDbStatus();

  async function checkDbStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        const dot = dbStatusPill.querySelector('.status-dot');
        dot.classList.add('active');
        dbStatusText.textContent = `ChromaDB Ready &bull; ${data.images} images, ${data.texts} text chunks`;
      } else {
        dbStatusText.textContent = 'Vector DB offline';
      }
    } catch (e) {
      dbStatusText.textContent = 'Backend ready';
    }
  }

  // Load Sample Images for quick testing
  const sampleThumbnails = document.getElementById('sampleThumbnails');
  loadSampleImages();

  async function loadSampleImages() {
    try {
      const res = await fetch('/api/sample-images');
      if (res.ok) {
        const data = await res.json();
        const samples = data.samples || [];
        if (samples.length > 0 && sampleThumbnails) {
          sampleThumbnails.innerHTML = samples.map(s => `
            <button type="button" class="sample-pill" data-url="${s.url}" data-breed="${s.breed}">
              <img src="${s.url}" alt="${s.breed}" />
              <span>${s.breed}</span>
            </button>
          `).join('');

          sampleThumbnails.querySelectorAll('.sample-pill').forEach(btn => {
            btn.addEventListener('click', async () => {
              const url = btn.getAttribute('data-url');
              const breed = btn.getAttribute('data-breed');
              try {
                const imgRes = await fetch(url);
                const blob = await imgRes.blob();
                const file = new File([blob], `${breed.toLowerCase().replace(/\s+/g, '_')}.jpg`, { type: blob.type || 'image/jpeg' });
                handleImageSelected(file);
                if (!textQuery.value.trim()) {
                  textQuery.value = `Tell me about the ${breed} breed, its personality traits, and grooming requirements.`;
                }
                updateQueryMode();
              } catch (err) {
                console.error('Failed to load sample image:', err);
              }
            });
          });
        }
      }
    } catch (e) {
      console.log('Sample images not yet loaded');
    }
  }

  // 2. Mode & Query Updates
  function updateQueryMode() {
    const hasImage = !!currentImageFile;
    const hasText = textQuery.value.trim().length > 0;

    if (hasImage && hasText) {
      activeQueryMode.textContent = 'Multimodal Fusion';
      activeQueryMode.style.color = '#818cf8';
    } else if (hasImage) {
      activeQueryMode.textContent = 'Image-Only';
      activeQueryMode.style.color = '#06b6d4';
    } else if (hasText) {
      activeQueryMode.textContent = 'Text-Only';
      activeQueryMode.style.color = '#a855f7';
    } else {
      activeQueryMode.textContent = 'Empty Query';
      activeQueryMode.style.color = '#94a3b8';
    }
  }

  textQuery.addEventListener('input', updateQueryMode);

  chipButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      textQuery.value = btn.getAttribute('data-query');
      updateQueryMode();
      textQuery.focus();
    });
  });

  // 3. Image Drag & Drop and Preview
  dropZone.addEventListener('click', (e) => {
    if (e.target !== btnRemoveImage) {
      imageInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelected(e.dataTransfer.files[0]);
    }
  });

  imageInput.addEventListener('change', () => {
    if (imageInput.files && imageInput.files[0]) {
      handleImageSelected(imageInput.files[0]);
    }
  });

  function handleImageSelected(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPEG, PNG, WebP).');
      return;
    }
    currentImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropPrompt.classList.add('hidden');
      previewWrapper.classList.remove('hidden');
      updateQueryMode();
    };
    reader.readAsDataURL(file);
  }

  btnRemoveImage.addEventListener('click', (e) => {
    e.stopPropagation();
    currentImageFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    previewWrapper.classList.add('hidden');
    dropPrompt.classList.remove('hidden');
    updateQueryMode();
  });

  // 4. Settings Accordion & Slider
  settingsToggle.addEventListener('click', () => {
    settingsBody.classList.toggle('collapsed');
    settingsChevron.innerHTML = settingsBody.classList.contains('collapsed') ? '&#9656;' : '&#9662;';
  });

  alphaSlider.addEventListener('input', () => {
    alphaValue.textContent = parseFloat(alphaSlider.value).toFixed(2);
  });

  // 5. Tabs
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(targetTab).classList.add('active');
    });
  });

  function switchTab(tabId) {
    tabButtons.forEach(b => {
      if (b.getAttribute('data-tab') === tabId) {
        b.classList.add('active');
      } else {
        b.classList.remove('active');
      }
    });
    tabPanes.forEach(p => {
      if (p.id === tabId) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });
  }

  // 6. Submit Query
  btnSubmit.addEventListener('click', async () => {
    const hasImage = !!currentImageFile;
    const queryStr = textQuery.value.trim();

    if (!hasImage && !queryStr) {
      alert('Please provide an image or enter a text prompt.');
      return;
    }

    // Set UI loading state
    btnSubmit.disabled = true;
    submitSpinner.classList.remove('hidden');
    btnSubmitText.textContent = 'Searching & Generating...';
    emptyAnswerPlaceholder.classList.add('hidden');
    answerText.classList.remove('hidden');
    answerText.innerHTML = '<span style="color: var(--text-muted);">Retrieving multimodal vector embeddings and querying vision LLM...</span>';

    const formData = new FormData();
    if (currentImageFile) {
      formData.append('image', currentImageFile);
    }
    if (queryStr) {
      formData.append('text', queryStr);
    }
    formData.append('fusion', fusionMode.value);
    formData.append('alpha', alphaSlider.value);
    formData.append('top_k_images', topKImages.value);
    formData.append('top_k_texts', topKTexts.value);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      renderResults(data);
    } catch (err) {
      answerText.innerHTML = `<div style="color: var(--danger);">Error processing query: ${err.message}</div>`;
    } finally {
      btnSubmit.disabled = false;
      submitSpinner.classList.add('hidden');
      btnSubmitText.textContent = 'Retrieve & Synthesize Answer';
    }
  });

  // 7. Render Results
  function renderResults(data) {
    // 7.1 Latency
    if (data.latency) {
      totalLatencyBadge.textContent = `${data.latency.total_sec}s (${data.latency.retrieval_sec}s ret + ${data.latency.generation_sec}s gen)`;
      totalLatencyBadge.classList.remove('hidden');
    }

    // 7.2 Answer formatting with interactive citation tags
    const rawAnswer = data.answer || 'No answer generated.';
    const formattedAnswer = rawAnswer.replace(/\[(text|image):(\d+)\]/g, (match, type, idx) => {
      return `<span class="citation-tag" data-type="${type}" data-idx="${idx}">${match}</span>`;
    });
    answerText.innerHTML = formattedAnswer;

    // Attach click handlers to citation tags
    document.querySelectorAll('.citation-tag').forEach(tag => {
      tag.addEventListener('click', () => {
        const type = tag.getAttribute('data-type');
        const idx = parseInt(tag.getAttribute('data-idx'), 10);
        if (type === 'text') {
          switchTab('textsTab');
          highlightElement(`chunk-item-${idx}`);
        } else if (type === 'image') {
          switchTab('imagesTab');
          highlightElement(`gallery-card-${idx}`);
        }
      });
    });

    // 7.3 Render Images Gallery
    const images = data.retrieved_images || [];
    imageCountBadge.textContent = images.length;
    if (images.length === 0) {
      imageGallery.innerHTML = '<div class="empty-tab-hint">No matching images retrieved.</div>';
    } else {
      imageGallery.innerHTML = images.map((img, i) => {
        const idx = i + 1;
        const label = img.metadata?.label || 'unknown';
        const score = img.score || 0;
        const scorePct = Math.round(score * 100);
        const imgUrl = `/media/${img.metadata?.path || ''}`;

        return `
          <div class="gallery-card" id="gallery-card-${idx}" data-img="${imgUrl}" data-label="${label}">
            <div class="gallery-img-box">
              <span class="card-idx-badge">[image:${idx}]</span>
              <img src="${imgUrl}" alt="${label}" loading="lazy" onerror="this.src='/static/placeholder.jpg'" />
            </div>
            <div class="gallery-info">
              <div class="gallery-label">${label.replace(/_/g, ' ')}</div>
              <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
              </div>
              <div class="score-text">Sim: ${score.toFixed(3)}${img.rrf_score ? ` &bull; RRF: ${img.rrf_score.toFixed(4)}` : ''}</div>
            </div>
          </div>
        `;
      }).join('');

      // Attach lightbox handler
      document.querySelectorAll('.gallery-card').forEach(card => {
        card.addEventListener('click', () => {
          modalImage.src = card.getAttribute('data-img');
          modalCaption.textContent = `Breed: ${card.getAttribute('data-label')}`;
          lightboxModal.classList.remove('hidden');
        });
      });
    }

    // 7.4 Render Text Chunks
    const texts = data.retrieved_texts || [];
    textCountBadge.textContent = texts.length;
    if (texts.length === 0) {
      chunksList.innerHTML = '<div class="empty-tab-hint">No matching text passages retrieved.</div>';
    } else {
      chunksList.innerHTML = texts.map((t, i) => {
        const idx = i + 1;
        const label = t.metadata?.label || 'unknown';
        const source = t.metadata?.source || 'Wikipedia';
        const score = t.score || 0;
        const tokens = t.metadata?.token_count || 0;
        const doc = t.document || '';

        return `
          <div class="chunk-item" id="chunk-item-${idx}">
            <div class="chunk-header">
              <div>
                <span class="chunk-badge">[text:${idx}]</span>
                <span style="font-weight: 600; font-size: 0.85rem; margin-left: 0.5rem; text-transform: capitalize;">${label.replace(/_/g, ' ')}</span>
              </div>
              <div class="chunk-source">${source} &bull; ${tokens} tok &bull; Sim: ${score.toFixed(3)}</div>
            </div>
            <div class="chunk-body">${doc}</div>
          </div>
        `;
      }).join('');
    }
  }

  function highlightElement(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('highlighted');
      setTimeout(() => {
        el.classList.remove('highlighted');
      }, 2500);
    }
  }

  // 8. Modal Close
  btnCloseModal.addEventListener('click', () => {
    lightboxModal.classList.add('hidden');
  });
  lightboxModal.addEventListener('click', (e) => {
    if (e.target === lightboxModal) {
      lightboxModal.classList.add('hidden');
    }
  });
});
