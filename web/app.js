document.addEventListener('DOMContentLoaded', () => {
  const modelSelect = document.getElementById('model-select');
  const effortSelect = document.getElementById('effort-select');
  const mcpSelect = document.getElementById('mcp-select');
  const btnClear = document.getElementById('btn-clear');
  const btnNewChat = document.getElementById('btn-new-chat');
  const chatStream = document.getElementById('chat-stream');
  const userInput = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const btnStop = document.getElementById('btn-stop');
  const queueBanner = document.getElementById('queue-banner');
  const queueText = document.getElementById('queue-text');
  
  const sessionsList = document.getElementById('sessions-list');

  // Project Path elements
  const projectPath = document.getElementById('project-path');
  const btnChangeFolder = document.getElementById('btn-change-folder');

  let currentSessionId = localStorage.getItem('apex_active_session_id') || '';
  let isGenerating = false;
  let currentAbortController = null;
  let generationStartedAt = 0;  // timestamp ms — para o watchdog detectar estado preso
  let messageQueue = [];

  function updateQueueBanner() {
    if (!queueBanner || !queueText) return;
    if (messageQueue.length > 0) {
      queueBanner.style.display = 'flex';
      queueText.textContent = `${messageQueue.length} mensagem${messageQueue.length > 1 ? 's' : ''} na fila (será enviada automaticamente a seguir)`;
    } else {
      queueBanner.style.display = 'none';
    }
  }

  function setGeneratingState(generating) {
    isGenerating = generating;
    generationStartedAt = generating ? Date.now() : 0;
    if (btnStop) {
      btnStop.style.display = generating ? 'flex' : 'none';
    }
    if (generating) {
      btnSend.disabled = false;
      btnSend.title = 'Enviar mensagem para a fila (Enter)';
    } else {
      btnSend.disabled = false;
      btnSend.classList.remove('btn-stop');
      btnSend.innerHTML = '<span>Enviar</span><span class="send-icon">➔</span>';
      btnSend.title = 'Enviar mensagem (Enter)';
      currentAbortController = null;
    }
  }

  if (btnStop) {
    btnStop.addEventListener('click', () => {
      if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
      }
      setGeneratingState(false);
    });
  }

  // ── i18n Internationalization (EN / PT-BR) ──────────────────────────
  const translations = {
    pt: {
      app_title: "Apex Harness — Web UI & Model Switcher",
      new_chat: "Nova Conversa (/clear)",
      monitor: "Monitor",
      monitor_toggle: "Clique para expandir/recolher",
      context: "Contexto",
      context_badge: "📊 Contexto:",
      decode: "Decode",
      kv_cache: "Cache KV",
      system_ram: "RAM Sistema",
      read: "📖 Lido",
      written: "✍ Escrito",
      total: "∑ Total",
      session_history: "Histórico de Sessões",
      fetching_sessions: "Buscando sessões...",
      empty_sessions: "Nenhuma sessão encontrada.",
      change: "Alterar",
      model: "🤖 Modelo:",
      loading_models: "Carregando modelos...",
      reasoning: "🧠 Raciocínio:",
      effort_medium: "Médio ⚖️",
      effort_low: "Baixo ⚡",
      effort_high: "Alto 🧠",
      effort_off: "Desligado 🚀",
      tools: "🔌 Ferramentas:",
      turbo_opt: "⚡ Modo Turbo (Sem MCPs, ~4.2 t/s)",
      all_opt: "🔌 Todos MCPs (Carregados)",
      configure_btn: "⚙️ Configurar",
      clear_context: "Limpar Contexto",
      input_placeholder: "Digite sua mensagem ou comando (pressione Enter para enviar, Shift+Enter para nova linha)...",
      send: "Enviar",
      stop: "Parar",
      reading: "Leitura",
      writing: "Escrita",
      last: "última",
      session: "sessão",
      hints: "Use <strong>/btw &lt;nota&gt;</strong> para interromper/guiar raciocínio • Troque o modelo no menu superior sem perder o histórico",
      mcp_modal_title: "🔌 Gerenciador de Ferramentas MCP",
      mcp_modal_desc: "Selecione quais servidores MCP deseja ativar. No <strong>Modo Turbo</strong>, todos os MCPs são desativados para reduzir o tamanho do prompt e acelerar a inferência.",
      mcp_btn_turbo: "⚡ Modo Turbo (Desligar Todos)",
      mcp_btn_all: "🔌 Selecionar Todos",
      mcp_btn_save: "Salvar e Aplicar",
      model_online: "Modelo: Online",
      model_offline: "Modelo: Offline",
      model_checking: "Modelo: Verificando...",
      save_success: "✅ Configuração salva!",
      saving: "Salvando...",
      lang_flag: "🇧🇷",
      lang_name: "PT"
    },
    en: {
      app_title: "Apex Harness — Web UI & Model Switcher",
      new_chat: "New Chat (/clear)",
      monitor: "Monitor",
      monitor_toggle: "Click to expand/collapse",
      context: "Context",
      context_badge: "📊 Context:",
      decode: "Decode",
      kv_cache: "KV Cache",
      system_ram: "System RAM",
      read: "📖 Read",
      written: "✍ Written",
      total: "∑ Total",
      session_history: "Session History",
      fetching_sessions: "Fetching sessions...",
      empty_sessions: "No sessions found.",
      change: "Change",
      model: "🤖 Model:",
      loading_models: "Loading models...",
      reasoning: "🧠 Reasoning:",
      effort_medium: "Medium ⚖️",
      effort_low: "Low ⚡",
      effort_high: "High 🧠",
      effort_off: "Off 🚀",
      tools: "🔌 Tools:",
      turbo_opt: "⚡ Turbo Mode (No MCPs, ~4.2 t/s)",
      all_opt: "🔌 All MCPs (Loaded)",
      configure_btn: "⚙️ Configure",
      clear_context: "Clear Context",
      input_placeholder: "Type your message or command (press Enter to send, Shift+Enter for newline)...",
      send: "Send",
      stop: "Stop",
      reading: "Read",
      writing: "Write",
      last: "last",
      session: "session",
      hints: "Use <strong>/btw &lt;note&gt;</strong> to steer reasoning • Switch models from top header without losing history",
      mcp_modal_title: "🔌 MCP Tools Manager",
      mcp_modal_desc: "Select which MCP servers to activate. In <strong>Turbo Mode</strong>, all MCPs are disabled to reduce prompt size and speed up inference.",
      mcp_btn_turbo: "⚡ Turbo Mode (Disable All)",
      mcp_btn_all: "🔌 Select All",
      mcp_btn_save: "Save & Apply",
      model_online: "Model: Online",
      model_offline: "Model: Offline",
      model_checking: "Model: Checking...",
      save_success: "✅ Settings saved!",
      saving: "Saving...",
      lang_flag: "🇺🇸",
      lang_name: "EN"
    }
  };

  let currentLang = localStorage.getItem('apex_lang') || 'pt';

  function applyLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('apex_lang', lang);
    const dict = translations[lang] || translations.pt;

    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (dict[key]) {
        el.innerHTML = dict[key];
      }
    });

    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      const key = el.getAttribute('data-i18n-title');
      if (dict[key]) {
        el.title = dict[key];
      }
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      if (dict[key]) {
        el.placeholder = dict[key];
      }
    });

    const langFlag = document.getElementById('lang-flag');
    const langText = document.getElementById('lang-text');
    if (langFlag) langFlag.textContent = dict.lang_flag;
    if (langText) langText.textContent = dict.lang_name;
  }

  const btnLangToggle = document.getElementById('btn-lang-toggle');
  if (btnLangToggle) {
    btnLangToggle.addEventListener('click', () => {
      const nextLang = currentLang === 'pt' ? 'en' : 'pt';
      applyLanguage(nextLang);
    });
  }

  // Initialize Web UI
  applyLanguage(currentLang);
  loadProject();
  loadModels();
  loadMcp();
  loadSessions();
  loadStatus();
  setInterval(loadStatus, 4000);

  // Watchdog: se isGenerating ficou preso (sem stream ativo há > 90 s), resetar.
  // Isto acontece quando o servidor reinicia, a ligação cai silenciosamente, ou
  // o socket fecha sem mandar o evento `done`. Sem isto, Enter e clique no botão
  // ficam no-ops para sempre (app.js linha 546: `if (isGenerating) abort()`),
  // obrigando o utilizador a recarregar a página.
  setInterval(async () => {
    if (!isGenerating || !generationStartedAt) return;
    const stuck = Date.now() - generationStartedAt > 90_000;
    if (!stuck) return;
    try {
      await fetch('/api/ping', { method: 'GET', signal: AbortSignal.timeout(3000) });
      // Servidor responde — a geração devia ter terminado mas o evento `done` perdeu-se.
      console.warn('[apex] watchdog: isGenerating preso, a resetar.');
      setGeneratingState(false);
    } catch (_) {
      // Servidor offline — não tocar no estado, pode estar a reiniciar.
    }
  }, 10_000);

  // ── Monitor ao vivo: estado EMA e cronómetro ──────────────────────────────
  const _monSmooth = {};
  function _ema(key, value, alpha = 0.25) {
    const v = parseFloat(value) || 0;
    if (!(key in _monSmooth)) { _monSmooth[key] = v; return v; }
    _monSmooth[key] = _monSmooth[key] + alpha * (v - _monSmooth[key]);
    return _monSmooth[key];
  }

  let _monGenStart = 0;
  let _monLastTps  = 0;

  // Colapsar/expandir o painel Monitor
  const monitorBody    = document.getElementById('monitor-body');
  const monitorChevron = document.getElementById('monitor-chevron');
  const monitorToggle  = document.getElementById('monitor-toggle');
  let   monitorOpen    = true;
  if (monitorToggle) {
    monitorToggle.addEventListener('click', () => {
      monitorOpen = !monitorOpen;
      monitorBody.style.display   = monitorOpen ? '' : 'none';
      monitorChevron.textContent  = monitorOpen ? '▲' : '▼';
    });
  }

  function _nFmt(v) {
    const n = Math.max(0, Math.round(parseFloat(v) || 0));
    return n.toLocaleString('pt-PT');
  }

  function atualizarMonitor(data) {
    if (!data) return;
    const status     = data.status || 'offline';
    const ctxTok     = parseInt(data.context_tokens)  || 0;
    const maxCtx     = parseInt(data.max_context)      || 65536;
    const ctxPct     = parseFloat(data.context_pct)   || 0;
    const cachedTok  = parseInt(data.cached_tokens)    || 0;
    const ramPct     = parseFloat(data.ram_pct)        || 0;
    const decodeTps  = _ema('decode', parseFloat(data.decode_tps)  || 0);
    const sessRead   = parseInt(data.session_prompt_tokens)     || 0;
    const sessWrite  = parseInt(data.session_completion_tokens) || 0;

    // Cronómetro de geração
    if (decodeTps > 0.3 && _monLastTps <= 0.3) _monGenStart = Date.now();
    if (decodeTps <= 0.3)                        _monGenStart = 0;
    _monLastTps = decodeTps;
    const elapsedSec = _monGenStart ? ((Date.now() - _monGenStart) / 1000).toFixed(0) : 0;

    // Estado
    const dot    = document.getElementById('mon-dot');
    const stLbl  = document.getElementById('mon-state');
    const elLbl  = document.getElementById('mon-elapsed');
    const pulse  = document.getElementById('monitor-pulse');
    if (dot && stLbl) {
      const isGen  = status === 'ready' && decodeTps > 0.3;
      const isIdle = status === 'ready' && decodeTps <= 0.3;
      dot.className = 'mon-dot-state ' + (isGen ? 'gen' : isIdle ? 'idle' : status === 'loading' ? 'loading' : 'offline');
      stLbl.textContent = isGen ? 'Gerando' : isIdle ? 'Idle' : status === 'loading' ? 'Carregando…' : 'Offline';
      if (elLbl) elLbl.textContent = isGen ? `${elapsedSec}s` : '';
      if (pulse)  pulse.className = 'monitor-pulse' + (isGen ? ' active' : '');
    }

    // Barra de contexto
    const ctxBar = document.getElementById('mon-ctx-bar');
    const ctxPctEl = document.getElementById('mon-ctx-pct');
    const ctxTokEl = document.getElementById('mon-ctx-tokens');
    if (ctxBar)   ctxBar.style.width = Math.min(100, ctxPct) + '%';
    if (ctxPctEl) ctxPctEl.textContent = ctxPct.toFixed(1) + '%';
    if (ctxTokEl) ctxTokEl.textContent = `${_nFmt(ctxTok)} / ${_nFmt(maxCtx)} tok`;
    // Cor da barra por uso
    if (ctxBar) {
      ctxBar.classList.remove('warn', 'danger');
      if (ctxPct > 85) ctxBar.classList.add('danger');
      else if (ctxPct > 65) ctxBar.classList.add('warn');
    }

    // Decode
    const decEl = document.getElementById('mon-decode');
    if (decEl) {
      decEl.textContent = decodeTps > 0.3 ? `${decodeTps.toFixed(1)} t/s` : '— t/s';
      decEl.className = 'mon-metric-val' + (decodeTps > 0.3 ? ' active-val' : '');
    }

    // Cache KV
    const cacheEl = document.getElementById('mon-cache');
    if (cacheEl) {
      const cachePct = ctxTok > 0 && cachedTok > 0 ? Math.round(cachedTok / ctxTok * 100) : 0;
      cacheEl.textContent = cachePct > 0 ? `${cachePct}%` : '—';
      cacheEl.className = 'mon-metric-val' + (cachePct > 0 ? ' cache-hit' : '');
    }

    // RAM
    const ramBar = document.getElementById('mon-ram-bar');
    const ramPctEl = document.getElementById('mon-ram-pct');
    if (ramBar)   { ramBar.style.width = Math.min(100, ramPct) + '%'; }
    if (ramPctEl)  ramPctEl.textContent = ramPct.toFixed(0) + '%';
    if (ramBar) {
      ramBar.classList.remove('warn', 'danger');
      if (ramPct > 85)      ramBar.classList.add('danger');
      else if (ramPct > 70) ramBar.classList.add('warn');
    }

    // Sessão
    const srEl = document.getElementById('mon-sess-read');
    const swEl = document.getElementById('mon-sess-written');
    const stEl = document.getElementById('mon-sess-total');
    if (srEl) srEl.textContent = _nFmt(sessRead);
    if (swEl) swEl.textContent = _nFmt(sessWrite);
    if (stEl) stEl.textContent = _nFmt(sessRead + sessWrite);
  }

  async function loadStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const dot = document.getElementById('model-status-dot');
      const modelText = document.getElementById('model-load-text');
      const ctxText = document.getElementById('context-load-text');

      if (dot && modelText && data.status !== undefined) {
        const isEn = currentLang === 'en';
        if (data.status === 'ready' && data.model_loaded_pct === 100) {
          dot.className = 'dot green';
          modelText.textContent = isEn ? 'Model: 100% Loaded' : 'Modelo: 100% Carregado';
        } else if (data.status === 'loading') {
          dot.className = 'dot yellow';
          modelText.textContent = isEn ? 'Model: Loading...' : 'Modelo: Carregando...';
        } else {
          dot.className = 'dot red';
          modelText.textContent = isEn ? 'Model: Offline (0%)' : 'Modelo: Não Carregado (0%)';
        }
      }
      if (ctxText && data.context_pct !== undefined) {
        ctxText.textContent = `${data.context_pct}% (${data.context_tokens} tok)`;
      }
      atualizarBarraTokens(data);
      atualizarMonitor(data);          // ← alimenta o painel lateral
    } catch (err) {
      console.error('Erro ao carregar status:', err);
    }
  }

  async function loadProject() {
    try {
      const res = await fetch('/api/project');
      const data = await res.json();
      if (data.project_dir) {
        projectPath.textContent = data.folder_name || data.project_dir;
        projectPath.title = data.project_dir;
      }
    } catch (err) {
      console.error('Erro ao carregar pasta:', err);
    }
  }

  btnChangeFolder.addEventListener('click', async () => {
    const current = projectPath.title || '';
    const newDir = prompt('Digite o caminho da nova pasta de trabalho:', current);
    if (!newDir || newDir.trim() === '' || newDir.trim() === current) return;

    try {
      const res = await fetch('/api/project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_dir: newDir.trim() })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        projectPath.textContent = data.folder_name || data.project_dir;
        projectPath.title = data.project_dir;
        appendSystemNotification(`📁 Pasta de trabalho alterada para: <strong>${data.project_dir}</strong>`);
      } else {
        alert('Erro: ' + (data.message || 'Pasta inválida'));
      }
    } catch (err) {
      alert('Erro ao alterar pasta: ' + err.message);
    }
  });

  // Load Models List & Active Model
  async function loadModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      modelSelect.innerHTML = '';

      if (data.available && data.available.length > 0) {
        // O servidor devolve `models` com metadados (label, tamanho, contexto,
        // visao, tool-use) e `available` com as chaves. Usamos os metadados para
        // etiquetas legiveis; antes mostrava-se a chave crua, ilegivel.
        const meta = {};
        (data.models || []).forEach(m => { if (m && m.key) meta[m.key] = m; });

        data.available.forEach(m => {
          const info = meta[m] || {};
          const opt = document.createElement('option');
          opt.value = m;

          let label = info.label || m.split('/').pop() || m;
          const tags = [];
          if (info.size_gb) tags.push(`${Number(info.size_gb).toFixed(1)} GB`);
          if (info.ctx) tags.push(`ctx ${Math.round(info.ctx / 1024)}k`);
          if (info.vision) tags.push('visão');
          if (info.tools) tags.push('tools');
          if (tags.length) label += `  · ${tags.join(' · ')}`;

          opt.textContent = m === data.active ? `⚡ ${label} (Ativo)` : label;
          if (m === data.active) opt.selected = true;
          modelSelect.appendChild(opt);
        });
      } else if (data.active) {
        const opt = document.createElement('option');
        opt.value = data.active;
        opt.textContent = `⚡ ${data.active} (Ativo)`;
        opt.selected = true;
        modelSelect.appendChild(opt);
      }

      if (data.reasoning_effort) {
        effortSelect.value = data.reasoning_effort;
      }
    } catch (err) {
      console.error('Erro ao carregar modelos:', err);
    }
  }

  // Model Hot-Swap Event
  modelSelect.addEventListener('change', async () => {
    const selectedModel = modelSelect.value;
    if (!selectedModel) return;

    const dot = document.getElementById('model-status-dot');
    const modelText = document.getElementById('model-load-text');
    if (dot) dot.className = 'dot yellow';
    if (modelText) modelText.textContent = 'Modelo: Carregando...';

    appendSystemNotification(`⏳ Solicitando carregamento do modelo: <strong>${selectedModel}</strong>...`);

    try {
      const res = await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        appendSystemNotification(`🔄 Modelo ativo alterado para: <strong>${data.active || selectedModel}</strong>`);
        loadModels();
        loadStatus();
      } else {
        appendSystemNotification(`⚠️ ${data.message || 'Falha ao carregar modelo'}`);
        alert(data.message || 'Falha ao carregar modelo');
        loadModels();
        loadStatus();
      }
    } catch (err) {
      appendSystemNotification(`⚠️ Erro de comunicação: ${err.message}`);
      alert('Erro ao trocar modelo: ' + err.message);
      loadStatus();
    }
  });

  // Reasoning Effort Event
  effortSelect.addEventListener('change', async () => {
    const effort = effortSelect.value;
    try {
      const res = await fetch('/api/effort', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ effort })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        appendSystemNotification(`🧠 Raciocínio (effort) ajustado para: <strong>${effort.toUpperCase()}</strong>`);
      }
    } catch (err) {
      alert('Erro ao ajustar effort: ' + err.message);
    }
  });

  // Clear Context & New Chat Events
  btnClear.addEventListener('click', () => startNewSession(true));
  btnNewChat.addEventListener('click', () => startNewSession(true));

  let cachedMcpData = null;

  async function loadMcp() {
    if (!mcpSelect) return;
    try {
      const res = await fetch('/api/mcp');
      const data = await res.json();
      cachedMcpData = data;
      mcpSelect.innerHTML = '';

      const optTurbo = document.createElement('option');
      optTurbo.value = 'turbo';
      optTurbo.textContent = '⚡ Modo Turbo (Sem MCPs, ~4.2 t/s)';
      mcpSelect.appendChild(optTurbo);

      const optAll = document.createElement('option');
      optAll.value = 'all';
      const count = (data.available_servers || []).length;
      optAll.textContent = `🔌 Todos MCPs (${count} servidores)`;
      mcpSelect.appendChild(optAll);

      if (data.available_servers && data.available_servers.length > 0) {
        const group = document.createElement('optgroup');
        group.label = 'Servidor específico';
        data.available_servers.forEach(s => {
          const opt = document.createElement('option');
          opt.value = `server:${s}`;
          opt.textContent = `📦 Só ${s}`;
          group.appendChild(opt);
        });
        mcpSelect.appendChild(group);
      }

      if (data.mode === 'turbo') {
        mcpSelect.value = 'turbo';
      } else if (data.mode === 'all') {
        mcpSelect.value = 'all';
      } else if (data.enabled_servers && data.enabled_servers.length === 1) {
        mcpSelect.value = `server:${data.enabled_servers[0]}`;
      } else {
        mcpSelect.value = 'all';
      }

      renderMcpChecklist(data);
    } catch (err) {
      console.warn('Erro ao carregar MCPs:', err);
    }
  }

  function renderMcpChecklist(data) {
    const list = document.getElementById('mcp-servers-checklist');
    if (!list || !data || !data.available_servers) return;
    list.innerHTML = '';

    const enabledSet = new Set(data.mode === 'turbo' ? [] : (data.enabled_servers || []));

    data.available_servers.forEach(serverName => {
      const item = document.createElement('label');
      item.className = 'mcp-item';

      const left = document.createElement('div');
      left.className = 'mcp-item-left';

      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.value = serverName;
      chk.checked = enabledSet.has(serverName);

      const label = document.createElement('span');
      label.className = 'mcp-item-name';
      label.textContent = serverName;

      left.appendChild(chk);
      left.appendChild(label);

      const badge = document.createElement('span');
      badge.className = 'mcp-item-badge';
      badge.textContent = 'MCP Server';

      item.appendChild(left);
      item.appendChild(badge);
      list.appendChild(item);
    });
  }

  const btnOpenMcpModal = document.getElementById('btn-open-mcp-modal');
  const btnCloseMcpModal = document.getElementById('btn-close-mcp-modal');
  const mcpModal = document.getElementById('mcp-modal');
  const btnMcpTurbo = document.getElementById('btn-mcp-turbo');
  const btnMcpAll = document.getElementById('btn-mcp-all');
  const btnSaveMcp = document.getElementById('btn-save-mcp');
  const mcpModalStatus = document.getElementById('mcp-modal-status');

  if (btnOpenMcpModal && mcpModal) {
    btnOpenMcpModal.addEventListener('click', () => {
      if (cachedMcpData) renderMcpChecklist(cachedMcpData);
      mcpModal.style.display = 'flex';
      if (mcpModalStatus) mcpModalStatus.textContent = '';
    });
  }

  if (btnCloseMcpModal && mcpModal) {
    btnCloseMcpModal.addEventListener('click', () => {
      mcpModal.style.display = 'none';
    });
  }

  if (mcpModal) {
    mcpModal.addEventListener('click', (e) => {
      if (e.target === mcpModal) {
        mcpModal.style.display = 'none';
      }
    });
  }

  if (btnMcpTurbo) {
    btnMcpTurbo.addEventListener('click', () => {
      const chks = document.querySelectorAll('#mcp-servers-checklist input[type="checkbox"]');
      chks.forEach(c => c.checked = false);
    });
  }

  if (btnMcpAll) {
    btnMcpAll.addEventListener('click', () => {
      const chks = document.querySelectorAll('#mcp-servers-checklist input[type="checkbox"]');
      chks.forEach(c => c.checked = true);
    });
  }

  if (btnSaveMcp) {
    btnSaveMcp.addEventListener('click', async () => {
      const chks = Array.from(document.querySelectorAll('#mcp-servers-checklist input[type="checkbox"]'));
      const checkedServers = chks.filter(c => c.checked).map(c => c.value);

      let payload = {};
      if (checkedServers.length === 0) {
        payload = { mode: 'turbo' };
      } else if (checkedServers.length === chks.length) {
        payload = { mode: 'all' };
      } else {
        payload = { mode: 'custom', servers: checkedServers };
      }

      if (mcpModalStatus) mcpModalStatus.textContent = 'Salvando...';

      try {
        const res = await fetch('/api/mcp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await res.json();
        if (mcpModalStatus) mcpModalStatus.textContent = '✅ Configuração salva!';
        await loadMcp();
        setTimeout(() => {
          if (mcpModal) mcpModal.style.display = 'none';
        }, 600);
      } catch (err) {
        if (mcpModalStatus) mcpModalStatus.textContent = '❌ Erro: ' + err.message;
      }
    });
  }

  if (mcpSelect) {
    mcpSelect.addEventListener('change', async () => {
      const val = mcpSelect.value;
      let payload = {};
      if (val === 'turbo') {
        payload = { mode: 'turbo' };
      } else if (val === 'all') {
        payload = { mode: 'all' };
      } else if (val.startsWith('server:')) {
        const serverName = val.substring(7);
        payload = { mode: 'custom', servers: [serverName] };
      }
      try {
        const res = await fetch('/api/mcp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await res.json();
        if (d.message) {
          console.log('[MCP]', d.message);
        }
        await loadMcp();
      } catch (err) {
        alert('Erro ao alterar MCP: ' + err.message);
      }
    });
  }

  function formatSessionDate(isoStr) {
    if (!isoStr) return '';
    try {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return isoStr;
      const hoje = new Date();
      if (d.toDateString() === hoje.toDateString()) {
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      }
      return d.toLocaleDateString([], { day: '2-digit', month: '2-digit' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return isoStr;
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Load Session History
  async function loadSessions() {
    try {
      const res = await fetch('/api/sessions');
      const data = await res.json();
      sessionsList.innerHTML = '';
      if (data.sessions && data.sessions.length > 0) {
        data.sessions.forEach(s => {
          const item = document.createElement('div');
          item.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
          item.dataset.id = s.id;

          const title = s.summary || `Sessão ${s.id.substring(5, 13)}`;
          const dateStr = s.last_active ? formatSessionDate(s.last_active) : (s.started_at ? formatSessionDate(s.started_at) : '');

          item.innerHTML = `
            <div class="session-info" title="${escapeHtml(title)}">
              <span class="session-title">${escapeHtml(title)}</span>
              <div class="session-meta">
                <span>${dateStr}</span>
                ${s.message_count ? `<span>· ${s.message_count} msgs</span>` : ''}
              </div>
            </div>
            <button class="btn-delete-session" title="Apagar sessão permanentemente" data-id="${s.id}">🗑️</button>
          `;

          item.querySelector('.session-info').addEventListener('click', () => {
            selectSession(s.id);
          });

          item.querySelector('.btn-delete-session').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSession(s.id);
          });

          sessionsList.appendChild(item);
        });

        // Se currentSessionId não existe na lista, seleciona a primeira da lista
        if (!currentSessionId || !data.sessions.some(s => s.id === currentSessionId)) {
          const initialId = (data.current_session_id && data.sessions.some(s => s.id === data.current_session_id))
            ? data.current_session_id
            : data.sessions[0].id;
          selectSession(initialId);
        }
      } else {
        sessionsList.innerHTML = '<div class="empty-sessions" style="font-size:0.8rem; color:#94a3b8">Nenhuma sessão salva no banco ou disco (~/.apex_sessions/sessions/).</div>';
        if (!currentSessionId) {
          startNewSession(false);
        }
      }
    } catch (err) {
      sessionsList.innerHTML = '<div class="empty-sessions" style="font-size:0.8rem; color:#ef4444">Erro ao carregar sessões.</div>';
    }
  }

  async function selectSession(sessionId) {
    if (!sessionId) return;
    currentSessionId = sessionId;
    localStorage.setItem('apex_active_session_id', sessionId);

    document.querySelectorAll('.session-item').forEach(el => {
      el.classList.toggle('active', el.dataset.id === sessionId);
    });

    try {
      const res = await fetch('/api/session/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: sessionId })
      });
      const data = await res.json();
      if (data.session) {
        renderSessionMessages(data.session.messages || []);
      }
    } catch (err) {
      console.warn('Erro ao carregar mensagens da sessão:', err);
    }
  }

  function renderSessionMessages(messages) {
    chatStream.innerHTML = '';
    if (!messages || messages.length === 0) {
      chatStream.innerHTML = `
        <div class="welcome-card">
          <h2>⚡ Apex Harness Web Interface</h2>
          <p>Sessão vazia. Digite sua mensagem ou comando abaixo para começar.</p>
          <div class="feature-tags">
            <span class="tag">💡 Interrupção /btw</span>
            <span class="tag">🔄 Model Hot-Swap</span>
            <span class="tag">🧠 Reasoning Traces</span>
            <span class="tag">💾 Session Memory</span>
          </div>
        </div>
      `;
      return;
    }

    messages.forEach(m => {
      if (m.role === 'user') {
        const isBtw = (m.content || '').startsWith('[BY-THE-WAY');
        const text = isBtw ? m.content.replace(/^\[BY-THE-WAY[^\]]*\]:\s*/, '') : m.content;
        appendUserMessage(text, isBtw);
      } else if (m.role === 'assistant') {
        appendHistoricalAssistantMessage(m.content);
      }
    });

    chatStream.scrollTop = chatStream.scrollHeight;
  }

  function appendHistoricalAssistantMessage(fullText) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';

    const roleDiv = document.createElement('div');
    roleDiv.className = 'message-role';
    roleDiv.innerHTML = '🤖 Apex Harness';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';

    let textToRender = fullText || '';
    const thinkMatch = textToRender.match(/<think>([\s\S]*?)<\/think>/);
    if (thinkMatch) {
      const thinkText = thinkMatch[1].trim();
      textToRender = textToRender.replace(/<think>[\s\S]*?<\/think>/, '').trim();

      const details = document.createElement('details');
      details.className = 'think-block';
      details.open = false;
      const summary = document.createElement('summary');
      summary.textContent = '🧠 Pensamento e Raciocínio (Trace)';
      details.appendChild(summary);

      const thinkContent = document.createElement('div');
      thinkContent.className = 'think-content';
      thinkContent.textContent = thinkText;
      details.appendChild(thinkContent);
      bubbleDiv.appendChild(details);
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'assistant-text';
    textDiv.innerHTML = window.marked ? marked.parse(textToRender) : textToRender;
    bubbleDiv.appendChild(textDiv);

    msgDiv.appendChild(roleDiv);
    msgDiv.appendChild(bubbleDiv);
    chatStream.appendChild(msgDiv);
  }

  async function startNewSession(refreshUI = true) {
    if (isGenerating) {
      if (!confirm('Uma resposta está a ser gerada. Deseja iniciar uma nova conversa mesmo assim?')) return;
      if (currentAbortController) currentAbortController.abort();
      setGeneratingState(false);
    }
    try {
      const res = await fetch('/api/session/new', { method: 'POST' });
      const data = await res.json();
      currentSessionId = data.session_id;
      localStorage.setItem('apex_active_session_id', currentSessionId);
      renderSessionMessages([]);
      if (refreshUI) {
        loadSessions();
      }
      userInput.focus();
    } catch (err) {
      alert('Erro ao criar nova sessão: ' + err.message);
    }
  }

  async function deleteSession(sessionId) {
    if (!confirm('Deseja apagar esta sessão permanentemente do disco e da memória?')) return;
    try {
      const res = await fetch('/api/session/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: sessionId })
      });
      const data = await res.json();
      if (sessionId === currentSessionId) {
        currentSessionId = data.active_session_id || '';
        localStorage.setItem('apex_active_session_id', currentSessionId);
        selectSession(currentSessionId);
      }
      loadSessions();
    } catch (err) {
      alert('Erro ao apagar sessão: ' + err.message);
    }
  }

  // (Removido botão fixo /btw: comando agora é utilizado diretamente no input)

  async function sendBtwNote(note) {
    if (!note) return;
    try {
      const res = await fetch('/api/btw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ note })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        appendUserMessage(note, true);
        loadStatus();
      } else {
        alert(data.message || 'Erro ao enviar /btw');
      }
    } catch (err) {
      alert('Erro ao enviar /btw: ' + err.message);
    }
  }

  // User Send Message Event
  btnSend.addEventListener('click', sendMessage);

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ------------------------------------------------------------------ caixa -----
  // A caixa cresce com o texto. Sem isto ficava presa em `rows="1"` (uma linha) e
  // nao se conseguia reler o que estava escrito antes de enviar.
  const ALTURA_MAX = () => Math.round(window.innerHeight * 0.4);

  function ajustarAlturaCaixa() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, ALTURA_MAX()) + 'px';
  }

  userInput.addEventListener('input', () => {
    ajustarAlturaCaixa();
    agendarContagemMensagem();
  });
  // Colar um bloco, ou apagar tudo, nao dispara sempre `input` da mesma forma.
  userInput.addEventListener('paste', () => setTimeout(() => { ajustarAlturaCaixa(); agendarContagemMensagem(); }, 0));
  window.addEventListener('resize', ajustarAlturaCaixa);

  // ------------------------------------------------------- contagem de tokens ---
  // Leitura/escrita, na barra de baixo. Os numeros do modelo vem do `usage` que o
  // llama-server manda no fim de cada pedido; a contagem do que se esta a
  // escrever vem do tokenizer real do modelo (/api/tokenize), com debounce para
  // nao disparar um pedido por tecla.
  const tokEls = {
    ctx: document.getElementById('tok-read-ctx'),
    msg: document.getElementById('tok-read-msg'),
    last: document.getElementById('tok-write-last'),
    session: document.getElementById('tok-write-session'),
    ctxPct: document.getElementById('tok-ctx'),
    speed: document.getElementById('tok-speed'),
    cache: document.getElementById('tok-cache'),
  };

  const nf = new Intl.NumberFormat('pt-PT');

  function porNumero(el, valor) {
    if (!el) return;
    el.textContent = nf.format(Math.max(0, Math.round(valor || 0)));
  }

  let timerMsg = null;
  let pedidoMsg = 0;   // numero de sequencia, para ignorar respostas fora de ordem

  function agendarContagemMensagem() {
    clearTimeout(timerMsg);
    const texto = userInput.value;
    if (!texto.trim()) {
      porNumero(tokEls.msg, 0);
      return;
    }
    timerMsg = setTimeout(() => contarMensagem(texto), 250);
  }

  async function contarMensagem(texto) {
    const seq = ++pedidoMsg;
    try {
      const res = await fetch('/api/tokenize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: texto })
      });
      const data = await res.json();
      // Se entretanto ja se escreveu mais, esta resposta e velha: ignora-a.
      if (seq !== pedidoMsg) return;
      porNumero(tokEls.msg, data.tokens);
    } catch (e) {
      // Sem backend para tokenizar: cai para a estimativa grosseira.
      porNumero(tokEls.msg, Math.ceil(texto.length / 4));
    }
  }

  function atualizarBarraTokens(d) {
    if (!d) return;
    porNumero(tokEls.ctx, d.context_tokens);
    porNumero(tokEls.last, d.completion_tokens);
    porNumero(tokEls.session, d.session_completion_tokens);

    if (tokEls.ctxPct) {
      const pct = d.context_pct != null ? d.context_pct : 0;
      tokEls.ctxPct.textContent = `contexto ${pct}% de ${nf.format(d.max_context || 0)}`;
    }
    if (tokEls.speed) {
      const dec = Number(d.decode_tps || 0);
      tokEls.speed.textContent = dec > 0 ? `${dec.toFixed(1)} t/s` : '— t/s';
    }
    if (tokEls.cache) {
      const cached = Number(d.cached_tokens || 0);
      const prompt = Number(d.context_tokens || 0);
      tokEls.cache.textContent = (cached > 0 && prompt > 0)
        ? `♻ ${Math.round((cached / prompt) * 100)}% do prompt em cache`
        : '♻ sem cache';
    }
  }

  // ------------------------------------------------------------------ imagens ---
  // Anexar, colar (Ctrl+V) e arrastar. As imagens so podem ser enviadas se o
  // backend tiver projetor multimodal -- o /props do llama-server diz isso em
  // `modalities.vision`, e o /api/vision do harness passa-o adiante.
  //
  // Sem esta verificacao o botao existiria e falharia a seguir, com um erro de
  // servidor incompreensivel a meio do stream.
  const btnAttach = document.getElementById('btn-attach');
  const fileInput = document.getElementById('file-input');
  const imageStrip = document.getElementById('image-strip');
  const visionWarning = document.getElementById('vision-warning');
  const inputFooter = document.querySelector('.input-footer');

  let pendingImages = [];
  let visionOk = false;

  const MAX_LADO = 1280;          // px no lado maior
  const MAX_DATAURL = 1400000;    // ~1 MB de imagem ja codificada

  async function carregarVisao() {
    try {
      const res = await fetch('/api/vision');
      const d = await res.json();
      visionOk = !!d.vision;
    } catch (e) {
      visionOk = false;
    }
    if (btnAttach) {
      btnAttach.disabled = !visionOk;
      btnAttach.title = visionOk
        ? 'Anexar imagem (ou colar com Ctrl+V, ou arrastar para aqui)'
        : 'Modelo texto apenas — sem suporte a imagens';
    }
    // A mensagem no footer é redundante: o botão desativado já comunica isso.
    // Manter sempre oculta para não poluir a interface.
    if (visionWarning) visionWarning.hidden = true;
    if (!visionOk) limparImagens();
  }

  function limparImagens() {
    pendingImages = [];
    renderImagens();
  }

  function renderImagens() {
    if (!imageStrip) return;
    imageStrip.innerHTML = '';
    imageStrip.hidden = pendingImages.length === 0;
    pendingImages.forEach((img, i) => {
      const div = document.createElement('div');
      div.className = 'image-thumb';
      const el = document.createElement('img');
      el.src = img.url;
      el.alt = img.name || `imagem ${i + 1}`;
      const x = document.createElement('button');
      x.type = 'button';
      x.textContent = '×';
      x.title = 'Remover';
      x.addEventListener('click', () => {
        pendingImages.splice(i, 1);
        renderImagens();
      });
      div.appendChild(el);
      div.appendChild(x);
      imageStrip.appendChild(div);
    });
  }

  function lerFicheiro(file) {
    return new Promise((resolve) => {
      if (!file || !file.type || !file.type.startsWith('image/')) return resolve(null);
      const reader = new FileReader();
      reader.onerror = () => resolve(null);
      reader.onload = () => {
        const dataUrl = String(reader.result || '');
        const img = new Image();
        img.onerror = () => resolve(null);
        img.onload = () => {
          const maior = Math.max(img.width, img.height);
          // Ja e pequena: manda-se como esta, sem recompressao.
          if (maior <= MAX_LADO && dataUrl.length <= MAX_DATAURL) {
            return resolve({ url: dataUrl, name: file.name });
          }
          // Reduzir no cliente e obrigatorio: uma foto de telemovel tem varios
          // MB, e em base64 isso e um corpo de pedido enorme e muitos tokens de
          // visao. 1280 px chega para o modelo perceber a imagem.
          const escala = MAX_LADO / maior;
          const cv = document.createElement('canvas');
          cv.width = Math.max(1, Math.round(img.width * escala));
          cv.height = Math.max(1, Math.round(img.height * escala));
          cv.getContext('2d').drawImage(img, 0, 0, cv.width, cv.height);
          // PNG pequeno fica PNG (texto de captura de ecra fica nitido); o resto
          // vai a JPEG, que e muito mais leve.
          const manterPng = /png/i.test(file.type) && dataUrl.length < 900000;
          resolve({
            url: cv.toDataURL(manterPng ? 'image/png' : 'image/jpeg', 0.9),
            name: file.name,
          });
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    });
  }

  async function juntarFicheiros(files) {
    if (!visionOk || !files || !files.length) return;
    for (const f of Array.from(files).slice(0, 6 - pendingImages.length)) {
      const img = await lerFicheiro(f);
      if (img && pendingImages.length < 6) pendingImages.push(img);
    }
    renderImagens();
  }

  if (btnAttach && fileInput) {
    btnAttach.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
      juntarFicheiros(fileInput.files);
      fileInput.value = '';
    });
  }

  // ── Anexo de ficheiros de texto (📎) ──────────────────────────────────────
  // Lê o conteúdo no browser e injeta como bloco de código no prompt.
  // Funciona com QUALQUER modelo de linguagem — não precisa de visão.
  const btnAttachFile  = document.getElementById('btn-attach-file');
  const fileTextInput  = document.getElementById('file-text-input');
  const fileStrip      = document.getElementById('file-strip');
  let   pendingTextFiles = [];   // [{name, ext, content}]

  const MAX_FILE_BYTES = 500_000;   // 500 KB por ficheiro — suficiente para a maioria dos casos

  function extOf(name) {
    const m = name.match(/\.([^.]+)$/);
    return m ? m[1].toLowerCase() : 'txt';
  }

  // Detectar linguagem para o bloco de código (melhora o syntax highlight do modelo)
  const EXT_LANG = {
    py:'python', js:'javascript', ts:'typescript', jsx:'javascript', tsx:'typescript',
    html:'html', css:'css', json:'json', jsonl:'json', yaml:'yaml', yml:'yaml',
    toml:'toml', sh:'bash', bash:'bash', zsh:'bash', sql:'sql',
    rs:'rust', go:'go', java:'java', c:'c', cpp:'cpp', h:'c', hpp:'cpp',
    rb:'ruby', php:'php', r:'r', tex:'latex', rst:'rst', xml:'xml',
    md:'markdown', markdown:'markdown', csv:'csv', tsv:'tsv',
    env:'bash', ini:'ini', cfg:'ini', conf:'ini', log:'', txt:'',
  };

  function langFor(ext) { return EXT_LANG[ext] ?? ''; }

  async function lerFicheiroTexto(file) {
    return new Promise((resolve) => {
      const ext = extOf(file.name);
      const reader = new FileReader();
      reader.onerror = () => resolve(null);
      reader.onload = () => {
        let text = String(reader.result || '');
        let truncated = false;
        if (text.length > MAX_FILE_BYTES) {
          text = text.slice(0, MAX_FILE_BYTES);
          truncated = true;
        }
        resolve({ name: file.name, ext, text, truncated, size: file.size });
      };
      reader.readAsText(file, 'utf-8');
    });
  }

  function renderFileStrip() {
    if (!fileStrip) return;
    fileStrip.innerHTML = '';
    fileStrip.hidden = pendingTextFiles.length === 0;
    pendingTextFiles.forEach((f, i) => {
      const badge = document.createElement('div');
      badge.className = 'file-badge';
      const kb = (f.size / 1024).toFixed(1);
      badge.innerHTML = `
        <span class="file-badge-icon">📄</span>
        <span class="file-badge-name" title="${f.name}">${f.name}</span>
        <span class="file-badge-size">${kb} KB</span>
        <button class="file-badge-remove" title="Remover">×</button>
      `;
      badge.querySelector('.file-badge-remove').addEventListener('click', () => {
        pendingTextFiles.splice(i, 1);
        renderFileStrip();
      });
      fileStrip.appendChild(badge);
    });
  }

  async function juntarFicheirosTexto(files) {
    if (!files || !files.length) return;
    for (const f of Array.from(files).slice(0, 8 - pendingTextFiles.length)) {
      if (pendingTextFiles.length >= 8) break;
      const result = await lerFicheiroTexto(f);
      if (result) pendingTextFiles.push(result);
    }
    renderFileStrip();
  }

  if (btnAttachFile && fileTextInput) {
    btnAttachFile.addEventListener('click', () => fileTextInput.click());
    fileTextInput.addEventListener('change', () => {
      juntarFicheirosTexto(fileTextInput.files);
      fileTextInput.value = '';
    });
  }

  function buildFileBlock(f) {
    const lang = langFor(f.ext);
    const header = `\n\n--- Ficheiro: ${f.name} ---`;
    const notice = f.truncated ? `\n[⚠️ Truncado a 500 KB — ficheiro original: ${(f.size/1024).toFixed(0)} KB]` : '';
    return `${header}${notice}\n\`\`\`${lang}\n${f.text}\n\`\`\``;
  }



  userInput.addEventListener('paste', (e) => {
    const itens = e.clipboardData && e.clipboardData.items;
    if (!itens) return;
    const ficheiros = [];
    for (const it of itens) {
      if (it.kind === 'file' && it.type.startsWith('image/')) {
        const f = it.getAsFile();
        if (f) ficheiros.push(f);
      }
    }
    if (ficheiros.length) {
      e.preventDefault();      // nao colar o caminho/nome como texto
      juntarFicheiros(ficheiros);
    }
  });

  if (inputFooter) {
    ['dragenter', 'dragover'].forEach((ev) =>
      inputFooter.addEventListener(ev, (e) => {
        if (!visionOk) return;
        e.preventDefault();
        inputFooter.classList.add('drag-over');
      }));
    ['dragleave', 'drop'].forEach((ev) =>
      inputFooter.addEventListener(ev, (e) => {
        e.preventDefault();
        inputFooter.classList.remove('drag-over');
      }));
    inputFooter.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files) juntarFicheiros(e.dataTransfer.files);
    });
  }

  carregarVisao();

  async function sendMessage() {
    const message = userInput.value.trim();

    // Check if user typed /btw (allowed even during generation)
    if (message.startsWith('/btw ') || message.startsWith('/btw')) {
      const note = message.replace(/^\/btw\s*/, '').trim();
      userInput.value = '';
      ajustarAlturaCaixa();
      if (!note) {
        alert('Digite uma nota para o comando /btw');
        return;
      }
      sendBtwNote(note);
      return;
    }

    // Construir mensagem final: injetar conteúdo dos ficheiros de texto antes da mensagem do utilizador
    let filesBlock = '';
    if (pendingTextFiles.length > 0) {
      filesBlock = pendingTextFiles.map(buildFileBlock).join('');
    }
    const finalMessage = filesBlock ? (filesBlock + (message ? '\n\n' + message : '')) : message;

    // Uma imagem sem texto é uma pergunta legítima ("o que é isto?").
    if (!finalMessage && !pendingImages.length) return;

    // As imagens só vão no primeiro envio
    const imagensAEnviar = pendingImages.map((i) => i.url);
    const filesAnexados = pendingTextFiles.slice();

    userInput.value = '';
    ajustarAlturaCaixa();
    porNumero(tokEls.msg, 0);
    limparImagens();
    pendingTextFiles = [];
    renderFileStrip();

    // Label da mensagem do utilizador: resumo dos ficheiros + texto
    const labelMsg = [
      filesAnexados.length ? `[${filesAnexados.length} ficheiro(s): ${filesAnexados.map(f => f.name).join(', ')}]` : '',
      message,
    ].filter(Boolean).join('  ') || (imagensAEnviar.length ? '[imagem]' : '');

    // Se já estiver gerando, coloca NA FILA! Não cancela a resposta anterior!
    if (isGenerating) {
      messageQueue.push({
        finalMessage,
        imagensAEnviar,
        filesAnexados,
        labelMsg
      });
      updateQueueBanner();
      return;
    }

    await dispatchMessage({ finalMessage, imagensAEnviar, filesAnexados, labelMsg });
  }

  async function dispatchMessage({ finalMessage, imagensAEnviar, filesAnexados, labelMsg }) {
    // Remover welcome card se presente
    const welcomeCard = chatStream.querySelector('.welcome-card');
    if (welcomeCard) welcomeCard.remove();

    appendUserMessage(labelMsg, false);

    // Inicializa AbortController para permitir cancelamento manual
    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;
    setGeneratingState(true);

    // Create Assistant response containers
    const { assistantMsgElem, textContentElem, getOrCreateThinkBlock, finishStatusPill } = createAssistantContainer();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: signal,
        body: JSON.stringify({
          message: finalMessage,
          images: imagensAEnviar && imagensAEnviar.length ? imagensAEnviar : undefined,
          session_id: currentSessionId || undefined
        })
      });

      if (!response.ok) {
        let errMsg = `HTTP ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.error) errMsg = errData.error;
          else if (errData && errData.message) errMsg = errData.message;
        } catch (e) {}
        textContentElem.textContent += `\n⚠️ Erro do servidor: ${errMsg}`;
        finishStatusPill();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let terminado = false;

      const processBlock = (block) => {
        const cleanBlock = block.trim();
        if (!cleanBlock.startsWith('data: ')) return;
        const jsonStr = cleanBlock.substring(6);
        try {
          const eventData = JSON.parse(jsonStr);
          if (eventData.type === 'session_id' && eventData.content) {
            currentSessionId = eventData.content;
            localStorage.setItem('apex_active_session_id', currentSessionId);
          }
          handleSseEvent(eventData, textContentElem, getOrCreateThinkBlock, finishStatusPill);
          if (eventData.type === 'done' || eventData.type === 'error') {
            terminado = true;
          }
        } catch (e) {
          console.error('Erro no SSE Event:', e);
        }
      };

      while (!terminado) {
        const { value, done } = await reader.read();
        if (done) {
          // Processa qualquer dado restante no buffer após EOF
          if (buffer.trim()) {
            processBlock(buffer);
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Mantém a última linha incompleta no buffer

        for (const block of lines) {
          processBlock(block);
          if (terminado) break;
        }
      }

      try { await reader.cancel(); } catch (e) { /* ja fechado */ }
    } catch (err) {
      if (err.name === 'AbortError') {
        textContentElem.textContent += '\n⏹ Geração interrompida pelo utilizador.';
      } else {
        textContentElem.textContent += `\n⚠️ Erro na conexão: ${err.message}`;
      }
      finishStatusPill();
    } finally {
      finishStatusPill();
      setGeneratingState(false);
      loadStatus();
      loadSessions();

      // Processar próxima mensagem na fila, se houver
      if (messageQueue.length > 0) {
        const next = messageQueue.shift();
        updateQueueBanner();
        setTimeout(() => dispatchMessage(next), 120);
      }
    }
  }

  function handleSseEvent(eventData, textElem, getThinkBlock, finishStatusPill) {
    if (eventData.type === 'think_start') {
      getThinkBlock();
    } else if (eventData.type === 'think_chunk') {
      const { thinkContentElem } = getThinkBlock();
      thinkContentElem.textContent += eventData.content;
    } else if (eventData.type === 'think_end') {
      // Done thinking
    } else if (eventData.type === 'text_chunk') {
      finishStatusPill();
      textElem.textContent += eventData.content;
      chatStream.scrollTop = chatStream.scrollHeight;
    } else if (eventData.type === 'done') {
      finishStatusPill();
      if (!textElem.textContent.trim() && eventData.content) {
        textElem.textContent = eventData.content;
      }
      chatStream.scrollTop = chatStream.scrollHeight;
    } else if (eventData.type === 'error') {
      finishStatusPill();
      // A mensagem do servidor ja costuma comecar por ⚠️; sem isto aparecia
      // "⚠️ Erro: ⚠️ O modelo...".
      const txt = String(eventData.content || '');
      textElem.textContent += txt.trimStart().startsWith('⚠') ? `\n${txt}` : `\n⚠️ Erro: ${txt}`;
      chatStream.scrollTop = chatStream.scrollHeight;
    } else if (eventData.type === 'usage') {
      // O servidor manda a contagem como string JSON dentro de `content` (o
      // campo que o cliente SSE le sempre). Ver server.py:_handle_post_chat.
      try {
        const u = JSON.parse(eventData.content);
        const d = {
          context_tokens: u.prompt_tokens,
          completion_tokens: u.completion_tokens,
          cached_tokens: u.cached_tokens,
          decode_tps: u.decode_tps,
          session_completion_tokens: u.session_completion_tokens,
        };
        porNumero(tokEls.ctx, d.context_tokens);
        porNumero(tokEls.last, d.completion_tokens);
        porNumero(tokEls.session, d.session_completion_tokens);
        if (tokEls.speed) {
          tokEls.speed.textContent = d.decode_tps > 0 ? `${Number(d.decode_tps).toFixed(1)} t/s` : '— t/s';
        }
        if (tokEls.cache) {
          tokEls.cache.textContent = (d.cached_tokens > 0 && d.context_tokens > 0)
            ? `♻ ${Math.round((d.cached_tokens / d.context_tokens) * 100)}% do prompt em cache`
            : '♻ sem cache';
        }
      } catch (e) {
        console.error('usage invalido:', e);
      }
    }
  }

  function appendUserMessage(text, isBtw = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message user ${isBtw ? 'btw-note' : ''}`;

    const roleDiv = document.createElement('div');
    roleDiv.className = 'message-role';
    roleDiv.innerHTML = isBtw ? '💡 Você (/btw Nota Lateral)' : '👤 Você';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    bubbleDiv.textContent = text;

    msgDiv.appendChild(roleDiv);
    msgDiv.appendChild(bubbleDiv);
    chatStream.appendChild(msgDiv);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  const CLAUDE_PHRASES = [
    "Pensando...",
    "Refletindo sobre a resposta...",
    "Examinando arquivos e contexto...",
    "Analisando restrições e código...",
    "Sintetizando arquitetura da solução...",
    "Consultando ferramentas...",
    "Elaborando raciocínio..."
  ];

  function createAssistantContainer() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';

    const roleDiv = document.createElement('div');
    roleDiv.className = 'message-role';
    roleDiv.innerHTML = '🤖 Apex Harness';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';

    // Claude Code activity status pill
    const statusPill = document.createElement('div');
    statusPill.className = 'claude-status-pill';
    statusPill.innerHTML = `
      <span class="claude-spinner"></span>
      <span class="claude-phrase">${CLAUDE_PHRASES[0]}</span>
      <span class="claude-timer">(0.0s)</span>
    `;
    bubbleDiv.appendChild(statusPill);

    const startTime = Date.now();
    let hasFinishedPill = false;
    const intervalId = setInterval(() => {
      if (hasFinishedPill) return;
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const timerSpan = statusPill.querySelector('.claude-timer');
      const phraseSpan = statusPill.querySelector('.claude-phrase');
      if (timerSpan) timerSpan.textContent = `(${elapsed}s)`;
      if (phraseSpan) {
        const idx = Math.floor(elapsed / 1.8) % CLAUDE_PHRASES.length;
        phraseSpan.textContent = CLAUDE_PHRASES[idx];
      }
    }, 200);

    function finishStatusPill() {
      if (hasFinishedPill) return;
      hasFinishedPill = true;
      clearInterval(intervalId);
      const totalSec = ((Date.now() - startTime) / 1000).toFixed(1);
      statusPill.innerHTML = `<span>🧠 Pensou por ${totalSec}s</span>`;
      statusPill.style.animation = 'none';
      statusPill.style.background = 'rgba(168, 85, 247, 0.08)';
      statusPill.style.borderColor = 'rgba(168, 85, 247, 0.2)';
      statusPill.style.color = '#c084fc';
    }

    const textContentElem = document.createElement('div');
    textContentElem.className = 'assistant-text';
    bubbleDiv.appendChild(textContentElem);

    msgDiv.appendChild(roleDiv);
    msgDiv.appendChild(bubbleDiv);
    chatStream.appendChild(msgDiv);

    let thinkDetailsElem = null;
    let thinkContentElem = null;

    function getOrCreateThinkBlock() {
      if (!thinkDetailsElem) {
        thinkDetailsElem = document.createElement('details');
        thinkDetailsElem.className = 'think-block';
        thinkDetailsElem.open = true;

        const summaryElem = document.createElement('summary');
        summaryElem.textContent = '🧠 Pensamento e Raciocínio (Trace)';
        thinkDetailsElem.appendChild(summaryElem);

        thinkContentElem = document.createElement('div');
        thinkContentElem.className = 'think-content';
        thinkDetailsElem.appendChild(thinkContentElem);

        bubbleDiv.insertBefore(thinkDetailsElem, textContentElem);
      }
      return { thinkDetailsElem, thinkContentElem };
    }

    return { assistantMsgElem: msgDiv, textContentElem, getOrCreateThinkBlock, finishStatusPill };
  }

  function appendSystemNotification(htmlText) {
    const notifDiv = document.createElement('div');
    notifDiv.className = 'system-notification';
    notifDiv.style.textAlign = 'center';
    notifDiv.style.fontSize = '0.8rem';
    notifDiv.style.color = '#94a3b8';
    notifDiv.style.margin = '0.5rem 0';
    notifDiv.innerHTML = htmlText;
    chatStream.appendChild(notifDiv);
    chatStream.scrollTop = chatStream.scrollHeight;
  }
});
