/**
 * Plithos — Shared JS  v3.0
 */
document.addEventListener('DOMContentLoaded', function () {

  // ── Sidebar collapse ───────────────────────────────────────
  const shell       = document.querySelector('.shell');
  const collapseBtn = document.getElementById('collapseBtn');

  if (collapseBtn && shell) {
    collapseBtn.addEventListener('click', () => {
      shell.classList.toggle('collapsed');
      localStorage.setItem('sidebarCollapsed', shell.classList.contains('collapsed'));
    });
  }

  // Restore state
  if (localStorage.getItem('sidebarCollapsed') === 'true' && shell) {
    shell.classList.add('collapsed');
  }

  // ── Mobile sidebar ─────────────────────────────────────────
  const mobileBtn  = document.getElementById('mobileMenuBtn');
  const sidebar    = document.querySelector('.sidebar');
  const overlayBg  = document.getElementById('overlayBg');

  if (mobileBtn && sidebar) {
    mobileBtn.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
      if (overlayBg) overlayBg.classList.toggle('show');
    });
  }

  if (overlayBg && sidebar) {
    overlayBg.addEventListener('click', () => {
      sidebar.classList.remove('mobile-open');
      overlayBg.classList.remove('show');
    });
  }

  // ── Filter tabs ────────────────────────────────────────────
  document.querySelectorAll('.filter-tabs').forEach(group => {
    group.querySelectorAll('.filter-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        group.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
      });
    });
  });

  // ── Time filter buttons ────────────────────────────────────
  document.querySelectorAll('.time-filter').forEach(group => {
    group.querySelectorAll('.time-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        group.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });
  });

  // ── Alert list item selection ──────────────────────────────
  document.querySelectorAll('.alert-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.alert-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
    });
  });

  // ── Critical popup dismiss ─────────────────────────────────
  document.querySelectorAll('.popup-btn-dism').forEach(btn => {
    btn.addEventListener('click', () => {
      const popup = btn.closest('.critical-popup');
      if (popup) popup.style.display = 'none';
    });
  });

  // ── KPI value entrance animation ──────────────────────────
  document.querySelectorAll('.kpi-value').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(6px)';
    el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    setTimeout(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, 150 + i * 80);
  });

  // ── SOC live pulse ─────────────────────────────────────────
  const socBadge = document.querySelector('.soc-badge');
  if (socBadge) {
    socBadge.addEventListener('click', () => {
      socBadge.style.opacity = socBadge.style.opacity === '0.5' ? '1' : '0.5';
    });
  }

  // ── Live clock in footer ───────────────────────────────────
  const clockEl = document.getElementById('footerClock');
  if (clockEl) {
    function tick() {
      clockEl.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
    }
    tick();
    setInterval(tick, 1000);
  }

  // ── Resize handler ─────────────────────────────────────────
  window.addEventListener('resize', () => {
    if (window.innerWidth > 900 && sidebar) {
      sidebar.classList.remove('mobile-open');
      if (overlayBg) overlayBg.classList.remove('show');
    }
  });

});
