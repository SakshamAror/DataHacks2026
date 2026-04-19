(function () {
  'use strict';

  let chart, areaSeries;

  function initChart() {
    const wrap      = document.getElementById('chart-wrap');
    const container = document.getElementById('chart');

    chart = LightweightCharts.createChart(container, {
      width:  wrap.clientWidth,
      height: wrap.clientHeight,
      layout: {
        background:  { type: 'solid', color: 'transparent' },
        textColor:   'rgba(255, 255, 255, 0.6)',
        fontFamily:  "'Inter', system-ui, sans-serif",
        fontSize:    12,
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {
          color: 'rgba(255, 255, 255, 0.25)',
          labelBackgroundColor: 'rgba(40, 40, 60, 0.85)',
        },
        horzLine: {
          color: 'rgba(255, 255, 255, 0.25)',
          labelBackgroundColor: 'rgba(40, 40, 60, 0.85)',
        },
      },
      rightPriceScale: {
        borderColor:  'rgba(255, 255, 255, 0.08)',
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        fixLeftEdge:    true,
        fixRightEdge:   true,
        rightOffset:    0,
        borderColor:    'rgba(255, 255, 255, 0.08)',
        timeVisible:    true,
        secondsVisible: false,
      },
      localization: {
        priceFormatter: function (p) {
          return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 0 });
        },
      },
      handleScroll: true,
      handleScale:  true,
    });

    areaSeries = chart.addAreaSeries({
      lineColor:   '#4ade80',
      topColor:    'rgba(74, 222, 128, 0.25)',
      bottomColor: 'rgba(74, 222, 128, 0.0)',
      lineWidth:   2,
    });

    new ResizeObserver(function () {
      chart.resize(wrap.clientWidth, wrap.clientHeight);
    }).observe(wrap);
  }

  // ── Metric helpers ──────────────────────────────────────────────────────────

  function el(id) { return document.getElementById(id); }

  function setVal(id, text, cls) {
    var e = el(id);
    e.textContent = text;
    e.className   = 'm-val' + (cls ? ' ' + cls : '');
  }

  function updateMetrics(m) {
    var pnlCls = m.total_pnl >= 0 ? 'positive' : 'negative';
    var pnlSign = m.total_pnl >= 0 ? '+$' : '-$';
    setVal('m-pnl', pnlSign + Math.abs(m.total_pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }), pnlCls);
    el('m-pnl-note').textContent = (m.return_pct >= 0 ? '+' : '') + m.return_pct.toFixed(2) + '% return';

    setVal('m-sharpe', m.sharpe_ratio.toFixed(2));

    setVal('m-drawdown', '\u2212' + m.max_drawdown_pct.toFixed(2) + '%', 'negative');
    el('m-drawdown-note').textContent = '\u2212$' + m.max_drawdown.toLocaleString('en-US', { maximumFractionDigits: 2 });

    setVal('m-winrate', m.win_rate.toFixed(1) + '%');
    el('m-settlements-note').textContent = m.total_settlements + ' settlements';

    setVal('m-trades', m.total_trades.toLocaleString());
    var avg = m.avg_trade_pnl;
    el('m-trades-note').textContent = 'Avg ' + (avg >= 0 ? '+' : '') + '$' + Math.abs(avg).toFixed(4);

    setVal('m-final', '$' + m.final_portfolio_value.toLocaleString('en-US', { maximumFractionDigits: 0 }));
    el('m-final-note').textContent = 'from $' + m.starting_cash.toLocaleString('en-US', { maximumFractionDigits: 0 });
  }

  // ── Run backtest ────────────────────────────────────────────────────────────

  window.runBacktest = function () {
    var btn     = el('run-btn');
    var overlay = el('chart-overlay');
    var errBar  = el('error-bar');

    btn.disabled = true;
    el('run-btn-text').textContent = 'Running\u2026';
    overlay.classList.remove('hidden');
    errBar.classList.add('hidden');

    fetch('/api/run-backtest', { method: 'POST' })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.error) throw new Error(data.error);

        if (data.series && data.series.length > 0) {
          areaSeries.setData(data.series);
          chart.timeScale().fitContent();
        }

        if (data.period && data.period.start !== 'N/A') {
          el('period-text').textContent =
            'my_strategy.py \u00b7 BTC \u00b7 5m \u00b7 ' +
            data.period.start + ' \u2013 ' + data.period.end;
        }

        updateMetrics(data.metrics);
      })
      .catch(function (err) {
        errBar.textContent = 'Backtest failed: ' + err.message;
        errBar.classList.remove('hidden');
      })
      .finally(function () {
        btn.disabled = false;
        el('run-btn-text').textContent = 'Run Backtest';
        overlay.classList.add('hidden');
      });
  };

  document.addEventListener('DOMContentLoaded', initChart);
})();
