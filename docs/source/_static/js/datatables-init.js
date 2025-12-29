(function () {
  function tagsArrayFromHtml(html) {
    if (html == null) return [];
    if (typeof html === 'number') return [String(html)];
    if (typeof html === 'string' && html.indexOf('<') === -1) {
      const trimmed = html.trim();
      return trimmed ? [trimmed] : [];
    }
    const div = document.createElement('div');
    div.innerHTML = html;

    // Prefer finding explicit tags first
    const tags = Array.from(div.querySelectorAll('.tag'))
      .map((el) => el.textContent.trim())
      .filter(Boolean);
    if (tags.length) {
      return [...new Set(tags)]; // Unique values
    }

    const text = (div.textContent || '').trim();
    if (!text) return [];

    // Fallback to splitting by common separators
    return text
      .split(/[,;|/\n\s]+/)
      .map((token) => token.trim())
      .filter(Boolean)
      .filter((token, index, self) => self.indexOf(token) === index);
  }

  function parseSizeToBytes(text) {
    if (!text) return 0;
    const match = String(text)
      .trim()
      .match(/([\d,.]+)\s*(TB|GB|MB|KB|B)/i);
    if (!match) return 0;
    const value = parseFloat(match[1].replace(/,/g, ''));
    const unit = match[2].toUpperCase();
    const factor = {
      B: 1,
      KB: 1024,
      MB: 1024 ** 2,
      GB: 1024 ** 3,
      TB: 1024 ** 4,
    }[unit] || 1;
    return value * factor;
  }

  function ensureTotalRowInFoot($table) {
    const $tbody = $table.find('tbody');
    const $rows = $tbody.find('tr');
    $rows.each(function () {
      const $row = $(this);
      const label = ($row.find('td').first().text() || '').trim().toLowerCase();
      if (!label.startsWith('total')) return;
      let $tfoot = $table.find('tfoot');
      if (!$tfoot.length) {
        $tfoot = $('<tfoot/>').appendTo($table);
      }
      $row.appendTo($tfoot);
    });
  }

  function initHeaderFilterShortcuts($table, dataTable, filterCols) {
    const headerCells = $table.find('thead th');
    headerCells.each(function (index) {
      if (!filterCols.includes(index)) return;
      $(this)
        .css('cursor', 'pointer')
        .attr('title', 'Click to filter this column')
        .off('click.eegdash')
        .on('click.eegdash', function () {
          dataTable.button('.buttons-searchPanes').trigger();
          setTimeout(function () {
            const container = $(dataTable.searchPanes.container());
            const pane = container.find('.dtsp-pane').eq(filterCols.indexOf(index));
            const title = pane.find('.dtsp-title');
            if (title.length) title.trigger('click');
          }, 0);
        });
    });
  }

  function applyTagPalette(target) {
    if (window.EEGDashTagPalette && typeof window.EEGDashTagPalette.apply === 'function') {
      window.EEGDashTagPalette.apply(target);
    }
  }

  function initModelsTable($) {
    const $table = $('#models-table');
    if (!$table.length || !$table.is(':visible')) return;
    if (!$.fn.DataTable) return;
    if ($.fn.DataTable.isDataTable($table)) {
      return;
    }

    const filterCols = [1, 2, 3, 6];

    const dataTable = $table.DataTable({
      dom: 'Blfrtip',
      paging: false,
      searching: true,
      info: false,
      language: {
        search: 'Filter models:',
        searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } },
      },
      buttons: [
        {
          extend: 'searchPanes',
          text: 'Filters',
          config: {
            cascadePanes: true,
            viewTotal: true,
            layout: 'columns-4',
            initCollapsed: false,
          },
        },
      ],
      columnDefs: [
        {
          targets: filterCols,
          render: { _: (d) => d, sp: (d) => tagsArrayFromHtml(d) },
          searchPanes: { show: true, orthogonal: 'sp' },
        },
      ],
      drawCallback: function (settings) {
        const wrapper = settings.nTableWrapper || document;
        applyTagPalette(wrapper);
      },
    });

    initHeaderFilterShortcuts($table, dataTable, filterCols);
    applyTagPalette($table.closest('.dataTables_wrapper').get(0) || document);
  }

  function initDatasetsTable($) {
    const $table = $('#datasets-table');
    if (!$table.length || !$table.is(':visible')) return;
    if (!$.fn.DataTable) return;
    if ($.fn.DataTable.isDataTable($table)) {
      return;
    }

    ensureTotalRowInFoot($table);

    const filterCols = [1, 2, 3, 4, 5];

    const headerCells = $table.find('thead th');
    const sizeIndex = headerCells
      .map(function (index) {
        const text = $(this).text().trim().toLowerCase();
        if (text === 'size on disk' || text === 'size') {
          return index;
        }
        return -1;
      })
      .toArray()
      .find((v) => v !== -1);

    const columnDefs = [
      {
        targets: filterCols,
        render: function (data, type, row) {
          if (type === 'sp' || type === 'filter' || type === 'search') {
            return tagsArrayFromHtml(data);
          }
          return data;
        },
        searchPanes: {
          show: true,
          orthogonal: 'sp',
        },
      },
    ];

    if (typeof sizeIndex === 'number') {
      columnDefs.push({
        targets: sizeIndex,
        render: function (data, type) {
          if (type === 'sort' || type === 'type') {
            return parseSizeToBytes(data);
          }
          return data;
        },
      });
    }

    const dataTable = $table.DataTable({
      dom: 'Blfrtip',
      paging: false,
      searching: true,
      info: false,
      language: {
        search: 'Filter dataset:',
        searchPanes: { collapse: { 0: 'Filters', _: 'Filters (%d)' } },
      },
      buttons: [
        {
          extend: 'searchPanes',
          text: 'Filters',
          config: {
            cascadePanes: true,
            viewTotal: true,
            layout: 'columns-4',
            initCollapsed: false,
          },
        },
      ],
      columnDefs,
      drawCallback: function (settings) {
        const wrapper = settings.nTableWrapper || document;
        applyTagPalette(wrapper);
      },
    });

    initHeaderFilterShortcuts($table, dataTable, filterCols);
    applyTagPalette($table.closest('.dataTables_wrapper').get(0) || document);
  }

  function initialiseTables() {
    if (typeof window === 'undefined' || !window.jQuery) {
      return;
    }
    const $ = window.jQuery;
    initModelsTable($);
    initDatasetsTable($);
  }

  if (typeof window !== 'undefined') {
    window.EEGDashTables = window.EEGDashTables || {};
    window.EEGDashTables.tagsArrayFromHtml = tagsArrayFromHtml;
  }

  if (typeof window !== 'undefined') {
    if (window.jQuery) {
      window.jQuery(initialiseTables);
    } else {
      document.addEventListener('DOMContentLoaded', initialiseTables);
    }
  }
})();
